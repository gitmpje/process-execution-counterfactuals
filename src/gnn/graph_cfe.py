"""
GraphCFE / CLEAR counterfactual explanation method.

Reference:
    Jing Ma et al. "CLEAR: Generative Counterfactual Explanations on Graphs."
    NeurIPS 2022.  https://github.com/jma712/GraphCFE

The GraphCFE model is a graph VAE that is trained *alongside* (or after)
the downstream GNN classifier.  At inference time it:
  1. Encodes the input graph to a latent vector z conditioned on the desired
     counterfactual label y_cf and on a confounding variable u.
  2. Decodes z back to a new adjacency matrix A' and node-feature matrix X'.
  3. Runs graph matching between G and G' to establish a node correspondence
     (the VAE decoder does not preserve vertex order).
  4. Compares the matched (A', X') with the original (A, X) to derive a
     minimal list of graph edits.

Because the model was trained on *dense* (padded) graphs of fixed size
`max_num_nodes`, the training dataset must be available so that:
  * `max_num_nodes` can be inferred (or set explicitly).
  * The confounding variable u can be estimated as the class-mean of the
    training graphs (a simple but faithful replication of the paper).

Public surface
--------------
GraphCFEExplainer                – wraps the trained model + dataset
                                   statistics; exposes `explain()`.
generate_graphcfe_counterfactual – functional entry-point (mirrors other
                                   counterfactual methods in the codebase).
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from numbers import Number
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import DenseGCNConv, DenseGraphConv
from torch_geometric.utils import to_dense_adj


# ---------------------------------------------------------------------------
# Graph-edit representation (same pattern as other CF methods in codebase)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GraphEdit:
    """A single modification to apply to a graph."""

    edit_type: str  # "add_edge", "remove_edge", "change_node_feat"
    # For edge edits:
    src: Optional[int] = None
    dst: Optional[int] = None
    # For node-feature edits:
    node_idx: Optional[int] = None
    feature_idx: Optional[int] = None
    old_value: Optional[float] = None
    new_value: Optional[float] = None

    def __repr__(self) -> str:
        if self.edit_type == "add_edge":
            return f"AddEdge({self.src} → {self.dst})"
        if self.edit_type == "remove_edge":
            return f"RemoveEdge({self.src} → {self.dst})"
        if self.edit_type == "change_node_feat":
            return (
                f"ChangeNodeFeat(node={self.node_idx}, "
                f"feat={self.feature_idx}, "
                f"{self.old_value:.3f} → {self.new_value:.3f})"
            )
        return f"GraphEdit({self.edit_type})"


# ---------------------------------------------------------------------------
# Dataset-statistics helper
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GraphCFEDatasetStats:
    """
    Summary statistics derived from the training set that GraphCFE needs at
    inference time.

    Parameters
    ----------
    max_num_nodes : int
        Padding target.  All graphs are zero-padded / truncated to this size.
    u_mean_class0 : torch.Tensor  shape (1, 1)
        Mean of the scalar confound u estimated from class-0 training graphs.
    u_mean_class1 : torch.Tensor  shape (1, 1)
        Mean of the scalar confound u estimated from class-1 training graphs.
    x_dim : int
        Node-feature dimensionality.
    """

    max_num_nodes: int
    u_mean_class0: torch.Tensor
    u_mean_class1: torch.Tensor
    x_dim: int

    @classmethod
    def from_dataset(
        cls,
        dataset: List[Data],
        labels: Optional[torch.Tensor] = None,
    ) -> "GraphCFEDatasetStats":
        """
        Compute statistics from a list of PyG Data objects (homogeneous).

        Parameters
        ----------
        dataset : list of torch_geometric.data.Data
        labels  : optional 1-D integer tensor with labels (0 or 1).
                  If None, ``data.y`` is used for each graph.
        """
        num_nodes_list = []
        feat_dims: set = set()

        for data in dataset:
            num_nodes_list.append(data.num_nodes)
            if data.x is not None:
                feat_dims.add(data.x.shape[1])

        max_num_nodes = int(max(num_nodes_list))
        x_dim = int(feat_dims.pop()) if len(feat_dims) == 1 else 1

        # Estimate u per class as mean node-degree (a simple proxy;
        # replace with a domain-specific confound if available).
        u0_vals, u1_vals = [], []
        for i, data in enumerate(dataset):
            label = int(data.y.item()) if labels is None else int(labels[i].item())
            deg = data.edge_index.shape[1] / max(data.num_nodes, 1)
            (u0_vals if label == 0 else u1_vals).append(deg)

        u0 = torch.tensor([[float(sum(u0_vals) / max(len(u0_vals), 1))]])
        u1 = torch.tensor([[float(sum(u1_vals) / max(len(u1_vals), 1))]])

        return cls(
            max_num_nodes=max_num_nodes,
            u_mean_class0=u0,
            u_mean_class1=u1,
            x_dim=x_dim,
        )


# ---------------------------------------------------------------------------
# Conversion helpers: HeteroData / Data  ↔  dense (padded) tensors
# ---------------------------------------------------------------------------


def _hetero_to_homogeneous_data(graph: HeteroData | Data) -> Data:
    if isinstance(graph, HeteroData):
        return graph.to_homogeneous()
    return graph


def _to_dense(
    data: Data,
    max_num_nodes: int,
    x_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convert a PyG Data object to dense (padded) adjacency + feature tensors.

    Returns
    -------
    features : (1, max_num_nodes, x_dim)
    adj      : (1, max_num_nodes, max_num_nodes)
    n_real   : number of real (non-padded) nodes
    """
    n_real = min(data.num_nodes, max_num_nodes)

    adj_dense = to_dense_adj(
        data.edge_index,
        max_num_nodes=max_num_nodes,
    )  # (1, max_num_nodes, max_num_nodes)

    if data.x is not None:
        x = data.x[:n_real]
        if x.shape[1] != x_dim:
            diff = x_dim - x.shape[1]
            x = F.pad(x, (0, diff)) if diff > 0 else x[:, :x_dim]
    else:
        x = torch.zeros(n_real, x_dim)

    pad_nodes = max_num_nodes - n_real
    if pad_nodes > 0:
        x = F.pad(x, (0, 0, 0, pad_nodes))

    features = x.unsqueeze(0).to(device)  # (1, max_num_nodes, x_dim)
    adj_dense = adj_dense.to(device)
    return features, adj_dense, n_real


# ---------------------------------------------------------------------------
# Graph matching
# ---------------------------------------------------------------------------


def _match_nodes(
    adj_orig: torch.Tensor,  # (n, n)  binary
    feat_orig: torch.Tensor,  # (n, x_dim)
    adj_cf: torch.Tensor,  # (n, n)  binary
    feat_cf: torch.Tensor,  # (n, x_dim)
) -> torch.Tensor:
    """
    Find the optimal bijection π : V(G') → V(G) that minimises the graph-edit
    distance proxy between G and π(G').

    We use a linear-sum-assignment (Hungarian algorithm) on a cost matrix
    whose entry C[i, j] measures how dissimilar node i of G is to node j of
    G' when we also account for their structural neighbourhoods:

        C[i, j] = ‖feat_orig[i] − feat_cf[j]‖₁
                + ‖adj_orig[i, :] − adj_cf[j, :]‖₁

    Returns
    -------
    perm : 1-D LongTensor of length n
        perm[i] = j means node j in G' corresponds to node i in G.
        Reorder G' before diffing:
            adj_cf_aligned  = adj_cf[perm][:, perm]
            feat_cf_aligned = feat_cf[perm]
    """
    # Feature cost  C_feat[i, j] = L1(feat_orig[i], feat_cf[j])
    feat_cost = torch.cdist(feat_orig.float(), feat_cf.float(), p=1)  # (n, n)
    # Structural cost  C_struct[i, j] = L1(adj_orig[i,:], adj_cf[j,:])
    adj_cost = torch.cdist(adj_orig.float(), adj_cf.float(), p=1)  # (n, n)

    cost = feat_cost + adj_cost  # (n, n)

    _, col_ind = linear_sum_assignment(cost.cpu().numpy())
    # col_ind[i] = index in G' that is matched to node i in G
    return torch.tensor(col_ind, dtype=torch.long)


def _compute_proximity(
    feat_orig: torch.Tensor,  # (n, x_dim)
    adj_orig: torch.Tensor,  # (n, n)  binary
    feat_cf: torch.Tensor,  # (n, x_dim)
    adj_cf: torch.Tensor,  # (n, n)  raw sigmoid output
) -> dict:
    """
    Compute proximity metrics between original and counterfactual graphs.

    Follows the CLEAR paper (GraphCFE) definition of proximity:
    - dist_x: mean pairwise L2 distance on flattened features / 4.0
    - dist_a: binary cross entropy on adjacency matrices
    - proximity: combined score (dist_x + dist_a)

    Parameters
    ----------
    feat_orig : (n, x_dim) – original node features
    adj_orig : (n, n) – original adjacency (binary)
    feat_cf : (n, x_dim) – counterfactual node features
    adj_cf : (n, n) – counterfactual adjacency (probabilistic output)

    Returns
    -------
    dict with keys 'dist_x', 'dist_a', 'proximity'
    """
    # Feature distance: pairwise L2 distance
    n = feat_orig.shape[0]
    if n > 0:
        pdist = nn.PairwiseDistance(p=2)
        feat_cf_reshaped = feat_cf.view(n, -1)
        feat_orig_reshaped = feat_orig.view(n, -1)
        dist_x = pdist(feat_orig_reshaped, feat_cf_reshaped) / 4.0
        dist_x = torch.mean(dist_x)
    else:
        dist_x = torch.tensor(0.0, device=device)

    # Adjacency distance: binary cross entropy
    dist_a = F.binary_cross_entropy(adj_cf.float(), adj_orig.float())

    # Proximity
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    cos_sim = cos(adj_orig, adj_cf)
    proximity_x = torch.mean(cos_sim)

    proximity_a = (adj_orig == torch.bernoulli(adj_cf)).float().mean()

    return {
        "dist_x": dist_x.item(),
        "dist_a": dist_a.item(),
        "dist_score": (dist_x + dist_a).item(),
        "proximity_x": proximity_x.item(),
        "proximity_a": proximity_a.item(),
    }


# ---------------------------------------------------------------------------
# Core explainer class
# ---------------------------------------------------------------------------


class GraphCFEExplainer:
    """
    Wraps a trained GraphCFE (CLEAR) model and exposes a clean ``explain()``
    method that returns graph edits.

    Parameters
    ----------
    graphcfe_model : GraphCFE  (from models.py / CLEAR repo)
        Trained VAE counterfactual generator; put into eval mode automatically.
    dataset_stats : GraphCFEDatasetStats
        Pre-computed statistics from the training set.  Build once with
        ``GraphCFEDatasetStats.from_dataset(train_dataset)`` and reuse.
    adj_threshold : float
        Binarisation threshold for the reconstructed (sigmoid) adjacency.
    feat_threshold : float
        Minimum absolute node-feature delta to register as an edit.
        Default 0.5 suits binary features; lower for continuous ones.
    device : str or torch.device
        Inference device; defaults to GPU when available.
    disable_u : bool
        Must match the flag used when training the GraphCFE model.
        When True the confounding variable u is ignored.
    """

    def __init__(
        self,
        graphcfe_model: torch.nn.Module,
        dataset_stats: GraphCFEDatasetStats,
        adj_threshold: float = 0.5,
        feat_threshold: float = 0.5,
        device: Optional[torch.device | str] = None,
        disable_u: bool = False,
        graph_matching: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = graphcfe_model.to(self.device).eval()
        self.stats = dataset_stats
        self.adj_threshold = adj_threshold
        self.feat_threshold = feat_threshold
        self.disable_u = disable_u
        self.graph_matching = graph_matching

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        graph: HeteroData | Data,
        target_class: int,
        n_samples: int = 1,
    ) -> List[GraphEdit]:
        """
        Generate a counterfactual explanation for ``graph`` targeting
        ``target_class``.

        Parameters
        ----------
        graph        : HeteroData or Data object (single graph, no batch dim).
        target_class : 0 or 1 – the desired prediction class.
        n_samples    : number of latent samples to draw; the candidate with
                       the fewest edits is returned.  Increase for diversity.

        Returns
        -------
        List[GraphEdit] – ordered edits that transform ``graph`` into the
        counterfactual.  Apply them in sequence to reproduce G'.
        """
        data = _hetero_to_homogeneous_data(graph)
        stats = self.stats
        dev = self.device

        features_orig, adj_orig, n_real = _to_dense(
            data, stats.max_num_nodes, stats.x_dim, dev
        )

        y_cf = torch.tensor([[float(target_class)]], device=dev)  # (1, 1)

        if self.disable_u:
            u = torch.zeros(1, 0, device=dev)
        else:
            u_source = stats.u_mean_class1 if target_class == 1 else stats.u_mean_class0
            u = u_source.to(dev)  # (1, 1)

        best_edits: Optional[List[GraphEdit]] = None
        best_count = float("inf")

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(features_orig, u, adj_orig, y_cf)

                adj_cf = output["adj_reconst"]  # (1, N, N)
                feat_cf = output["features_reconst"]  # (1, N, x_dim)

                edits = self._matched_diff_to_edits(
                    adj_orig[0],
                    adj_cf[0],
                    features_orig[0],
                    feat_cf[0],
                    n_real,
                )

                if len(edits) < best_count:
                    best_count = len(edits)
                    best_edits = edits

        return best_edits if best_edits is not None else []

    def explain_with_evaluation(
        self,
        graph: HeteroData | Data,
        target_class: int,
        n_samples: int = 1,
        evaluation_model: Optional[torch.nn.Module] = None,
    ) -> Tuple[List[GraphEdit], dict]:
        """
        Generate a counterfactual explanation for ``graph`` targeting
        ``target_class``, returning both edits and proximity metrics.

        Parameters
        ----------
        graph        : HeteroData or Data object (single graph, no batch dim).
        target_class : 0 or 1 – the desired prediction class.
        n_samples    : number of latent samples to draw; the candidate with
                       the fewest edits is returned.
        evaluation_model : optional classifier used to evaluate whether the
                           generated counterfactual indeed predicts the
                           desired target class.

        Returns
        -------
        edits : List[GraphEdit]
            Ordered edits that transform ``graph`` into the counterfactual.
        proximity : dict
            Dictionary with keys:
            - 'dist_x': feature distance (L2 pairwise / 4.0)
            - 'dist_a': adjacency distance (binary cross entropy)
            - 'proximity': combined proximity score (dist_x + dist_a)
            - 'evaluation_prediction': predicted class from the evaluation model
              for the selected counterfactual (if ``evaluation_model`` is set).
            - 'evaluation_valid': ``True`` if the selected counterfactual is
              predicted as ``target_class`` by ``evaluation_model``.
        """
        data = _hetero_to_homogeneous_data(graph)
        stats = self.stats
        dev = self.device

        features_orig, adj_orig, n_real = _to_dense(
            data, stats.max_num_nodes, stats.x_dim, dev
        )

        y_cf = torch.tensor([[float(target_class)]], device=dev)  # (1, 1)

        if self.disable_u:
            u = torch.zeros(1, 0, device=dev)
        else:
            u_source = stats.u_mean_class1 if target_class == 1 else stats.u_mean_class0
            u = u_source.to(dev)  # (1, 1)

        best_edits: Optional[List[GraphEdit]] = None
        best_count = float("inf")
        best_proximity: Optional[dict] = None

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(features_orig, u, adj_orig, y_cf)

                adj_cf = output["adj_reconst"]  # (1, N, N)
                feat_cf = output["features_reconst"]  # (1, N, x_dim)

                edits = self._matched_diff_to_edits(
                    adj_orig[0],
                    adj_cf[0],
                    features_orig[0],
                    feat_cf[0],
                    n_real,
                )

                if len(edits) < best_count:
                    best_count = len(edits)
                    best_edits = edits
                    # Compute proximity for this best candidate
                    best_proximity = _compute_proximity(
                        features_orig[0, :n_real],
                        adj_orig[0, :n_real, :n_real],
                        feat_cf[0, :n_real],
                        adj_cf[0, :n_real, :n_real],
                    )

                    if evaluation_model is not None:
                        evaluation_results = self._evaluate_candidate(
                            graph,
                            feat_cf[0, :n_real],
                            adj_cf[0, :n_real, :n_real],
                            evaluation_model,
                            target_class,
                        )
                        best_proximity.update(evaluation_results)

        return (
            best_edits if best_edits is not None else [],
            best_proximity if best_proximity is not None else {},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        graph: HeteroData | Data,
        candidate_feat: torch.Tensor,
        candidate_adj: torch.Tensor,
        evaluation_model: torch.nn.Module,
        target_class: int,
    ) -> dict:
        """Evaluate the selected counterfactual candidate with a classifier."""
        # Convert to a homogeneous graph if needed, then override features and edges.
        if isinstance(graph, HeteroData):
            eval_data = graph.to_homogeneous()
        else:
            eval_data = graph.clone()

        edge_index = (
            (candidate_adj > self.adj_threshold)
            .nonzero(as_tuple=False)
            .t()
            .contiguous()
        )
        eval_data.x = candidate_feat
        eval_data.edge_index = edge_index.to(self.device)

        batch = torch.zeros(
            eval_data.num_nodes if eval_data.num_nodes else 0,
            dtype=torch.long,
            device=self.device,
        )

        try:
            out = evaluation_model(eval_data.x, eval_data.edge_index, batch)
        except TypeError:
            try:
                out = evaluation_model(eval_data)
            except Exception as exc:
                raise ValueError(
                    "evaluation_model must accept either (x, edge_index, batch) "
                    "or a single PyG Data object."
                ) from exc

        prediction = out.argmax(dim=-1)
        if prediction.numel() > 1:
            prediction = prediction[0]
        prediction = int(prediction.cpu().item())

        return {
            "evaluation_prediction": prediction,
            "evaluation_valid": prediction == target_class,
        }

    def _matched_diff_to_edits(
        self,
        adj_orig: torch.Tensor,  # (N, N)
        adj_cf: torch.Tensor,  # (N, N)  raw sigmoid output
        feat_orig: torch.Tensor,  # (N, x_dim)
        feat_cf: torch.Tensor,  # (N, x_dim)
        n_real: int,
    ) -> List[GraphEdit]:
        """
        1. Restrict to the n_real real (non-padded) nodes.
        2. Binarise the reconstructed adjacency.
        3. Run graph matching (Hungarian) to align G' node indices to G.
        4. Diff the aligned matrices to produce GraphEdit objects.
        """
        # Restrict to real nodes and binarise
        A = (adj_orig[:n_real, :n_real] > self.adj_threshold).float()
        Xo = feat_orig[:n_real]  # (n, x_dim)
        Acf = (adj_cf[:n_real, :n_real] > self.adj_threshold).float()
        Xcf = feat_cf[:n_real]

        # Graph matching: find permutation π s.t. Acf[π][:,π] ≈ A
        if self.graph_matching:
            perm = _match_nodes(A, Xo, Acf, Xcf)  # (n,)

            A_cf_aligned = Acf[perm][:, perm]  # (n, n)
            X_cf_aligned = Xcf[perm]  # (n, x_dim)
        else:
            A_cf_aligned = Acf
            X_cf_aligned = Xcf

        return self._diff_to_edits(A, A_cf_aligned, Xo, X_cf_aligned)

    def _diff_to_edits(
        self,
        adj_orig: torch.Tensor,  # (n, n)  binary
        adj_cf_aligned: torch.Tensor,  # (n, n)  binary, node-matched
        feat_orig: torch.Tensor,  # (n, x_dim)
        feat_cf_aligned: torch.Tensor,  # (n, x_dim)
    ) -> List[GraphEdit]:
        edits: List[GraphEdit] = []

        # Edge additions
        for i, j in (
            (adj_cf_aligned - adj_orig).clamp(min=0).nonzero(as_tuple=False).tolist()
        ):
            if i != j:
                edits.append(GraphEdit(edit_type="add_edge", src=i, dst=j))

        # Edge removals
        for i, j in (
            (adj_orig - adj_cf_aligned).clamp(min=0).nonzero(as_tuple=False).tolist()
        ):
            if i != j:
                edits.append(GraphEdit(edit_type="remove_edge", src=i, dst=j))

        # Node-feature changes
        diff = (feat_cf_aligned - feat_orig).abs()
        for node_idx, feat_idx in (
            (diff > self.feat_threshold).nonzero(as_tuple=False).tolist()
        ):
            edits.append(
                GraphEdit(
                    edit_type="change_node_feat",
                    node_idx=int(node_idx),
                    feature_idx=int(feat_idx),
                    old_value=float(feat_orig[node_idx, feat_idx].item()),
                    new_value=float(feat_cf_aligned[node_idx, feat_idx].item()),
                )
            )

        return edits


# ---------------------------------------------------------------------------
# Functional entry-point  (mirrors other CF methods in the codebase)
# ---------------------------------------------------------------------------


def generate_graphcfe_counterfactual(
    graphcfe_model: torch.nn.Module,
    graph: HeteroData | Data,
    target_class: int,
    dataset_stats: GraphCFEDatasetStats,
    adj_threshold: float = 0.5,
    feat_threshold: float = 0.5,
    n_samples: int = 1,
    device: Optional[torch.device | str] = None,
    disable_u: bool = False,
    graph_matching: bool = False,
) -> List[GraphEdit]:
    """
    Generate a list of graph edits that produce a counterfactual for ``graph``
    with ``target_class`` as the predicted label, using a trained GraphCFE model.

    Parameters
    ----------
    graphcfe_model  : trained GraphCFE instance (from models.py / CLEAR repo).
    graph           : single HeteroData or Data object to explain.
    target_class    : desired prediction class (0 or 1).
    dataset_stats   : ``GraphCFEDatasetStats`` computed from training data.
                      Build once with ``GraphCFEDatasetStats.from_dataset()``.
    adj_threshold   : binarisation threshold for reconstructed adjacency.
    feat_threshold  : minimum absolute node-feature delta to register as edit.
    n_samples       : number of latent samples; best (fewest edits) is kept.
    device          : inference device.
    disable_u       : must match the training flag on the GraphCFE model.

    Returns
    -------
    List[GraphEdit]
        Ordered sequence of graph modifications.  An empty list means the
        model produced a reconstruction identical to the input.

    Example
    -------
    >>> stats = GraphCFEDatasetStats.from_dataset(train_dataset)
    >>> edits = generate_graphcfe_counterfactual(
    ...     graphcfe_model=cfe_model,
    ...     graph=hetero_graph,
    ...     target_class=1,
    ...     dataset_stats=stats,
    ... )
    >>> for e in edits:
    ...     print(e)
    AddEdge(2 → 5)
    RemoveEdge(0 → 3)
    ChangeNodeFeat(node=1, feat=0, 0.000 → 1.000)
    """
    explainer = GraphCFEExplainer(
        graphcfe_model=graphcfe_model,
        dataset_stats=dataset_stats,
        adj_threshold=adj_threshold,
        feat_threshold=feat_threshold,
        device=device,
        disable_u=disable_u,
        graph_matching=graph_matching,
    )
    return explainer.explain(graph, target_class, n_samples=n_samples)


def generate_graphcfe_counterfactual_with_evaluation(
    graphcfe_model: torch.nn.Module,
    graph: HeteroData | Data,
    target_class: int,
    dataset_stats: GraphCFEDatasetStats,
    adj_threshold: float = 0.5,
    feat_threshold: float = 0.5,
    n_samples: int = 1,
    device: Optional[torch.device | str] = None,
    disable_u: bool = False,
    graph_matching: bool = False,
    evaluation_model: Optional[torch.nn.Module] = None,
) -> Tuple[List[GraphEdit], dict]:
    """
    Generate a counterfactual explanation with proximity metrics.

    Returns both the ordered list of graph edits and proximity statistics
    between the original and counterfactual graphs.

    Parameters
    ----------
    graphcfe_model  : trained GraphCFE instance.
    graph           : single HeteroData or Data object to explain.
    target_class    : desired prediction class (0 or 1).
    dataset_stats   : GraphCFEDatasetStats computed from training data.
    adj_threshold   : binarisation threshold for reconstructed adjacency.
    feat_threshold  : minimum node-feature delta to register as edit.
    n_samples       : number of latent samples; best (fewest edits) is kept.
    device          : inference device.
    disable_u       : must match the training flag on the GraphCFE model.
    graph_matching  : enable Hungarian algorithm for node alignment.
    evaluation_model : optional classifier used to verify whether the best
                       counterfactual candidate predicts ``target_class``.

    Returns
    -------
    Tuple[List[GraphEdit], dict]
        - edits: ordered list of graph modifications
        - proximity: dict with keys 'dist_x', 'dist_a', 'proximity', and if
          ``evaluation_model`` is provided, also 'evaluation_prediction' and
          'evaluation_valid'.
    """
    explainer = GraphCFEExplainer(
        graphcfe_model=graphcfe_model,
        dataset_stats=dataset_stats,
        adj_threshold=adj_threshold,
        feat_threshold=feat_threshold,
        device=device,
        disable_u=disable_u,
        graph_matching=graph_matching,
    )
    return explainer.explain_with_evaluation(
        graph,
        target_class,
        n_samples=n_samples,
        evaluation_model=evaluation_model,
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphCFE(nn.Module):
    def __init__(self, init_params, args):
        super(GraphCFE, self).__init__()
        self.vae_type = init_params["vae_type"]  # graphVAE
        self.x_dim = init_params["x_dim"]
        self.h_dim = args.dim_h
        self.z_dim = args.dim_z
        self.u_dim = 1  # init_params['u_dim']
        self.dropout = args.dropout
        self.max_num_nodes = init_params["max_num_nodes"]
        self.encoder_type = "gcn"
        self.graph_pool_type = "mean"
        self.disable_u = args.disable_u

        if self.disable_u:
            self.u_dim = 0
            print("disable u!")
        if self.encoder_type == "gcn":
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == "graphConv":
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)

        # prior
        self.prior_mean = MLP(
            self.u_dim,
            self.z_dim,
            self.h_dim,
            n_layers=1,
            activation="none",
            slope=0.1,
            device=device,
        )
        self.prior_var = nn.Sequential(
            MLP(
                self.u_dim,
                self.z_dim,
                self.h_dim,
                n_layers=1,
                activation="none",
                slope=0.1,
                device=device,
            ),
            nn.Sigmoid(),
        )

        # encoder
        self.encoder_mean = nn.Sequential(
            nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU(),
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU(),
            nn.Sigmoid(),
        )

        # decoder
        self.decoder_x = nn.Sequential(
            nn.Linear(self.z_dim + 1, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.max_num_nodes * self.x_dim),
        )
        self.decoder_a = nn.Sequential(
            nn.Linear(self.z_dim + 1, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.max_num_nodes * self.max_num_nodes),
            nn.Sigmoid(),
        )
        self.graph_norm = nn.BatchNorm1d(self.h_dim)

    def encoder(self, features, u, adj, y_cf):
        # Q(Z|X,U,A,Y^CF)
        # input: x, u, A, y^cf
        # output: z
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim
        # graph_rep = self.graph_norm(graph_rep)

        if self.disable_u:
            z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
        else:
            z_mu = self.encoder_mean(torch.cat((graph_rep, u, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, u, y_cf), dim=1))

        return z_mu, z_logvar

    def get_represent(self, features, u, adj, y_cf):
        u_onehot = u
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)

        return z_mu, z_logvar

    def decoder(self, z, y_cf, u):
        if self.disable_u:
            adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(
                -1, self.max_num_nodes, self.max_num_nodes
            )
        else:
            adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(
                -1, self.max_num_nodes, self.max_num_nodes
            )

        features_reconst = self.decoder_x(torch.cat((z, y_cf), dim=1)).view(
            -1, self.max_num_nodes, self.x_dim
        )
        return features_reconst, adj_reconst

    def graph_pooling(self, x, type="mean"):
        if type == "max":
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == "sum":
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == "mean":
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def prior_params(self, u):  # P(Z|U)
        if self.disable_u:
            z_u_mu = torch.zeros((len(u), self.h_dim)).to(device)
            z_u_logvar = torch.ones((len(u), self.h_dim)).to(device)
        else:
            z_u_logvar = self.prior_var(u)
            z_u_mu = self.prior_mean(u)
        return z_u_mu, z_u_logvar

    def reparameterize(self, mu, logvar):
        """
        compute z = mu + std * epsilon
        """
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def score(self):
        return

    def forward(self, features, u, adj, y_cf):
        u_onehot = u

        z_u_mu, z_u_logvar = self.prior_params(u_onehot)
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)
        # reparameterize
        z_sample = self.reparameterize(z_mu, z_logvar)
        # decoder
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf, u_onehot)

        return {
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "adj_permuted": adj,
            "features_permuted": features,
            "adj_reconst": adj_reconst,
            "features_reconst": features_reconst,
            "z_u_mu": z_u_mu,
            "z_u_logvar": z_u_logvar,
        }


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        n_layers,
        activation="none",
        slope=0.1,
        device="cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError(
                "Wrong argument type for hidden_dim: {}".format(hidden_dim)
            )

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError(
                "Wrong argument type for activation: {}".format(activation)
            )

        self._act_f = []
        for act in self.activation:
            if act == "lrelu":
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == "xtanh":
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == "sigmoid":
                self._act_f.append(F.sigmoid)
            elif act == "none":
                self._act_f.append(lambda x: x)
            else:
                ValueError("Incorrect activation: {}".format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(
                nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim)
            )
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=0.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h
