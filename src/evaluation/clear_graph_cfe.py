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
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import DenseGCNConv, DenseGraphConv

from gnn.utils import Metadata

from evaluation.metrics import (
    compute_proximity,
    evaluate_candidate,
    GraphEdit,
    matched_diff_to_edits,
)
from evaluation.utils import hetero_to_homogeneous_data, to_dense

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
        metadata: Metadata,
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
        data = hetero_to_homogeneous_data(graph, metadata)
        stats = self.stats
        dev = self.device

        features_orig, adj_orig, _ = to_dense(
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

                adj_cf_prob = output["adj_reconst"]  # (1, N, N)
                adj_cf = torch.bernoulli(adj_cf_prob)
                feat_cf = output["features_reconst"]  # (1, N, x_dim)

                edits = matched_diff_to_edits(
                    adj_orig[0],
                    adj_cf[0],
                    features_orig[0],
                    feat_cf[0],
                    self.adj_threshold,
                    self.feat_threshold,
                    self.graph_matching,
                )

                if len(edits) < best_count:
                    best_count = len(edits)
                    best_edits = edits

        return best_edits if best_edits is not None else []

    def explain_with_evaluation(
        self,
        graph: HeteroData | Data,
        metadata: Metadata,
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
        data = hetero_to_homogeneous_data(graph, metadata)
        stats = self.stats
        dev = self.device

        features_orig, adj_orig, _ = to_dense(
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

                adj_cf_prob = output["adj_reconst"]  # (1, N, N)
                adj_cf = torch.bernoulli(adj_cf_prob)
                feat_cf = output["features_reconst"]  # (1, N, x_dim)

                edits = matched_diff_to_edits(
                    adj_orig[0],
                    adj_cf[0],
                    features_orig[0],
                    feat_cf[0],
                    self.adj_threshold,
                    self.feat_threshold,
                    self.graph_matching,
                )

                if len(edits) < best_count:
                    best_count = len(edits)
                    best_edits = edits
                    best_feat_cf = feat_cf
                    best_adj_cf = adj_cf
                    # Compute proximity for this best candidate
                    best_proximity = compute_proximity(
                        features_orig[0],
                        adj_orig[0],
                        feat_cf[0],
                        adj_cf[0],
                        adj_cf_prob[0],
                    )

                    if evaluation_model is not None:
                        evaluation_results = evaluate_candidate(
                            graph,
                            metadata,
                            feat_cf[0],
                            adj_cf[0],
                            evaluation_model,
                            target_class,
                            adj_threshold=self.adj_threshold,
                        )
                        best_proximity.update(evaluation_results)

        return (
            best_feat_cf,
            best_adj_cf,
            best_edits if best_edits is not None else [],
            best_proximity if best_proximity is not None else {},
        )


# ---------------------------------------------------------------------------
# Functional entry-point  (mirrors other CF methods in the codebase)
# ---------------------------------------------------------------------------


def generate_graphcfe_counterfactual(
    graphcfe_model: torch.nn.Module,
    graph: HeteroData | Data,
    metadata: Metadata,
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
    return explainer.explain(graph, metadata, target_class, n_samples=n_samples)


def generate_graphcfe_counterfactual_with_evaluation(
    graphcfe_model: torch.nn.Module,
    graph: HeteroData | Data,
    metadata: Metadata,
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
        metadata,
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


# Build dense training tensors for GraphCFE and train the VAE on the available dataset.
def build_graphcfe_training_tensors(
    dataset, labels, stats, device, sample_size: Optional[int] = None
):
    # Sample from dataset if sample_size is specified
    if sample_size is not None:
        n_total = len(dataset)
        if sample_size > n_total:
            sample_size = n_total
        indices = torch.randperm(n_total, device="cpu")[:sample_size]
        dataset = [dataset[i] for i in indices]
        if isinstance(labels, torch.Tensor):
            labels = labels[indices]
        else:
            labels = [labels[i] for i in indices]

    n = len(dataset)

    features = torch.empty(
        n, stats.max_num_nodes, stats.x_dim, dtype=torch.float32, device=device
    )
    adjs = torch.empty(
        n,
        stats.max_num_nodes,
        stats.max_num_nodes,
        dtype=torch.float32,
        device=device,
    )
    ys = torch.empty(n, 1, dtype=torch.float32, device=device)
    us = torch.empty(n, 1, dtype=torch.float32, device=device)

    for i, (data, label) in enumerate(zip(dataset, labels, strict=False)):
        feat, adj, _ = to_dense(data, stats.max_num_nodes, stats.x_dim, device)
        features[i] = feat[0]
        adjs[i] = adj[0]
        ys[i] = float(label.item())

        num_edges = int(data.edge_index.shape[1])
        num_nodes = max(int(data.num_nodes), 1)
        u_value = float(num_edges / num_nodes)
        us[i] = u_value

    return features, adjs, ys, us


def train_graphcfe_model(
    graphcfe_model,
    features,
    adjs,
    ys,
    us,
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    device="cpu",
    patience=10,
    min_delta=1e-4,
):
    """Train GraphCFE model using CLEAR loss from https://github.com/jma712/GraphCFE.

    Parameters
    ----------
    graphcfe_model : torch.nn.Module
        The GraphCFE model to train.
    features : torch.Tensor
        Input feature tensor of shape (num_samples, max_num_nodes, x_dim).
    adjs : torch.Tensor
        Input adjacency tensor of shape (num_samples, max_num_nodes, max_num_nodes).
    ys : torch.Tensor
        Target labels of shape (num_samples, 1).
    us : torch.Tensor
        Confounding variable u of shape (num_samples, 1).
    num_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Learning rate for the Adam optimizer.
    device : str or torch.device
        Device to train on.
    patience : int
        Number of epochs to wait for improvement before early stopping.
    min_delta : float
        Minimum change in loss to qualify as an improvement.

    Returns
    -------
    graphcfe_model : torch.nn.Module
        The trained model (best state if early stopping triggered).
    """
    graphcfe_model.to(device)
    optimizer = torch.optim.Adam(graphcfe_model.parameters(), lr=learning_rate)
    num_samples = features.size(0)

    # Early stopping state
    best_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    def distance_feature(feat_1, feat_2):
        """Computes mean pairwise L2 distance between feature tensors."""
        pdist = nn.PairwiseDistance(p=2)
        output = pdist(feat_1, feat_2) / 4.0
        return torch.mean(output)

    def distance_graph_prob(adj_1, adj_2_prob):
        """Computes binary cross entropy distance between adjacency matrices."""
        return F.binary_cross_entropy(adj_2_prob, adj_1)

    for epoch in range(1, num_epochs + 1):
        graphcfe_model.train()
        perm = torch.randperm(num_samples)  # Create permutation on CPU
        total_loss = 0.0

        for start in range(0, num_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_x = features[batch_idx].to(device)
            batch_adj = adjs[batch_idx].to(device)
            batch_y = ys[batch_idx].to(device)
            batch_u = us[batch_idx].to(device)

            outputs = graphcfe_model(batch_x, batch_u, batch_adj, batch_y)
            recon_adj = outputs["adj_reconst"]
            recon_x = outputs["features_reconst"]
            z_mu = outputs["z_mu"]
            z_logvar = outputs["z_logvar"]
            z_u_mu = outputs["z_u_mu"]
            z_u_logvar = outputs["z_u_logvar"]

            # KL loss: divergence between encoder and prior distributions
            loss_kl = 0.5 * (
                (z_u_logvar - z_logvar)
                + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())
                - 1.0
            )
            loss_kl = torch.mean(loss_kl)

            # Similarity loss: weighted distance on features and adjacency
            batch_size_actual = batch_x.size(0)
            dist_x = distance_feature(
                batch_x.view(batch_size_actual, -1),
                recon_x.view(batch_size_actual, -1),
            )
            dist_a = distance_graph_prob(batch_adj, recon_adj)

            beta = 10.0
            loss_sim = beta * dist_x + 10.0 * dist_a

            # Total loss (without CFE loss since no classifier during training)
            loss = loss_sim + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size_actual

        avg_loss = total_loss / num_samples

        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_model_state = {
                k: v.cpu().clone() for k, v in graphcfe_model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch % 10 == 0
            or epoch == 1
            or epoch == num_epochs
            or epochs_without_improvement == 0
        ):
            print(
                f"[GraphCFE] epoch {epoch}/{num_epochs} "
                f"loss={avg_loss:.4f} "
                f"loss_sim={loss_sim.item():.4f} "
                f"loss_kl={loss_kl.item():.4f} "
                f"best={best_loss:.4f} "
                f"patience={epochs_without_improvement}/{patience}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"[GraphCFE] Early stopping at epoch {epoch}. "
                f"No improvement for {patience} epochs. Best loss: {best_loss:.4f}"
            )
            break

    # Restore best model state
    if best_model_state is not None:
        graphcfe_model.load_state_dict(best_model_state)
        graphcfe_model.to(device)

    graphcfe_model.eval()
    return graphcfe_model
