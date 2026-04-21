from __future__ import annotations

import dataclasses
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, HeteroData

from gnn.hetero_graph_data import to_homogeneous_data
from gnn.utils import Metadata

from evaluation.utils import match_nodes


# ---------------------------------------------------------------------------
# Metrics: CLEAR proximity and evaluation with classifier
# ---------------------------------------------------------------------------


def compute_proximity(
    feat_orig: torch.Tensor,  # (n, x_dim)
    adj_orig: torch.Tensor,  # (n, n)  binary
    feat_cf: torch.Tensor,  # (n, x_dim)
    adj_cf: torch.Tensor,  # (n, n)  binary
    adj_cf_prob: Optional[torch.Tensor] = None,  # (n, n)  raw sigmoid output
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
    adj_cf : (n, n) – counterfactual adjacency (binary)
    adj_cf_prob : (n, n) – counterfactual adjacency (probabilistic output), if not provided use adj_cf

    Returns
    -------
    dict with keys 'dist_x', 'dist_a', 'proximity'
    """
    device = feat_orig.device
    adj_cf_prob = adj_cf_prob if adj_cf_prob is not None else adj_cf

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
    dist_a = F.binary_cross_entropy(adj_cf_prob.float(), adj_orig.float())

    # Proximity
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    cos_sim = cos(feat_orig, feat_cf)
    proximity_x = torch.mean(cos_sim)

    proximity_a = (adj_orig == adj_cf).float().mean()

    return {
        "dist_x": dist_x.item(),
        "dist_a": dist_a.item(),
        "proximity_x": proximity_x.item(),
        "proximity_a": proximity_a.item(),
    }


def evaluate_candidate(
    graph: HeteroData | Data,
    metadata: Metadata,
    candidate_feat: torch.Tensor,
    candidate_adj: torch.Tensor,
    evaluation_model: torch.nn.Module,
    target_class: int,
    adj_threshold: float = 0.5,
) -> dict:
    """Evaluate the selected counterfactual candidate with a classifier."""
    device = candidate_feat.device

    # Convert to a homogeneous graph if needed, then override features and edges.
    if isinstance(graph, HeteroData):
        eval_data = to_homogeneous_data(
            graph,
            metadata.node_num_keys,
            metadata.node_cat_keys,
            metadata.node_types,
            metadata.one_hot_encoding,
            metadata.unique_node_type_attribute_columns,
        )
    else:
        eval_data = graph.clone()

    edge_index = (
        (candidate_adj > adj_threshold).nonzero(as_tuple=False).t().contiguous()
    )
    eval_data.x = candidate_feat
    eval_data.edge_index = edge_index.to(device)

    batch = torch.zeros(
        eval_data.num_nodes if eval_data.num_nodes else 0,
        dtype=torch.long,
        device=device,
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


def matched_diff_to_edits(
    adj_orig: torch.Tensor,  # (N, N)
    adj_cf: torch.Tensor,  # (N, N)  raw sigmoid output
    feat_orig: torch.Tensor,  # (N, x_dim)
    feat_cf: torch.Tensor,  # (N, x_dim)
    adj_threshold: float = 0.5,
    feat_threshold: float = 0.5,
    graph_matching: bool = False,
) -> List[GraphEdit]:
    """
    1. Restrict to the n_real real (non-padded) nodes plus any extra nodes
       present in the reconstructed graph.
    2. Binarise the reconstructed adjacency.
    3. Run graph matching (Hungarian) to align G' node indices to G.
    4. Diff the aligned matrices to produce GraphEdit objects.
    """
    # Use the maximum node size across original and reconstructed tensors so
    # that insertions / deletions beyond the original n_real region are not
    # ignored.
    n = max(
        adj_orig.size(0),
        adj_orig.size(1),
        adj_cf.size(0),
        adj_cf.size(1),
        feat_orig.size(0),
        feat_cf.size(0),
    )

    def _pad_adj(matrix: torch.Tensor, size: int) -> torch.Tensor:
        if matrix.size(0) == size and matrix.size(1) == size:
            return matrix
        return F.pad(
            matrix,
            (0, size - matrix.size(1), 0, size - matrix.size(0)),
            value=0.0,
        )

    def _pad_feat(matrix: torch.Tensor, size: int) -> torch.Tensor:
        if matrix.size(0) == size:
            return matrix
        return F.pad(matrix, (0, 0, 0, size - matrix.size(0)), value=0.0)

    A = _pad_adj(adj_orig[:n, :n], n)
    X = _pad_feat(feat_orig[:n], n)
    Acf = _pad_adj(adj_cf[:n, :n], n)
    Xcf = _pad_feat(feat_cf[:n], n)

    A = (A > adj_threshold).float()
    Acf = (Acf > adj_threshold).float()

    # Graph matching: find permutation π s.t. Acf[π][:,π] ≈ A
    if graph_matching:
        perm = match_nodes(A, X, Acf, Xcf)  # (n,)

        A_cf_aligned = Acf[perm][:, perm]  # (n, n)
        X_cf_aligned = Xcf[perm]  # (n, x_dim)
    else:
        A_cf_aligned = Acf
        X_cf_aligned = Xcf

    edits: List[GraphEdit] = []

    # Edge additions
    for i, j in (A_cf_aligned - A).clamp(min=0).nonzero(as_tuple=False).tolist():
        if i != j:
            edits.append(GraphEdit(edit_type="add_edge", src=i, dst=j))

    # Edge removals
    for i, j in (A - A_cf_aligned).clamp(min=0).nonzero(as_tuple=False).tolist():
        if i != j:
            edits.append(GraphEdit(edit_type="remove_edge", src=i, dst=j))

    # Node-feature changes
    diff = (X_cf_aligned - X).abs()
    for node_idx, feat_idx in (diff > feat_threshold).nonzero(as_tuple=False).tolist():
        edits.append(
            GraphEdit(
                edit_type="change_node_feat",
                node_idx=int(node_idx),
                feature_idx=int(feat_idx),
                old_value=float(X[node_idx, feat_idx].item()),
                new_value=float(X_cf_aligned[node_idx, feat_idx].item()),
            )
        )

    return edits


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
