from typing import Tuple

import torch
import torch.nn.functional as F
from networkx import Graph
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_dense_adj

from gnn.hetero_graph_data import to_homogeneous_data
from gnn.utils import Metadata, build_hetero_data


# ---------------------------------------------------------------------------
# Conversion helpers: HeteroData / Data  ↔  dense (padded) tensors
# ---------------------------------------------------------------------------


def hetero_to_homogeneous_data(graph: HeteroData | Data, metadata: Metadata) -> Data:
    if isinstance(graph, HeteroData):
        return to_homogeneous_data(
            graph,
            metadata.node_num_keys,
            metadata.node_cat_keys,
            metadata.node_types,
            metadata.one_hot_encoding,
            metadata.unique_node_type_attribute_columns,
        )
    return graph


def to_dense(
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


def match_nodes(
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


def get_dense_representation(
    process_execution: Graph,
    metadata: Metadata,
    object_type_column: str,
    event_activity: str,
    device: str,
):
    hetero_data, _, _, _, _, _ = build_hetero_data(
        graph=process_execution,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=object_type_column,
        event_activity_col=event_activity,
        viewpoint=metadata.viewpoint,
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
        add_reverse_edges=metadata.add_reverse_edges,
    )
    data = to_homogeneous_data(
        hetero_data,
        metadata.node_num_keys,
        metadata.node_cat_keys,
        metadata.node_types,
        metadata.one_hot_encoding,
        metadata.unique_node_type_attribute_columns,
    )
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
    return data.x.to(device), adj, data.num_nodes


def pad_dense_tensors(
    features_a: torch.Tensor,
    adj_a: torch.Tensor,
    features_b: torch.Tensor,
    adj_b: torch.Tensor,
):
    n = max(features_a.size(0), features_b.size(0))

    if features_a.size(0) < n:
        features_a = F.pad(features_a, (0, 0, 0, n - features_a.size(0)))
    if features_b.size(0) < n:
        features_b = F.pad(features_b, (0, 0, 0, n - features_b.size(0)))
    if adj_a.size(0) < n:
        adj_a = F.pad(adj_a, (0, n - adj_a.size(1), 0, n - adj_a.size(0)))
    if adj_b.size(0) < n:
        adj_b = F.pad(adj_b, (0, n - adj_b.size(1), 0, n - adj_b.size(0)))

    return features_a, adj_a, features_b, adj_b
