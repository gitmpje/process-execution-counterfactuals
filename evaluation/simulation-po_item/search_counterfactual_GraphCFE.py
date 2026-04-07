# %% Import dependencies
import json
import os
import pm4py
import torch
import yaml

from copy import deepcopy
from random import seed
from types import SimpleNamespace

import torch.nn.functional as F
from torch_geometric.data import HeteroData

import torch.nn as nn

from gnn.graph_cfe import (
    GraphCFE,
    generate_graphcfe_counterfactual,
    generate_graphcfe_counterfactual_with_proximity,
    GraphCFEDatasetStats,
    _to_dense,
)

from gnn.hetero_graph_data import build_hetero_data
from gnn.utils import Metadata
from process_execution.process_execution import extract_process_execution

from utils import _replace_scenario_prefix

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Replace $SCENARIO_PREFIX tokens in config
SCENARIO_PREFIX = os.environ.get("SCENARIO_PREFIX", "scenario_01")
if SCENARIO_PREFIX is not None:
    cfg = _replace_scenario_prefix(cfg, SCENARIO_PREFIX)

# Dataset
dataset_cfg = cfg["dataset"]
path_ocel = dataset_cfg["path_ocel"]
path_dataset = dataset_cfg["path_dataset"]
path_labels = dataset_cfg["path_labels"]
path_metadata = dataset_cfg["path_metadata"]
normalize = dataset_cfg.get("normalize", False)
one_hot_encoding = dataset_cfg.get("one_hot_encoding", False)
add_reverse_edges = dataset_cfg.get("add_reverse_edges", False)

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]
process_execution_object_types = process_execution_cfg["object_types"]
process_execution_target_activity = process_execution_cfg.get("target_activity")
trace_backward = process_execution_cfg.get("trace_backward", False)

# GNN
gnn_cfg = cfg["gnn"]
path_model = gnn_cfg["path_model"]
random_seed = gnn_cfg.get("random_seed", 0)

torch.manual_seed(random_seed)
seed(random_seed)

# Counterfactual search
counterfactual_cfg = cfg["counterfactual"]
viewpoint_event_label = counterfactual_cfg["viewpoint_event_label"]
max_change_size = counterfactual_cfg["max_change_size"]

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

# %% Load OCEL
ocel = pm4py.read_ocel2_json(path_ocel)

# Convert timestamp to epoch
ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

# %% Convert OCEL to Networkx graph
ocel_nx = pm4py.convert_ocel_to_networkx(ocel)

# %% Load model and dataset, and define process outcome function
model = torch.load(path_model, weights_only=False)
model.eval()

dataset = torch.load(path_dataset, weights_only=False)

homogeneous_dataset = []
labels = []
for data in dataset:
    homogeneous_dataset.append(data.to_homogeneous())
    labels.append(torch.tensor([data.y], dtype=torch.long))

stats = GraphCFEDatasetStats.from_dataset(homogeneous_dataset, labels)

# %% Train GraphCFE model
device = "cuda" if torch.cuda.is_available() else "cpu"


# Build dense training tensors for GraphCFE and train the VAE on the available dataset.
def _build_graphcfe_training_tensors(dataset, labels, stats, device):
    features = []
    adjs = []
    ys = []
    us = []
    for data, label in zip(dataset, labels):
        feat, adj, _ = _to_dense(data, stats.max_num_nodes, stats.x_dim, device)
        features.append(feat)
        adjs.append(adj)
        ys.append(
            torch.tensor([[float(label.item())]], dtype=torch.float32, device=device)
        )

        num_edges = int(data.edge_index.shape[1])
        num_nodes = max(int(data.num_nodes), 1)
        u_value = float(num_edges / num_nodes)
        us.append(torch.tensor([[u_value]], dtype=torch.float32, device=device))

    return (
        torch.cat(features, dim=0),
        torch.cat(adjs, dim=0),
        torch.cat(ys, dim=0),
        torch.cat(us, dim=0),
    )


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
):
    """Train GraphCFE model using CLEAR loss from https://github.com/jma712/GraphCFE."""
    graphcfe_model.to(device)
    optimizer = torch.optim.Adam(graphcfe_model.parameters(), lr=learning_rate)
    num_samples = features.size(0)

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
        perm = torch.randperm(num_samples, device=device)
        total_loss = 0.0

        for start in range(0, num_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_x = features[batch_idx]
            batch_adj = adjs[batch_idx]
            batch_y = ys[batch_idx]
            batch_u = us[batch_idx]

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

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"[GraphCFE] epoch {epoch}/{num_epochs} "
                f"loss={(total_loss / num_samples):.4f} "
                f"loss_sim={loss_sim.item():.4f} "
                f"loss_kl={loss_kl.item():.4f}"
            )

    graphcfe_model.eval()
    return graphcfe_model


graphcfe_args = SimpleNamespace(
    dim_h=gnn_cfg.get("dim_h", 64),
    dim_z=gnn_cfg.get("dim_z", 32),
    dropout=gnn_cfg.get("dropout", 0.1),
    disable_u=counterfactual_cfg.get("disable_u", False),
)

graphcfe_init_params = {
    "vae_type": "graphVAE",
    "x_dim": stats.x_dim,
    "max_num_nodes": stats.max_num_nodes,
}

graphcfe_model = GraphCFE(graphcfe_init_params, graphcfe_args)

train_features, train_adjs, train_ys, train_us = _build_graphcfe_training_tensors(
    homogeneous_dataset,
    labels,
    stats,
    device,
)

graphcfe_model = train_graphcfe_model(
    graphcfe_model,
    train_features,
    train_adjs,
    train_ys,
    train_us,
    num_epochs=counterfactual_cfg.get("graphcfe_epochs", 200),
    batch_size=counterfactual_cfg.get("graphcfe_batch_size", 16),
    learning_rate=counterfactual_cfg.get("graphcfe_learning_rate", 1e-3),
    device=device,
)

# %%
# Determine viewpoint_event_id from labels file + config label
with open(path_labels, "r") as f:
    labels_data = json.load(f)

if "viewpoint_event_labels" not in labels_data:
    raise KeyError(f"Missing 'viewpoint_event_labels' key in {path_labels}")

viewpoint_event_ids = []
for event_id, event_label in labels_data["viewpoint_event_labels"].items():
    if event_label == viewpoint_event_label:
        process_execution = extract_process_execution(
            ocel_nx,
            event_id,
            object_types=process_execution_object_types,
            target_activity_type=process_execution_target_activity,
            backward=trace_backward,
        )
        # if process_outcome(process_execution) == viewpoint_event_label:
        viewpoint_event_ids.append(event_id)

if not viewpoint_event_ids:
    raise ValueError(
        f"No event found in {path_labels} with actual and predicted label {viewpoint_event_label}"
    )
viewpoint_event_id = viewpoint_event_ids[0]

# %% Select graph to explain
# Extract target process execution
data_dict = {}
for event_id in viewpoint_event_ids:
    process_execution = deepcopy(
        extract_process_execution(
            ocel_nx,
            event_id,
            object_types=process_execution_object_types,
            target_activity_type=process_execution_target_activity,
            backward=trace_backward,
        )
    )

    data, _, _, _, feat_label_dict, node_label_dict = build_hetero_data(
        graph=process_execution,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=ocel.object_type_column,
        event_activity_col=ocel.event_activity,
        viewpoint=metadata.viewpoint,
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
        add_reverse_edges=metadata.add_reverse_edges,
    )

    data_dict[event_id] = (data, node_label_dict)

# %% GraphCFE
hetero_data = data_dict[viewpoint_event_id][0]

print("counterfactual_label =", viewpoint_event_label)

# edits = generate_graphcfe_counterfactual(
#     graphcfe_model=graphcfe_model,
#     graph=hetero_data,
#     target_class=int(viewpoint_event_label),
#     adj_threshold=0.0,
#     feat_threshold=0.0,
#     dataset_stats=stats,
#     graph_matching=False,
# )

edits, proximity = generate_graphcfe_counterfactual_with_proximity(
    graphcfe_model=graphcfe_model,
    graph=hetero_data,
    target_class=int(viewpoint_event_label),
    # adj_threshold=0.0,
    # feat_threshold=0.0,
    dataset_stats=stats,
    graph_matching=False,
)


# %% Modify process execution graph
def apply_graphcfe_edits_to_hetero_data(
    hetero_data: HeteroData,
    edits: list,
    unique_edges: bool = True,
) -> HeteroData:
    """Apply GraphCFE edits directly to a HeteroData object.

    The GraphCFE edits are produced on the homogeneous view of the graph.
    This function maps homogeneous node indices back to the hetero node types
    and local node indices, then applies edge and feature changes in the
    hetero representation.
    """
    data = deepcopy(hetero_data)
    node_types = list(data.node_types)
    node_counts = [int(data[node_type].num_nodes) for node_type in node_types]
    node_offsets = []
    offset = 0
    for count in node_counts:
        node_offsets.append(offset)
        offset += count

    def _homogeneous_to_hetero(index: int) -> tuple[str, int]:
        idx = int(index)
        for node_type, start, count in zip(node_types, node_offsets, node_counts):
            if start <= idx < start + count:
                return node_type, idx - start
        raise IndexError(f"Homogeneous node index {idx} out of range")

    def _get_edit_value(edit, key, default=None):
        if isinstance(edit, dict):
            return edit.get(key, default)
        return getattr(edit, key, default)

    def _find_edge_type(src_type, dst_type):
        candidates = [
            edge_type
            for edge_type in data.edge_index_dict.keys()
            if edge_type[0] == src_type and edge_type[2] == dst_type
        ]
        if not candidates:
            raise ValueError(f"No hetero edge type found for {src_type} -> {dst_type}")
        if len(candidates) == 1:
            return candidates[0]
        return candidates

    for edit in edits:
        edit_type = _get_edit_value(edit, "action") or _get_edit_value(
            edit, "edit_type"
        )
        if edit_type == "change_node_feat":
            node_idx = int(_get_edit_value(edit, "node_idx"))
            feature_idx = int(_get_edit_value(edit, "feature_idx"))
            new_value = float(_get_edit_value(edit, "new_value"))
            node_type, local_idx = _homogeneous_to_hetero(node_idx)
            if not hasattr(data[node_type], "x"):
                raise ValueError(
                    f"HeteroData node type {node_type} has no feature tensor"
                )
            data[node_type].x[local_idx, feature_idx] = new_value

        elif edit_type in {"remove_edge", "add_edge"}:
            src = int(_get_edit_value(edit, "src"))
            dst = int(_get_edit_value(edit, "dst"))
            src_type, src_local = _homogeneous_to_hetero(src)
            dst_type, dst_local = _homogeneous_to_hetero(dst)
            edge_type = _find_edge_type(src_type, dst_type)

            if isinstance(edge_type, list):
                matched = None
                for candidate in edge_type:
                    edge_index = data.edge_index_dict[candidate]
                    if (
                        (edge_index[0] == src_local) & (edge_index[1] == dst_local)
                    ).any():
                        matched = candidate
                        break
                edge_type = matched or edge_type[0]

            edge_index = data.edge_index_dict[edge_type]
            if edit_type == "remove_edge":
                mask = ~((edge_index[0] == src_local) & (edge_index[1] == dst_local))
                data.edge_index_dict[edge_type] = edge_index[:, mask]
            else:
                new_edge = torch.tensor(
                    [[src_local], [dst_local]],
                    dtype=edge_index.dtype,
                    device=edge_index.device,
                )
                edge_index = torch.cat([edge_index, new_edge], dim=1)
                if unique_edges:
                    edge_index = torch.unique(edge_index, dim=1)
                data.edge_index_dict[edge_type] = edge_index

        else:
            raise ValueError(f"Unsupported GraphCFE edit type: {edit_type}")

    return data


hetero_data_cf = apply_graphcfe_edits_to_hetero_data(hetero_data, edits)


def predict(data: HeteroData) -> bool:
    batch_dict = {
        node_type: torch.zeros(
            data[node_type].num_nodes if data[node_type].num_nodes else 0,
            dtype=torch.long,
            device=device,
        )
        for node_type in metadata.node_types
    }
    out = model(data.x_dict, data.edge_index_dict, batch_dict)

    return bool(out.argmax(dim=-1).cpu().item())
