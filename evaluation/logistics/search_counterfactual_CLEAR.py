# %% Import dependencies
import json
import os
import pm4py
import torch
import yaml

from copy import deepcopy
from networkx import Graph
from random import seed
from types import SimpleNamespace

from evaluation.clear_graph_cfe import (
    build_graphcfe_training_tensors,
    GraphCFE,
    generate_graphcfe_counterfactual_with_evaluation,
    GraphCFEDatasetStats,
    train_graphcfe_model,
)
from evaluation.metrics import compute_proximity
from evaluation.utils import to_dense, hetero_to_homogeneous_data


from gnn.hetero_graph_data import build_hetero_data, to_homogeneous_data
from gnn.utils import Metadata
from process_execution.process_execution import extract_process_execution

from utils import clean_ocel_dataset

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

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
homogeneous = gnn_cfg.get("homogeneous", False)
random_seed = gnn_cfg.get("random_seed", 0)

torch.manual_seed(random_seed)
seed(random_seed)

# Counterfactual search
counterfactual_cfg = cfg["counterfactual"]
viewpoint_label = counterfactual_cfg["viewpoint_label"]

# Counterfactual GraphCFE
graphcfe_cfg = cfg["counterfactual_graphcfe"]
path_graphcfe_model = graphcfe_cfg["path_model"]
graphcfe_args = SimpleNamespace(
    dim_h=graphcfe_cfg["dim_h"],
    dim_z=graphcfe_cfg["dim_z"],
    dropout=graphcfe_cfg["dropout"],
    disable_u=graphcfe_cfg.get("disable_u", False),
)

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

# %% Load OCEL
ocel = pm4py.read_ocel2_json(path_ocel)
ocel = clean_ocel_dataset(ocel)

# Convert timestamp to epoch
ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

# %% Convert OCEL to Networkx graph
ocel_nx = pm4py.convert_ocel_to_networkx(ocel)

# %% Load model and dataset, and define process outcome function
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(path_model, weights_only=False)
model.eval()

dataset = torch.load(path_dataset, weights_only=False)

homogeneous_dataset = []
labels = []
for data in dataset:
    homogeneous_dataset.append(
        to_homogeneous_data(
            data,
            metadata.node_num_keys,
            metadata.node_cat_keys,
            metadata.node_types,
            metadata.one_hot_encoding,
            metadata.unique_node_type_attribute_columns,
        )
    )
    labels.append(torch.tensor([data.y], dtype=torch.long))

stats = GraphCFEDatasetStats.from_dataset(homogeneous_dataset, labels)


@torch.no_grad()
def process_outcome(p: Graph) -> bool:
    """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

    Args:
        ocel_graph (Graph): The process execution to classify.
    Returns:
        float: The predicted value.
    """
    hetero_data, _, _, _, _, _ = build_hetero_data(
        graph=p,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=ocel.object_type_column,
        event_activity_col=ocel.event_activity,
        viewpoint=metadata.viewpoint,
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
        add_reverse_edges=metadata.add_reverse_edges,
    )

    if homogeneous:
        data = to_homogeneous_data(
            hetero_data,
            metadata.node_num_keys,
            metadata.node_cat_keys,
            metadata.node_types,
            metadata.one_hot_encoding,
            metadata.unique_node_type_attribute_columns,
        )
        batch = torch.zeros(
            data.num_nodes if data.num_nodes else 0,
            dtype=torch.long,
            device=device,
        )
        data = data.to(device)
        out = model(data.x, data.edge_index, batch)
    else:
        data = hetero_data.to(device)

        try:
            # For a single graph, create a batch vector of zeros (all nodes belong to graph 0)
            batch_dict = {
                node_type: torch.zeros(
                    data[node_type].num_nodes if data[node_type].num_nodes else 0,
                    dtype=torch.long,
                    device=device,
                )
                for node_type in metadata.node_types
            }
            out = model(data.x_dict, data.edge_index_dict, batch_dict)

        except Exception as e:
            print(f"Error occurred while processing graph: {e}")
            print(data)
            raise e

    return bool(out.argmax(dim=-1).cpu().item())


# %% Train GraphCFE model
graphcfe_init_params = {
    "vae_type": graphcfe_cfg["vae_type"],
    "x_dim": stats.x_dim,
    "max_num_nodes": stats.max_num_nodes,
}
graphcfe_model = GraphCFE(graphcfe_init_params, graphcfe_args)

train_features, train_adjs, train_ys, train_us = build_graphcfe_training_tensors(
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
    num_epochs=graphcfe_cfg.get("n_epochs", 200),
    batch_size=graphcfe_cfg.get("batch_size", 10),
    learning_rate=graphcfe_cfg.get("learning_rate", 1e-3),
    patience=graphcfe_cfg.get("patience", 10),
    device=device,
)

torch.save(graphcfe_model, path_graphcfe_model)

# graphcfe_model = torch.load(path_graphcfe_model, weights_only=False)

# %%
# Determine viewpoint id(s)

viewpoint_ids = []
if counterfactual_cfg.get("viewpoint_id"):
    viewpoint_ids.append(counterfactual_cfg.get("viewpoint_id"))
else:
    with open(path_labels, "r") as f:
        labels_data = json.load(f)
    for event_id, event_label in labels_data["viewpoint_event_labels"].items():
        if event_label == viewpoint_label:
            viewpoint_ids.append(event_id)

if not viewpoint_ids:
    raise ValueError(
        f"No event found in {path_labels} with actual label {viewpoint_label}"
    )

# %% Select graph to explain
# Extract target process execution
viewpoint_data_dict = {}
for viewpoint_id in viewpoint_ids:
    process_execution = deepcopy(
        extract_process_execution(
            ocel_nx,
            viewpoint_id,
            object_types=process_execution_object_types,
            target_activity_type=process_execution_target_activity,
            backward=trace_backward,
        )
    )

    if process_outcome(process_execution) != viewpoint_label:
        print(f"Skipping {viewpoint_id}")
        continue

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

    viewpoint_data_dict[viewpoint_id] = (data, node_label_dict)

# %% GraphCFE
print("counterfactual_label =", not viewpoint_label)

results = []
for viewpoint_id, record in viewpoint_data_dict.items():
    hetero_data = record[0]

    features_cf, adj_cf, edits, proximity_metrics = (
        generate_graphcfe_counterfactual_with_evaluation(
            graphcfe_model=graphcfe_model,
            graph=hetero_data,
            metadata=metadata,
            target_class=int(not viewpoint_label),
            adj_threshold=0.001,
            feat_threshold=0.001,
            dataset_stats=stats,
            graph_matching=True,
            evaluation_model=model,
        )
    )

    data = hetero_to_homogeneous_data(hetero_data, metadata)
    features_orig, adj_orig, _ = to_dense(
        data, stats.max_num_nodes, stats.x_dim, device
    )

    proximity_metrics_all = {}
    for other_id, record_i in viewpoint_data_dict.items():
        data_i = hetero_to_homogeneous_data(record_i[0], metadata)
        features_i, adj_i, _ = to_dense(
            data_i, stats.max_num_nodes, stats.x_dim, device
        )
        proximity_metrics_all[other_id] = compute_proximity(
            features_cf, adj_cf, features_i, adj_i
        )

    results.append(
        {
            "viewpoint_id": viewpoint_id,
            "edits": str(edits),
            "proximity_metrics": proximity_metrics,
            "proximity_metrics_all": proximity_metrics_all,
        }
    )

# Only store results if evaluated for multiple process executions
if len(viewpoint_ids) > 1:
    print(len(results), "results collected")
    run_id = os.getenv("RUN_ID")
    file_name = os.path.basename(os.path.dirname(__file__))
    with open(
        f"results/{file_name}{'_' + config_file.split('/')[-1].split('.')[0]}-CLEAR{f'-{run_id}' if run_id else ''}.json",
        "w",
    ) as f:
        json.dump(results, f)
