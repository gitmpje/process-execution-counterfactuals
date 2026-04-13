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

from gnn.graph_cfe import (
    GraphCFE,
    generate_graphcfe_counterfactual_with_evaluation,
    GraphCFEDatasetStats,
)

from gnn.hetero_graph_data import build_hetero_data
from gnn.utils import Metadata
from process_execution.process_execution import extract_process_execution

from utils import (
    train_graphcfe_model,
    _build_graphcfe_training_tensors,
    _replace_scenario_prefix,
)

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
homogeneous = gnn_cfg.get("homogeneous", False)
random_seed = gnn_cfg.get("random_seed", 0)

torch.manual_seed(random_seed)
seed(random_seed)

# Counterfactual search
counterfactual_cfg = cfg["counterfactual"]
viewpoint_event_label = counterfactual_cfg["viewpoint_event_label"]

# Counterfactual GraphCFE
graphcfe_cfg = cfg["counterfactual_graphcfe"]
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
    homogeneous_dataset.append(data.to_homogeneous())
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
    try:
        data, _, _, _, _, _ = build_hetero_data(
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

        data = data.to(device)

        if homogeneous:
            data = data.to_homogeneous()
            batch = torch.zeros(
                data.num_nodes if data.num_nodes else 0,
                dtype=torch.long,
                device=device,
            )
            out = model(data.x, data.edge_index, batch)
        else:
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
    num_epochs=graphcfe_cfg.get("n_epochs", 200),
    batch_size=graphcfe_cfg.get("batch_size", 10),
    learning_rate=graphcfe_cfg.get("learning_rate", 1e-3),
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
        if process_outcome(process_execution) == viewpoint_event_label:
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

edits, evaluation = generate_graphcfe_counterfactual_with_evaluation(
    graphcfe_model=graphcfe_model,
    graph=hetero_data,
    target_class=int(viewpoint_event_label),
    # adj_threshold=0.0,
    # feat_threshold=0.0,
    dataset_stats=stats,
    graph_matching=False,
    evaluation_model=model,
)

print(edits)
print(evaluation)
