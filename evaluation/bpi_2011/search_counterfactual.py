# %% Import dependencies
import json
import os
import pm4py
import torch
import yaml

from networkx import Graph

from tree_search.action import Action
from tree_search.tree_search import TreeSearchCounterFactual
from tree_search.feature_helpers import (
    build_object_substitution_features,
    build_node_deletion_features,
    build_node_attribute_features,
    construct_attribute_spec_dict,
)
from tree_search.feature_selection import (
    get_nodes_by_importance,
    get_feature_labels_by_importance,
)

from gnn.hetero_graph_data import build_hetero_data
from gnn.utils import Metadata, generate_explanation
from utils import convert_event_log_ocel

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Target case for which to generate counterfactual
target_process_execution_id = "00000001"

# Search algorithm
num_bins = 2  # number of bins to use for numeric attribute range
max_change_size = 2
node_importance_threshold = 0.001
attr_importance_threshold = 0.01

# Dataset
dataset_cfg = cfg.get("dataset", {})
viewpoint = dataset_cfg.get("viewpoint")
path_xes = dataset_cfg.get("path_xes")
path_model = dataset_cfg.get("path_model")
path_metadata = dataset_cfg.get("path_metadata")

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

# %% Load OCEL and build DFG
event_log = pm4py.read_xes(path_xes)

# %% Convert event log to OCEL

ocel, ocel_nx = convert_event_log_ocel(event_log, viewpoint)

# %% Load model and define process outcome function

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_model, weights_only=False)
model = model.to(device)
model.eval()


@torch.no_grad()
def process_outcome(p: Graph) -> bool:
    """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

    Args:
        ocel_graph (Graph): The process execution to classify.
    Returns:
        float: The predicted value.
    """
    data, _, _, _, _, _ = build_hetero_data(
        graph=p,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=ocel.object_type_column,
        event_activity_col=ocel.event_activity,
        viewpoint=metadata.viewpoint,
        node_y_mapping={target_process_execution_id: None},
        normalize=metadata.normalized,
        feature_per_category=metadata.feature_per_category,
    )

    data = data.to(device)

    # For a single graph, create a batch vector of zeros (all nodes belong to graph 0)
    batch_dict = {
        node_type: torch.zeros(
            data[node_type].num_nodes, dtype=torch.long, device=device
        )
        for node_type in metadata.node_types
    }
    out = model(data.x_dict, data.edge_index_dict, batch_dict)

    return bool(out.argmax(dim=-1).cpu().item())


# %% Configure counterfactual generation (tree search)

# Select process execution to generate counterfactual for
selected_attributes = []
for node_type in ["EVENT", "OBJECT"]:
    for d in metadata.node_cat_keys[node_type].values():
        selected_attributes.extend(d.keys())

    for d in metadata.node_num_keys[node_type].values():
        selected_attributes.extend(d.keys())

attribute_spec_dict = construct_attribute_spec_dict(
    attributes=selected_attributes,
    ocel=ocel,
    node_cat_keys=metadata.node_cat_keys,
    node_num_keys=metadata.node_num_keys,
    num_bins=num_bins,
)

# Extract target process execution
events = set([e for e, _ in ocel_nx.in_edges(target_process_execution_id)])
nodes = events | set([target_process_execution_id])
target_process_execution = ocel_nx.subgraph(nodes)

counterfactual_label = not process_outcome(target_process_execution)

target_explanation, feat_label_dict, node_label_dict = generate_explanation(
    G=target_process_execution,
    metadata=metadata,
    model=model,
    object_type_col=ocel.object_type_column,
    event_activity_col=ocel.event_activity,
    verbose=True,
)

nodes_ordered = [
    n["label"]
    for n in get_nodes_by_importance(
        explanation=target_explanation, node_label_dict=node_label_dict
    )
    if n["importance"] >= node_importance_threshold
]
attr_ordered = {
    node_type: [
        f["feature"] for f in features if f["importance"] >= attr_importance_threshold
    ]
    for node_type, features in get_feature_labels_by_importance(
        explanation=target_explanation,
        feat_label_dict=feat_label_dict,
        feature_per_category=metadata.feature_per_category,
    ).items()
}

counterfactual_label = not process_outcome(target_process_execution)


# Features for object node attributes
object_node_attributes = build_node_attribute_features(
    target_nodes=target_process_execution.nodes(data=True),
    attribute_spec_dict=attribute_spec_dict,
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
    node_type="OBJECT",
    object_type_column=ocel.object_type_column,
)

# Object substitution features
target_nodes_for_subst = (
    (n, target_process_execution.nodes(data=True)[n])
    for n in nodes_ordered
    if target_process_execution.nodes(data=True)[n]
    .get("attr", {})
    .get(ocel.object_type_column, "")
    in [""]
)

object_substitution_features = build_object_substitution_features(
    target_nodes=target_nodes_for_subst,
    ocel_nodes=ocel_nx.nodes(data=True),
    graph=target_process_execution,
    object_type_column=ocel.object_type_column,
    attribute_spec_dict=attribute_spec_dict,
)

# Features for event node attributes
event_node_attributes = build_node_attribute_features(
    target_nodes=target_process_execution.nodes(data=True),
    attribute_spec_dict=attribute_spec_dict,
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
    node_type="EVENT",
)

# Events that can be deleted
node_deletion_features = build_node_deletion_features(
    target_process_execution.nodes(data=True),
    nodes_order=nodes_ordered,
    viewpoint=metadata.viewpoint,
    object_type_column=ocel.object_type_column,
)

available_features = (
    object_node_attributes
    # + object_substitution_features
    # + node_deletion_features
    + event_node_attributes
)
print(f"Total number of features: {len(available_features)}")
selected_features = []
for feature in available_features:
    if feature.action_space_size() > 0:
        print(feature)
        selected_features.append(feature)

print(f"counterfactual_label={counterfactual_label}")
print(f"Selected number of features (action space size > 0): {len(selected_features)}")

# %% Run tree search algorithm to find counter factuals
tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counterfactual_label,
    log_file="logs/bpi_2011.log",
)

selected_actions = tree_search.search_layer(
    [(Action(), available_features)],
    target_process_execution,
)

# %% Display results
print(f"Number of selected actions: {len(selected_actions)}")
for selected_action in sorted(
    selected_actions, key=lambda a: a.action_size(), reverse=True
):
    print(f"Change size {selected_action.action_size()}:", selected_action)


# %% Visualization
def visualize_process_execution(
    process_execution: Graph,
    output_file_name: str = "target_process_execution.svg",
):
    from networkx import nx_agraph
    from process_execution.process_execution import ProcessExecution
    from process_execution.visualization import (
        apply_node_styles_nx,
        apply_edge_styles_nx,
    )

    process_execution = ProcessExecution(process_execution)
    process_execution.construct_node_label()
    process_execution.construct_edge_label()

    apply_node_styles_nx(process_execution)
    apply_edge_styles_nx(process_execution)

    agraph = nx_agraph.to_agraph(process_execution)
    agraph.draw(output_file_name, prog="dot")


visualize_process_execution(target_process_execution)
