# %% Import dependencies
import json
import os
import pm4py
import torch
import yaml

from networkx import Graph

from tree_search.action_helpers import (
    build_object_substitution_actions,
    build_node_deletion_actions,
    build_node_attribute_actions,
    build_event_move_actions,
    build_event_insertion_actions,
    build_object_insertion_actions,
    construct_attribute_spec_dict,
    get_nodes_by_importance,
    get_feature_labels_by_importance,
)
from tree_search.action_set import ActionSet
from tree_search.tree_search import TreeSearchCounterFactual

from gnn.hetero_graph_data import build_hetero_data
from gnn.utils import Metadata, generate_explanation
from utils import convert_event_log_ocel

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Dataset
dataset_cfg = cfg["dataset"]
path_xes = dataset_cfg["path_xes"]
path_metadata = dataset_cfg["path_metadata"]

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]

# GNN
gnn_cfg = cfg["gnn"]
path_model = gnn_cfg["path_model"]

# Counterfactual search
counterfactual_cfg = cfg["counterfactual"]
viewpoint_object_id = counterfactual_cfg["viewpoint_object_id"]
depth_first = counterfactual_cfg.get("depth_first")
num_bins = counterfactual_cfg["num_bins"]
max_change_size = counterfactual_cfg["max_change_size"]
node_importance_threshold = counterfactual_cfg["node_importance_threshold"]
attr_importance_threshold = counterfactual_cfg["attr_importance_threshold"]

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

# %% Load event log
event_log = pm4py.read_xes(path_xes)

# %% Convert event log to OCEL and Networkx graph
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
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
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
events = set([e for e, _ in ocel_nx.in_edges(viewpoint_object_id)])
nodes = events | set([viewpoint_object_id])
target_process_execution = ocel_nx.subgraph(nodes).copy()

counterfactual_label = not process_outcome(target_process_execution)

target_explanation, hetero_data, feat_label_dict, node_label_dict = generate_explanation(
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
        node_cat_keys=metadata.node_cat_keys,
        one_hot_encoding=metadata.one_hot_encoding,
    ).items()
}

# Actions for object node attributes
object_node_attributes = build_node_attribute_actions(
    target_nodes=target_process_execution.nodes(data=True),
    attribute_spec_dict=attribute_spec_dict,
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
    node_type="OBJECT",
    object_type_column=ocel.object_type_column,
)

# Object substitution actions
target_nodes_for_subst = (
    (n, target_process_execution.nodes(data=True)[n])
    for n in nodes_ordered
    if target_process_execution.nodes(data=True)[n]
    .get("attr", {})
    .get(ocel.object_type_column, "")
    in [""]
)

object_substitution_actions = build_object_substitution_actions(
    target_nodes=target_nodes_for_subst,
    ocel_nodes=ocel_nx.nodes(data=True),
    graph=target_process_execution,
    object_type_column=ocel.object_type_column,
    attribute_spec_dict=attribute_spec_dict,
)

# Actions for event node attributes
event_node_attributes = build_node_attribute_actions(
    target_nodes=target_process_execution.nodes(data=True),
    attribute_spec_dict=attribute_spec_dict,
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
    node_type="EVENT",
)

# Event move actions
target_event_nodes = [
    (n, target_process_execution.nodes(data=True)[n])
    for n in nodes_ordered
    if target_process_execution.nodes(data=True)[n].get("attr", {}).get("type", "")
    == "EVENT"
]
event_move_actions = build_event_move_actions(
    target_event_nodes=target_event_nodes,
    candidate_event_nodes=target_event_nodes,
)

# Events to insert
event_insertion_object_ids = [
    n
    for n, attr in target_process_execution.nodes(data="attr")
    if attr.get(ocel.object_type_column, "") in [viewpoint]
]
event_insertion_actions = build_event_insertion_actions(
    target_event_nodes=target_event_nodes,
    event_activities=list(ocel.events[ocel.event_activity].unique()),
    object_ids=event_insertion_object_ids,
)

# Objects to insert
object_insertion_actions = build_object_insertion_actions(
    target_event_nodes=target_event_nodes,
    object_types=set(
        attr[ocel.object_type_column]
        # only from target_process_execution, as other node types might not exist in learned GNN
        for _, attr in target_process_execution.nodes(data="attr")
        if attr.get(ocel.object_type_column)
    ),
    metadata=metadata,
)

# Events that can be deleted
node_deletion_actions = build_node_deletion_actions(
    target_process_execution.nodes(data=True),
    nodes_order=nodes_ordered,
    viewpoint=metadata.viewpoint,
    object_type_column=ocel.object_type_column,
)

# Group actions if depth_first is defined
if depth_first == "node":
    actions_grouped = {node: [] for node in nodes_ordered}
    for action in object_node_attributes + event_node_attributes:
        if action.action_space_size() > 0:
            actions_grouped[action.node_id].append(action)

    for action in object_substitution_actions:
        if action.action_space_size() > 0:
            actions_grouped[action.object_id].append(action)

    for action in event_move_actions:
        if action.action_space_size() > 0:
            actions_grouped[action.event_id].append(action)

    for action in event_insertion_actions:
        if action.action_space_size() > 0:
            actions_grouped[action.event_id].append(action)

    for action in object_insertion_actions:
        if action.action_space_size() > 0:
            actions_grouped[action.event_id].append(action)

    for action in node_deletion_actions:
        if action.action_space_size() > 0:
            for option in action.deletion_options:
                for node in option:
                    actions_grouped[node].append(action)


elif depth_first == "attribute":
    actions_grouped = {attr: [] for attrs in attr_ordered.values() for attr in attrs}
    for action in object_node_attributes + event_node_attributes:
        if action.action_space_size() > 0:
            actions_grouped[action.attribute_name].append(action)

    # Include all node actions (substitute/insert/delete) in one group
    actions_grouped["node"] = (
        object_substitution_actions
        + event_move_actions
        + object_insertion_actions
        + event_insertion_actions
        + node_deletion_actions
    )

else:
    available_actions = (
        object_node_attributes
        + event_node_attributes
        + object_substitution_actions
        + event_move_actions
        + object_insertion_actions
        + event_insertion_actions
        + node_deletion_actions
    )


# %% Run tree search algorithm to find counter factuals
tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counterfactual_label,
    log_file="logs/bpi_2011.log",
)

print(f"counterfactual_label={counterfactual_label}")
if depth_first:
    for key, group in actions_grouped.items():
        print(f"Selected number of actions for {key}: {len(group)}")

    selected_action_sets = tree_search.search_depth_first(
        actions_grouped=actions_grouped,
        process_execution=target_process_execution,
    )
else:
    print(f"Selected number of actions: {len(available_actions)}")
    selected_action_sets = tree_search.search_layer(
        actions_to_explore=[(ActionSet(), available_actions)],
        process_execution=target_process_execution,
    )

# %% Display results
print(f"Number of selected action sets: {len(selected_action_sets)}")
for selected_action in sorted(
    selected_action_sets, key=lambda a: a.action_size(), reverse=True
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
