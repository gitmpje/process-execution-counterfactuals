# %% Import dependencies
import json
import gzip
import os

import numpy as np
import pm4py
import torch

from collections import Counter
from datetime import datetime
from networkx import Graph

from tree_search.tree_search import Action, TreeSearchCounterFactual
from tree_search.feature import (
    EventNodeDeletion,
    NodeAttributeNumeric,
    ObjectNodeSubstitution,
)

from gnn.hetero_graph_dataset import build_hetero_dataset
from process_execution.process_execution import (
    extract_process_execution,
)
from process_execution.utils import build_ocel_dfg

dirname = os.path.dirname(__file__)
path_ocel = os.path.join(dirname, "ocel/data/order-management.sqlite")

tmp_dir = os.path.join(dirname, "tmp")
path_model = os.path.join(tmp_dir, "example_order_management-pe.pth")
path_metadata = os.path.join(tmp_dir, "example_order_management-pe-metadata.json")

# Unzip .gz files and store to temporary directory
for var_path in ["path_ocel", "path_model"]:
    path = globals()[var_path]
    if not path.endswith(".gz"):
        continue

    tmp_path = os.path.join(tmp_dir, os.path.basename(path).rstrip(".gz"))
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with gzip.open(path) as f:
        with open(tmp_path, "w") as f_out:
            f_out.write(f.read().decode())

    globals()[var_path] = tmp_path

# %% Load OCEL and build DFG with aggregation edges
target_activity = "place order"
viewpoint = "orders"
backward = False
y_key = "class"

ocel = pm4py.read_ocel2_sqlite(path_ocel)

ocel_nx = build_ocel_dfg(ocel)

# Change object type "items" to "_items" to avoid issue with HanConv
for _, d in ocel_nx.nodes(data=True):
    if d["attr"].get(ocel.object_type_column) == "items":
        d["attr"][ocel.object_type_column] = "_items"

# %% Extract process executions
# Convert timestamp to epoch
format_string = "%Y-%m-%d %H:%M:%S"
for _, attr in ocel_nx.nodes(data="attr"):
    if attr.get("type", "") == "EVENT":
        attr["epoch"] = datetime.strptime(
            attr["ocel:timestamp"], format_string
        ).timestamp()

# Extract events related to target activities

with open(path_ocel.replace(".sqlite", "-selected.json")) as f:
    events_to_trace = json.load(f)

events_to_trace = [(e, e.replace("place_", "")) for e in events_to_trace]
print(f"Number of events selected: {len(events_to_trace)}")


def process_time(trace_graph: Graph):
    start_events = [
        d["epoch"]
        for _, d in trace_graph.nodes(data="attr")
        if d.get(ocel.event_activity, "") == "place order"
    ]

    finish_events = [
        d["epoch"]
        for _, d in trace_graph.nodes(data="attr")
        if d.get(ocel.event_activity, "") == "package delivered"
    ]

    if not (start_events and finish_events):
        return None

    return max(finish_events) - min(start_events)


process_executions = {}
for event, viewpoint_object in events_to_trace:
    process_execution = extract_process_execution(
        ocel_nx,
        event,
        ["order", "_items", "packages"],
        backward=backward,
    )
    # process_execution.construct_node_label()
    # process_execution.construct_edge_label()

    p_time = process_time(process_execution)
    if not p_time:
        print(f"Incomplete process execution for {event}")
        continue

    process_executions[event] = {
        "process_execution": process_execution,
        "target_node": event,
        "process_time": p_time,
        "viewpoint_object": viewpoint_object,
    }

# Calculate normalized process times
p_all = []
for trace_dict in process_executions.values():
    p_all.append(trace_dict["process_time"])

# %% Determine class
threshold = np.quantile(p_all, 0.5)
for trace_dict in process_executions.values():
    trace_dict["class"] = trace_dict["process_time"] <= threshold

print("Classes:", Counter([d["class"] for d in process_executions.values()]))


# %% Load model and define process outcome function
# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_model, weights_only=False)
model = model.to(device)
model.eval()

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
node_num_keys = metadata_dict["node_num_keys"]
activities = metadata_dict["activities"]
node_types_set = metadata_dict["node_types_set"]
edge_types_set = metadata_dict["edge_types_set"]


# %%
@torch.no_grad()
def process_outcome(p: Graph) -> bool:
    """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

    Args:
        ocel_graph (Graph): The process execution to classify.
    Returns:
        float: The predicted value.
    """
    graph_map = {"_tmp": {"process_execution": p, y_key: np.nan}}
    dataset, _, _ = build_hetero_dataset(
        graph_map,
        node_num_keys,
        ocel.object_type_column,
        ocel.event_activity,
        viewpoint,
        y_key,
        activities,
        allow_multiple_viewpoint_nodes=True,
    )

    data = dataset[0].to(device)
    # For a single graph, create a batch vector of zeros (all nodes belong to graph 0)
    batch = torch.zeros(data[viewpoint].num_nodes, dtype=torch.long, device=device)
    out = model(data.x_dict, data.edge_index_dict, batch)

    return bool(out.argmax(dim=-1).cpu().item())


# %% Configure counterfactual generation (tree search)
# Select process execution to generate counterfactual for
target_process_execution_id = "place_o-990363"
max_change_size = 40

selected_attributes = {}

target_process_execution = process_executions[target_process_execution_id][
    "process_execution"
]
counterfactual_label = not process_executions[target_process_execution_id]["class"]

# Object substitution features
object_substitution_features = []
for node_id, node_data in target_process_execution.nodes(data=True):
    if node_data["attr"].get("type", "") != "OBJECT":
        continue

    # Only allow substitution of specific object types
    if node_data["attr"].get(ocel.object_type_column, "") not in ["products"]:
        continue

    # Select substition resources based on object type and capability
    substitution_objects = [
        (subst_id, subst_data)
        for subst_id, subst_data in ocel_nx.nodes(data=True)
        if subst_data["attr"].get(ocel.object_type_column, "")
        == node_data["attr"].get(ocel.object_type_column, "")
        and subst_id != node_id
    ]

    object_substitution_features.append(
        ObjectNodeSubstitution(
            object_id=node_id,
            object_data=node_data,
            substitution_objects=substitution_objects,
            event_ids=[
                e
                for e, _, attr in target_process_execution.in_edges(
                    node_id, data="attr"
                )
                if attr["type"] == "E2O"
            ],
        )
    )

# Events that can be deleted
event_nodes = [
    node_id
    for node_id, attr in target_process_execution.nodes(data="attr")
    if attr.get("type", "") == "EVENT"
]
# event_deletion_features = [
#     EventNodeDeletion(
#         deletion_options=[event_nodes[:i] for i in range(1, len(event_nodes))]
#     )
# ]
event_deletion_features = [
    EventNodeDeletion(
        allowed_deletions=event_nodes
    )
]

# Features for event node attributes
event_node_attributes = [
    NodeAttributeNumeric(
        node_id=node_id,
        attribute_name=attr_name,
        value_original=attr[attr_name],
        value_range=selected_attributes[attr_name],
    )
    for node_id, attr in target_process_execution.nodes(data="attr")
    if attr.get("type", "") == "EVENT"
    for attr_name in attr.keys()
    if attr_name in selected_attributes
]

# Features for object node attributes
object_node_attributes = [
    NodeAttributeNumeric(
        node_id=node_id,
        attribute_name=attr_name,
        value_original=attr[attr_name],
        value_range=selected_attributes[attr_name],
    )
    for node_id, attr in target_process_execution.nodes(data="attr")
    if attr.get("type", "") == "OBJECT"
    for attr_name in attr.keys()
    if attr_name in selected_attributes
]

available_features = (
    object_node_attributes
    # + object_substitution_features
    + event_deletion_features
    # + event_node_attributes
)
for feature in available_features:
    print(feature)
print(f"Total number of features: {len(available_features)}")

# %% Run tree search algorithm to find counter factuals
from logging import DEBUG

tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counterfactual_label,
    log_level=DEBUG,
    log_file="log/example_gnn_order_management.log"
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
def visualize_trace_graph(
    identifier: str, output_file_name: str = "figures/target_process_execution.svg"
):
    from networkx import nx_agraph
    from process_execution.visualization import (
        apply_node_styles_nx,
        apply_edge_styles_nx,
    )

    graph = process_executions[identifier]["process_execution"]

    apply_node_styles_nx(graph)
    apply_edge_styles_nx(graph)

    agraph = nx_agraph.to_agraph(graph)
    agraph.draw(output_file_name, prog="dot")


visualize_trace_graph(target_process_execution_id)
