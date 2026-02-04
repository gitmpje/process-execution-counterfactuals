# %% Import dependencies
import json
import gzip
import networkx as nx
import os
import pm4py
import torch

from collections import Counter
from networkx import Graph
from numpy import arange

from tree_search.feature import (
    NodeAttributeNumeric,
    ObjectNodeSubstitution,
)
from tree_search.tree_search import Action, TreeSearchCounterFactual
from tree_search.tree_search_parallel import TreeSearchCounterFactualParallel

from process_execution.process_execution import (
    extract_process_execution,
    ProcessExecution,
)
from process_execution.utils import load_graphml_with_json_attrs
from gnn.gcn_graph_classification import convert_trace_graphs_to_pyg
from process_execution.visualization import (
    apply_node_styles_nx,
    apply_edge_styles_nx,
)


dirname = os.path.dirname(__file__)

path_ocel = os.path.join(dirname, "data/example_gnn.json.gz")
path_graphml = os.path.join(dirname, "data/example_gnn.graphml.gz")

path_model = os.path.join(dirname, "data/example_gnn_han.pth")
path_vocab = os.path.join(dirname, "data/example_gnn-vocab.json")

tmp_dir = os.path.join(dirname, "tmp")

# Unzip .gz files and store to temporary directory
for var_path in ["path_ocel", "path_graphml", "path_model", "path_vocab"]:
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
target_object_types = ["PackingUnit"]

ocel = pm4py.read_ocel2_json(path_ocel)
ocel_nx = load_graphml_with_json_attrs(path_graphml)

# %% Extract process executions
# Define event attribute and activity to base classification on
selected_activity = "Object-departing-WB"
selected_attribute = "a"

# Extract events related to target object types
df_events = ocel.events.copy()
df_events.set_index(ocel.event_id_column, inplace=True)
df_relations = ocel.relations.copy()
df_relations.set_index(ocel.event_id_column, inplace=True)
df_events_objects = df_events.join(df_relations, rsuffix="_relations")

events_to_trace = df_events_objects[
    (df_events_objects[ocel.object_type_column].isin(target_object_types))
].index.values

print(f"Number of events selected: {len(events_to_trace)}")


def determine_class_event_attribute(trace_graph: Graph):
    for _, data in trace_graph.nodes(data="attr"):
        if (
            data.get(ocel.event_activity, "") == selected_activity
            and data.get(selected_attribute, 1) < 0.25
        ):
            return False
    return True


trace_graphs = {}
for event in events_to_trace:
    trace_graph = extract_process_execution(
        ocel_nx,
        event,
        ["ProductionLot", "PackingUnit"],
        "Object-creating_class_instance",
    )
    trace_graph.construct_node_label()
    trace_graph.construct_edge_label()

    trace_graphs[event] = {
        "process_execution": trace_graph,
        "class": determine_class_event_attribute(trace_graph),
    }


print("Classes:", Counter([d["class"] for d in trace_graphs.values()]))

# %% Load GNN model and define process outcome function

# Define vocabulary and numeric attribute keys
with open(path_vocab) as f:
    vocab_dict = json.load(f)
node_labels = vocab_dict["node_labels"]
node_numeric_keys = vocab_dict["node_numeric_keys"]

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_model, weights_only=False)
model = model.to(device)
model.eval()


def process_outcome(p: ProcessExecution):
    """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

    Args:
        p (ProcessExecution): The process execution to classify.
    Returns:
        bool: The predicted class label (True/False).
    """

    # Convert the single ProcessExecution into the converter's expected input
    try:
        graph_map = {"_tmp": {"process_execution": p}}
        data_list = convert_trace_graphs_to_pyg(
            graph_map, node_labels, node_numeric_keys, []
        )
        if not data_list:
            raise RuntimeError("converter returned empty list")
        data = data_list[0]
    except Exception as e:
        print("process_outcome: conversion to PyG data failed:", e)
        raise

    batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    data = data.to(device)
    # Forward through the model
    try:
        with torch.no_grad():
            out = model(data.x, data.edge_index, batch_vec)

            # Expecting graph-level logits shaped (1, n_classes)
            if out.dim() == 2 and out.size(0) == 1:
                pred = int(out.argmax(dim=1).item())
            else:
                # Fallback: take global argmax
                pred = int(out.argmax().item())
    except Exception as e:
        print("process_outcome: model forward failed:", e)
        raise

    return bool(pred)


# %% Configure counterfactual generation (tree search)
# Select process execution to generate counterfactual for
target_process_execution_id = "100023"

selected_event_attributes = {
    "a": arange(0, 1.01, 0.5),
    "quantity": range(0, 1001, 500),
}
max_change_size = 10
counter_factual_label = not trace_graphs[target_process_execution_id]["class"]

target_process_execution = trace_graphs[target_process_execution_id][
    "process_execution"
]

# Object substitution features
object_substitution_features = []
for node_id, data in target_process_execution.nodes(data=True):
    if data["attr"].get("type", "") != "OBJECT":
        continue

    # Only allow substitution of production resources
    if data["attr"].get(ocel.object_type_column, "") != "ProductionResource":
        continue

    substitution_objects = [
        (subst_id, subst_data)
        for subst_id, subst_data in ocel_nx.nodes(data=True)
        if subst_data["attr"].get(ocel.object_type_column, "")
        == data["attr"].get(ocel.object_type_column, "")
        and subst_data["attr"].get("capability", "")
        == data["attr"].get("capability", "")
        and subst_id != node_id
    ]

    object_substitution_features.append(
        ObjectNodeSubstitution(
            object_id=node_id,
            substitution_objects=substitution_objects,
        )
    )

# Features for event node attributes
event_node_attributes = [
    NodeAttributeNumeric(
        node_id=node_id,
        attribute_name=attr_name,
        value_original=attr[attr_name],
        value_range=selected_event_attributes[attr_name],
    )
    for node_id, attr in target_process_execution.nodes(data="attr")
    if attr.get("type", "") == "EVENT"
    for attr_name in attr.keys()
    if attr_name in selected_event_attributes
]


available_features = event_node_attributes  # + object_substitution_features
for feature in available_features:
    print(feature)

# %% Run tree search algorithm to find counter factuals
tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counter_factual_label,
)

selected_actions = tree_search.search_layer(
    [(Action(), available_features)],
    target_process_execution,
)

# %% Run tree search algorithm in parallel
tree_search_parallel = TreeSearchCounterFactualParallel(
    process_outcome=process_outcome,
    max_changes=max_change_size,
    counterfactual_label=counter_factual_label,
    num_workers=5,
)

selected_actions = tree_search_parallel.find_counterfactuals(
    available_features,
    target_process_execution,
)

# %% Display results
if selected_actions:
    print("Number of selected actions: ", len(selected_actions))
    for selected_action in selected_actions:
        print("Objective value:", selected_action.objective_value())
        print(
            [
                f"{feature}: {change_value}"
                for feature, change_value in selected_action.node_attributes_modification.items()
                if change_value != 0
            ],
            [
                (feature.event_id, feature.object_id, subst[0])
                for feature, subst in selected_action.object_substitution.items()
                if subst and feature.object_id != subst[0]
            ],
        )
else:
    print("No counterfactual actions found")

# %% Visualize target process execution
apply_node_styles_nx(target_process_execution)  # apply coloring + tooltip
apply_edge_styles_nx(target_process_execution)  # apply coloring + tooltip

# Draw base process execution graph
agraph = nx.nx_agraph.to_agraph(target_process_execution)
agraph.draw("figures/target_process_execution.svg", prog="dot")
