# %% Import dependencies
import json
import gzip
import networkx as nx
import os
import pm4py
import torch

import numpy as np

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
from process_execution.visualization import (
    apply_node_styles_nx,
    apply_edge_styles_nx,
)
from process_execution.utils import build_ocel_dfg
from gnn.hetero_graph_dataset import build_hetero_dataset
from gnn.utils import Metadata

dirname = os.path.dirname(__file__)
path_ocel = os.path.join(dirname, "data/example_DB1_ocel.json.gz")

tmp_dir = os.path.join(dirname, "tmp")
path_pe_model = os.path.join(tmp_dir, "example_DB1-pe.pth")
path_metadata = os.path.join(tmp_dir, "example_DB1-pe-metadata.json")

# path_model = os.path.join(tmp_dir, "example_DB1.pth")

# Unzip .gz files and store to temporary directory
for var_path in ["path_ocel", "path_pe_model"]:
    path = globals()[var_path]
    if not path.endswith(".gz"):
        continue

    tmp_path = os.path.join(tmp_dir, os.path.basename(path).rstrip(".gz"))
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with gzip.open(path) as f:
        with open(tmp_path, "w") as f_out:
            f_out.write(f.read().decode())

    globals()[var_path] = tmp_path

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

target_object_types = ["PackingUnit"]
target_activity = "Aggregation-ADD"
backward = False
y_key = "class"

# %% Load OCEL and build DFG with aggregation edges
ocel = pm4py.read_ocel2_json(path_ocel)

selected_aggregation_activity_qualifier = [
    ("Aggregation-ADD", "childObject"),
]
ocel_nx = build_ocel_dfg(
    ocel, selected_aggregation_activity_qualifier, include_object_relations=True
)

# %% Extract process executions
# Extract events related to target object types
df_events = ocel.events.copy()
df_events.set_index(ocel.event_id_column, inplace=True)
df_relations = ocel.relations.copy()
df_relations.set_index(ocel.event_id_column, inplace=True)
df_events_objects = df_events.join(df_relations, rsuffix="_relations")

df_events_objects.set_index(ocel.object_id_column, append=True, inplace=True)
events_to_trace = df_events_objects[
    (df_events_objects[ocel.object_type_column].isin(target_object_types))
    & (df_events_objects[ocel.event_activity] == target_activity)
].index.values

print(f"Number of events selected: {len(events_to_trace)}")


def determine_class_quality(G: Graph, event: str):
    return int(G.nodes()[event]["attr"].get("averageQuality") >= 1.0)


process_executions = {}
for event, viewpoint_object in events_to_trace:
    process_execution = extract_process_execution(
        ocel_nx,
        event,
        ["ProductionLot", "PackingUnit"],
        "Object-creating_class_instance",
    )
    process_execution.construct_node_label()
    process_execution.construct_edge_label()

    process_executions[event] = {
        "process_execution": process_execution,
        "class": determine_class_quality(ocel_nx, event),
        "viewpoint_object": viewpoint_object,
    }

print("Classes:", Counter([d["class"] for d in process_executions.values()]))

# %% Load GNN model and define process outcome function

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_pe_model, weights_only=False)
model = model.to(device)
model.eval()


@torch.no_grad()
def process_outcome(p: ProcessExecution) -> bool:
    """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

    Args:
        ocel_graph (Graph): The process execution to classify.
    Returns:
        float: The predicted value.
    """
    graph_map = {"_tmp": {"process_execution": p, y_key: np.nan}}
    dataset, _, _ = build_hetero_dataset(
        graph_map,
        metadata.node_num_keys,
        ocel.object_type_column,
        ocel.event_activity,
        metadata.viewpoint,
        y_key,
        metadata.activities,
        allow_multiple_viewpoint_nodes=True,
    )

    data = dataset[0].to(device)
    # For a single graph, create a batch vector of zeros (all nodes belong to graph 0)
    batch_dict = {node_type: torch.zeros(
        data[node_type].num_nodes, dtype=torch.long, device=device
    ) for node_type in metadata.node_types}
    out = model(data.x_dict, data.edge_index_dict, batch_dict)

    return bool(out.argmax(dim=-1).cpu().item())

# %% Configure counterfactual generation (tree search)
# Select process execution to generate counterfactual for
target_process_execution_id = "14602"

selected_attributes = {
    "process_yield": arange(0, 1.01, 0.5),
    # "averageQuality": arange(0, 1.01, 0.5),
    # "quantity": range(0, 1001, 500),
}
max_change_size = 10
counter_factual_label = not process_executions[target_process_execution_id]["class"]

target_process_execution = process_executions[target_process_execution_id][
    "process_execution"
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

# Object substitution features
object_substitution_features = []
for node_id, node_data in target_process_execution.nodes(data=True):
    if node_data["attr"].get("type", "") != "OBJECT":
        continue

    # Only allow substitution of specific object types
    if node_data["attr"].get(ocel.object_type_column, "") not in ["ProductionResource"]:
        continue

    # Select substition resources based on object type and capability
    substitution_objects = [
        (subst_id, subst_data)
        for subst_id, subst_data in ocel_nx.nodes(data=True)
        if subst_data["attr"].get(ocel.object_type_column, "")
        == node_data["attr"].get(ocel.object_type_column, "")
        and subst_data["attr"].get("capability", "")
        == node_data["attr"].get("capability", "")
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


available_features = (
    object_node_attributes
    # + event_node_attributes
    # + object_substitution_features
    # + event_deletion_features
)
for feature in available_features:
    print(feature)
print(f"Total number of features: {len(available_features)}")

# %% Run tree search algorithm to find counter factuals
tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counter_factual_label,
    log_file="log/example_gnn_graph_classification.log",
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
print(f"Number of selected actions: {len(selected_actions)}")
for selected_action in sorted(
    selected_actions, key=lambda a: a.action_size(), reverse=True
):
    print(f"Change size {selected_action.action_size()}:", selected_action)

# %% Visualize target process execution
apply_node_styles_nx(target_process_execution)  # apply coloring + tooltip
apply_edge_styles_nx(target_process_execution)  # apply coloring + tooltip

# Draw base process execution graph
agraph = nx.nx_agraph.to_agraph(target_process_execution)
agraph.draw("figures/target_process_execution.svg", prog="dot")
