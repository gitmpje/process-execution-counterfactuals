# %% Import dependencies
import gzip
import os

import numpy as np
import pm4py
import torch

from collections import Counter
from networkx import Graph
from numpy import arange

from tree_search.action_set import ActionSet
from tree_search.tree_search import TreeSearchCounterFactual
from tree_search.action import (
    EventNodeDeletion,
    NodeAttributeNumeric,
    ObjectNodeSubstitution,
)
from gnn.hetero_graph_data import build_hetero_data, build_hetero_dataset
from process_execution.process_execution import (
    extract_process_execution,
    ProcessExecution,
)
from process_execution.utils import build_ocel_dfg

PROCESS_EXECUTION_LEVEL = False

dirname = os.path.dirname(__file__)
path_ocel = os.path.join(dirname, "data/example_DB1_ocel.json.gz")

tmp_dir = os.path.join(dirname, "tmp")
if PROCESS_EXECUTION_LEVEL:
    path_model = os.path.join(tmp_dir, "example_DB1-activities-pe.pth")
else:
    path_model = os.path.join(tmp_dir, "example_DB1-activities.pth")

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
target_object_types = ["PackingUnit"]
target_event = "Aggregation-ADD"

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
    & (df_events_objects[ocel.event_activity] == target_event)
].index.unique()

print(f"Number of events selected: {len(events_to_trace)}")


def determine_class_quality(G: Graph, event: str):
    return int(G.nodes()[event]["attr"].get("averageQuality") >= 1.0)


trace_graphs = {}
for event, viewpoint_object in events_to_trace:
    process_execution = extract_process_execution(
        ocel_nx,
        event,
        ["ProductionLot", "PackingUnit"],
        "Object-creating_class_instance",
    )
    process_execution.construct_node_label()
    process_execution.construct_edge_label()

    trace_graphs[event] = {
        "process_execution": process_execution,
        "class": determine_class_quality(ocel_nx, event),
        "viewpoint_object": viewpoint_object,
    }

print("Classes:", Counter([d["class"] for d in trace_graphs.values()]))

# %% Create and store HeteroData dataset
NODE_TYPE_OBJECT = "OBJECT"
NODE_TYPE_EVENT = "EVENT"

viewpoint = "PackingUnit"
y_key = "class"

path_dataset = os.path.join(
    tmp_dir,
    f"example_DB1-{viewpoint.replace(' ', '_')}-{y_key.replace(' ', '_')}-activities.pt",
)

# Define node types
object_types = list(ocel.objects[ocel.object_type_column].unique())
event_types = []  # list(ocel.events[ocel.event_activity].unique())
event_object_types = object_types + event_types

# Define activities
activities = list(ocel.events[ocel.event_activity].unique())

# Define numeric node attributes
object_num_keys = {}
object_num_keys[NODE_TYPE_OBJECT] = ocel.objects.select_dtypes(
    include=[np.number]
).columns
for t in object_types:
    object_num_keys[t] = (
        ocel.objects[ocel.objects[ocel.object_type_column] == t]
        .select_dtypes(include=[np.number])
        .dropna(axis=1)
        .columns
    )

event_num_keys = {}
event_num_keys[NODE_TYPE_EVENT] = ocel.events.select_dtypes(include=[np.number]).columns
for t in event_types:
    event_num_keys[t] = (
        ocel.events[ocel.events[ocel.event_activity] == t]
        .select_dtypes(include=[np.number])
        .columns
    )

node_num_keys = {
    NODE_TYPE_OBJECT: object_num_keys,
    NODE_TYPE_EVENT: event_num_keys,
}

# %% Load model and define process outcome function
# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_model, weights_only=False)
model = model.to(device)
model.eval()

# %% Configure counterfactual generation (tree search)
# Select process execution to generate counterfactual for
target_process_execution_id = "14602"
max_change_size = 5

selected_attributes = {
    "temperature": arange(0, 1.01, 0.5),
    "quantity": range(0, 1001, 500),
    "process_yield": arange(0, 1.01, 0.5),
}

target_object_node = trace_graphs[target_process_execution_id]["viewpoint_object"]

if PROCESS_EXECUTION_LEVEL:

    @torch.no_grad()
    def process_outcome(p: ProcessExecution) -> bool:
        """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

        Args:
            p (ProcessExecution): The process execution to classify.
        Returns:
            float: The predicted value.
        """
        graph_map = {"_tmp": {"process_execution": p, y_key: np.nan}}
        dataset, _, _, _, _ = build_hetero_dataset(
            graph_map,
            node_num_keys,
            ocel.object_type_column,
            ocel.event_activity,
            viewpoint,
            y_key,
            activities,
        )

        data = dataset[0].to(device)
        out = model(data.x_dict, data.edge_index_dict)

        return bool(out.argmax(dim=-1).cpu().item())
else:

    @torch.no_grad()
    def process_outcome(ocel_graph: Graph) -> bool:
        """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

        Args:
            ocel_graph (Graph): The process execution to classify.
        Returns:
            float: The predicted value.
        """
        hetero_data, _, _, y_nodes = build_hetero_data(
            ocel_graph,
            node_num_keys,
            ocel.object_type_column,
            ocel.event_activity,
            viewpoint,
            activities=activities,
        )

        mask = torch.zeros(len(y_nodes), dtype=torch.bool)
        mask[y_nodes.index(target_object_node)] = True

        hetero_data = hetero_data.to(device)
        out = model(hetero_data.x_dict, hetero_data.edge_index_dict)

        return bool(out[mask].argmax(dim=-1).cpu().item())


target_process_execution = trace_graphs[target_process_execution_id][
    "process_execution"
]
counterfactual_label = not trace_graphs[target_process_execution_id]["class"]

# Object substitution actions
object_substitution_actions = []
for node_id, node_data in target_process_execution.nodes(data=True):
    if node_data["attr"].get("type", "") != "OBJECT":
        continue

    # Only allow substitution of production resources
    if node_data["attr"].get(ocel.object_type_column, "") != "ProductionResource":
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

    object_substitution_actions.append(
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
node_deletion_actions = [
    EventNodeDeletion(deletion_options=[[node_id]])
    for node_id, attr in target_process_execution.nodes(data="attr")
    if attr.get("type", "") == "EVENT"
]

# Actions for event node attributes
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

# Actions for object node attributes
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

available_actions = object_node_attributes  # object_substitution_actions + node_deletion_actions + event_node_attributes
for action in available_actions:
    print(action)
print(f"Total number of actions: {len(available_actions)}")

# %% Run tree search algorithm to find counter factuals
tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counterfactual_label,
)

selected_action_sets = tree_search.search_layer(
    [(ActionSet(), available_actions)],
    target_process_execution if PROCESS_EXECUTION_LEVEL else ocel_nx,
)

# %% Display results
print(f"Number of selected action sets: {len(selected_action_sets)}")
for selected_action in sorted(
    selected_action_sets, key=lambda a: a.action_size(), reverse=True
):
    print(f"Change size {selected_action.action_size()}:", selected_action)
