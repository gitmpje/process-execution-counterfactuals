# %% Import dependencies
import gzip
import os

import numpy as np
import pm4py
import torch

from collections import Counter
from networkx import Graph

from tree_search.tree_search import Action
from tree_search.tree_search_parallel import TreeSearchCounterFactualParallel
from tree_search.feature import (
    EventNodeDeletion,
    NodeAttributeNumeric,
    ObjectNodeSubstitution,
)

from gnn.hetero_graph_dataset import build_hetero_data
from process_execution.process_execution import (
    extract_process_execution,
)
from process_execution.utils import build_ocel_dfg

dirname = os.path.dirname(__file__)
path_ocel = os.path.join(dirname, "data/ContainerLogistics.json.gz")

tmp_dir = os.path.join(dirname, "tmp")
path_model = os.path.join(tmp_dir, "ContainerLogistics_gnn-activities.pth")

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
target_activities = ["Register Customer Order"]

ocel = pm4py.read_ocel2_json(path_ocel)

ocel_nx = build_ocel_dfg(ocel)

# %% Extract process executions
# Convert timestamp to epoch
for _, attr in ocel_nx.nodes(data="attr"):
    if attr.get("type", "") == "EVENT":
        attr["epoch"] = attr["ocel:timestamp"].timestamp()

# Extract events related to target activities
df_events = ocel.events.copy()
df_events.set_index(ocel.event_id_column, inplace=True)
df_relations = ocel.relations.copy()
df_relations.set_index(ocel.event_id_column, inplace=True)
df_events_objects = df_events.join(df_relations, rsuffix="_relations")

df_events_objects.set_index(ocel.object_id_column, append=True, inplace=True)
events_to_trace = df_events_objects[
    (df_events_objects[ocel.event_activity].isin(target_activities))
].index.unique()

print(f"Number of events selected: {len(events_to_trace)}")


def process_time(trace_graph: Graph):
    register_events = [
        d["epoch"]
        for _, d in trace_graph.nodes(data="attr")
        if d.get(ocel.event_activity, "") == "Register Customer Order"
    ]

    depart_events = [
        d["epoch"]
        for _, d in trace_graph.nodes(data="attr")
        if d.get(ocel.event_activity, "") == "Depart"
    ]

    if not (register_events and depart_events):
        return None

    return max(depart_events) - min(register_events)


trace_graphs = {}
for event, viewpoint_object in events_to_trace:
    process_execution = extract_process_execution(
        ocel_nx,
        event,
        ["Customer Order", "Handling Unit", "Container", "Transport Document"],
        # "Register Customer Order",
        backward=False,
    )
    process_execution.construct_node_label()
    process_execution.construct_edge_label()

    p_time = process_time(process_execution)
    if not p_time:
        print(f"Incomplete process execution for {event}")
        continue

    # Only include process executions with rescheduled container
    reschedule_container = any(
        n
        for n, attr in process_execution.nodes(data="attr")
        if attr.get("ocel:activity", "") == "Reschedule Container"
    )

    if not reschedule_container:
        continue

    trace_graphs[event] = {
        "process_execution": process_execution,
        "target_node": event,
        "process_time": p_time,
        "viewpoint_object": viewpoint_object,
        "reschedule_container": reschedule_container,
    }


# Calculate normalized process times
p_all = []
for trace_graph in trace_graphs.values():
    p_all.append(trace_graph["process_time"])
p_mean = np.mean(p_all)
p_std = np.std(p_all)
for trace_graph in trace_graphs.values():
    trace_graph["process_time_normalized"] = (
        trace_graph["process_time"] - p_mean
    ) / p_std

# %% Determine class
p_65 = np.quantile(p_all, 0.65)
for trace_graph in trace_graphs.values():
    trace_graph["class"] = trace_graph["process_time"] < p_65

print("Classes:", Counter([d["class"] for d in trace_graphs.values()]))

# %% Define metadata
NODE_TYPE_OBJECT = "OBJECT"
NODE_TYPE_EVENT = "EVENT"

viewpoint = "Customer Order"
y_key = "class"
path_dataset = os.path.join(
    tmp_dir,
    f"logistics-{viewpoint.replace(' ', '_')}-{y_key.replace(' ', '_')}-activities.pt",
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

node_y_mapping = {v["viewpoint_object"]: v[y_key] for v in trace_graphs.values()}


# %% Load model and define process outcome function
# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_model, weights_only=False)
model = model.to(device)
model.eval()

# %%
for target_process_execution_id in [
    k for k, v in trace_graphs.items() if not v["class"]
]:
    target_object_node = trace_graphs[target_process_execution_id]["viewpoint_object"]

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
        print(target_object_node)
        mask = torch.zeros(len(y_nodes), dtype=torch.bool)
        mask[y_nodes.index(target_object_node)] = True

        hetero_data = hetero_data.to(device)
        out = model(hetero_data.x_dict, hetero_data.edge_index_dict)

        return bool(out[mask].argmax(dim=-1).cpu().item())

    print(target_process_execution_id, process_outcome(ocel_nx))

# %% Configure counterfactual generation (tree search)
# Select process execution to generate counterfactual for
target_process_execution_id = "reg_co117"
max_change_size = 5
num_workers = 10

selected_attributes = {}

target_object_node = trace_graphs[target_process_execution_id]["viewpoint_object"]


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


print(process_outcome(ocel_nx))

target_process_execution = trace_graphs[target_process_execution_id][
    "process_execution"
]
counterfactual_label = not trace_graphs[target_process_execution_id]["class"]

# Object substitution features
object_substitution_features = []
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
event_deletion_features = [
    EventNodeDeletion(deletion_options=[[node_id]])
    for node_id, attr in target_process_execution.nodes(data="attr")
    if attr.get("type", "") == "EVENT"
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
    + object_substitution_features
    + event_deletion_features
    + event_node_attributes
)
for feature in available_features:
    print(feature)
print(f"Total number of features: {len(available_features)}")

# %% Run tree search algorithm to find counter factuals
tree_search = TreeSearchCounterFactualParallel(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counterfactual_label,
    num_workers=num_workers,
)

selected_actions = tree_search.search_layer(
    [(Action(), available_features)],
    ocel_nx,
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

    graph = trace_graphs[identifier]["process_execution"]

    apply_node_styles_nx(graph)
    apply_edge_styles_nx(graph)

    agraph = nx_agraph.to_agraph(graph)
    agraph.draw(output_file_name, prog="dot")


visualize_trace_graph(target_process_execution_id)
