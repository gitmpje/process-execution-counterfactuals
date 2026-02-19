# %% Import dependencies
import os
import pm4py

from collections import Counter
from networkx import Graph
from numpy import arange

from tree_search.feature import (
    EventNodeDeletion,
    NodeAttributeNumeric,
    ObjectNodeSubstitution,
)
from tree_search.tree_search import Action, TreeSearchCounterFactual

from process_execution.process_execution import (
    extract_process_execution,
    ProcessExecution,
)
from process_execution.utils import build_ocel_dfg

path_ocel = "data/example_event_attribute_ocel.json"

dirname = os.path.dirname(__file__)

# %% Load OCEL and build DFG with aggregation edges
target_object_types = ["PackingUnit"]

ocel = pm4py.read_ocel2_json(os.path.join(dirname, path_ocel))

selected_aggregation_activity_qualifier = [
    ("Aggregation-ADD", "childObject"),
]
ocel_nx = build_ocel_dfg(ocel, selected_aggregation_activity_qualifier)

# %% Extract process executions
# Define event attribute and activity to base classification on
selected_activity = "Object-departing-WB"
selected_attribute = "temperature"

# Extract events related to target object types
df_events = ocel.events.copy()
df_events.set_index(ocel.event_id_column, inplace=True)
df_relations = ocel.relations.copy()
df_relations.set_index(ocel.event_id_column, inplace=True)
df_events_objects = df_events.join(df_relations, rsuffix="_relations")

events_to_trace = df_events_objects[
    (df_events_objects[ocel.object_type_column].isin(target_object_types))
].index.unique()

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
        "class": determine_class_event_attribute(process_execution),
    }


print("Classes:", Counter([d["class"] for d in trace_graphs.values()]))

# %% Configure counterfactual generation (tree search)
# Select process execution to generate counterfactual for
target_process_execution_id = "151"

selected_event_attributes = {
    "temperature": arange(0, 1.01, 0.5),
    "quantity": range(0, 1001, 500),
}
max_change_size = 10

counter_factual_label = not trace_graphs[target_process_execution_id]["class"]


# Define dummy function that determines process outcome
# Knowing the 'root cause' event attribute
def process_outcome(p: ProcessExecution):
    for _, data in p.nodes(data="attr"):
        if (
            data.get(ocel.event_activity, "") == selected_activity
            and data.get(selected_attribute, 0) < 0.25
        ):
            return False
    return True


target_process_execution = trace_graphs[target_process_execution_id][
    "process_execution"
]

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
        value_range=selected_event_attributes[attr_name],
    )
    for node_id, attr in target_process_execution.nodes(data="attr")
    if attr.get("type", "") == "EVENT"
    for attr_name in attr.keys()
    if attr_name in selected_event_attributes
]

available_features = event_node_attributes
for feature in available_features:
    print(feature)

tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counter_factual_label,
)
print(
    "Maximum number of actions to evaluate: ",
    tree_search.maximum_number_of_actions(available_features),
)

# %% Run tree search algorithm to find counter factuals
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
