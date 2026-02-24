# %% Import dependencies
import os
import pm4py

from collections import Counter
from networkx import Graph

from tree_search.feature_helpers import (
    build_object_substitution_features,
    build_node_attribute_numeric,
)
from tree_search.tree_search import Action, TreeSearchCounterFactual
from process_execution.process_execution import (
    extract_process_execution,
    ProcessExecution,
)
from process_execution.utils import build_ocel_dfg

path_ocel = "data/example_object_substitution_ocel.json"

dirname = os.path.dirname(__file__)

# %% Load OCEL and build DFG with aggregation edges
target_object_types = ["PackingUnit"]

ocel = pm4py.read_ocel2_json(os.path.join(dirname, path_ocel))

selected_aggregation_activity_qualifier = [
    ("Aggregation-ADD", "childObject"),
]
ocel_nx = build_ocel_dfg(ocel, selected_aggregation_activity_qualifier)

# %% Extract process executions
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


def determine_class_quality(G: Graph, event: str):
    return G.nodes()[event]["attr"].get("averageQuality") >= 1.0


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
        "class": determine_class_quality(ocel_nx, event),
    }


print("Classes:", Counter([d["class"] for d in trace_graphs.values()]))

# %% Configure counterfactual generation (tree search)
# Select process execution to generate counterfactual for
target_process_execution_id = "198"

discretized_event_attributes = {
    "temperature": (0.5, 1.01),
    "quantity": (500, 1001),
}
max_change_size = 10
counter_factual_label = not trace_graphs[target_process_execution_id]["class"]


# Define dummy function that determines process outcome
# Knowing that DB2 is the root cause of the lower quality
def process_outcome(p: ProcessExecution):
    return "DB2" not in p.nodes()


target_process_execution = trace_graphs[target_process_execution_id][
    "process_execution"
]


# Object substitution features
def _check_capability(node_attr, subst_attr):
    return node_attr.get("capability", "") == subst_attr.get("capability", "")


target_nodes_for_subst = (
    (n, d)
    for n, d in target_process_execution.nodes(data=True)
    if d.get("attr", {}).get(ocel.object_type_column, "") == "ProductionResource"
)

object_substitution_features = build_object_substitution_features(
    target_nodes=target_nodes_for_subst,
    ocel_nodes=ocel_nx.nodes(data=True),
    graph=target_process_execution,
    check=_check_capability,
    object_type_column=ocel.object_type_column,
    discretized_event_attributes=discretized_event_attributes,
)

# Features for event node attributes
event_node_attributes = build_node_attribute_numeric(
    target_nodes=target_process_execution.nodes(data=True),
    selected_event_attributes=discretized_event_attributes,
    node_type="EVENT",
)

available_features = object_substitution_features + event_node_attributes
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
