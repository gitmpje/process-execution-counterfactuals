# %% Import dependencies
import os
import pm4py

from collections import Counter
from networkx import Graph
from numpy import arange

from analysis.branch_and_bound import (
    BranchAndBoundCounterFactual,
    EventNodeDeletion,
    NodeAttributeNumeric,
    ObjectNodeSubstitution,
)
from analysis.process_execution import extract_process_execution, ProcessExecution
from analysis.utils import build_ocel_dfg

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
df_events.set_index("ocel:eid", inplace=True)
df_relations = ocel.relations.copy()
df_relations.set_index("ocel:eid", inplace=True)
df_events_objects = df_events.join(df_relations, rsuffix="_relations")

events_to_trace = df_events_objects[
    (df_events_objects["ocel:type"].isin(target_object_types))
].index.values

print(f"Number of events selected: {len(events_to_trace)}")


def determine_class_event_attribute(trace_graph: Graph):
    for _, data in trace_graph.nodes(data="attr"):
        if (
            data.get("ocel:activity", "") == selected_activity
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

# %% Configure counterfactual generation (branch & bound)
# Select process execution to generate counterfactual for
target_process_execution_id = "151"

selected_event_attributes = {
    "temperature": arange(0, 1.01, 0.5),
    "quantity": range(0, 1001, 500),
}
max_changes = 10
counter_factual_label = not trace_graphs[target_process_execution_id]["class"]
num_workers = 10


# Define dummy function that determines process outcome
# Knowing the 'root cause' event attribute
def process_outcome(p: ProcessExecution):
    for _, data in p.nodes(data="attr"):
        if (
            data.get("ocel:activity", "") == selected_activity
            and data.get(selected_attribute, 0) < 0.25
        ):
            return False
    return True


target_process_execution = trace_graphs[target_process_execution_id][
    "process_execution"
]

# One feature for all object node substitution options
allowed_substitutions = {}
for node_id, data in target_process_execution.nodes(data=True):
    if data["attr"].get("type", "") != "OBJECT":
        continue

    allowed_substitutions[node_id] = [
        (subst_id, subst_data)
        for subst_id, subst_data in ocel_nx.nodes(data=True)
        if subst_id != node_id
        and subst_data["attr"].get("ocel:type", "") == data["attr"].get("ocel:type", "")
        and subst_data["attr"].get("capability", "")
        == data["attr"].get("capability", "")
    ]

object_substitution_feature = ObjectNodeSubstitution(
    allowed_substitutions=allowed_substitutions,
)

# Events that can be deleted
event_deletion_feature = EventNodeDeletion(
    allowed_deletions=[
        node_id
        for node_id, attr in target_process_execution.nodes(data="attr")
        if attr.get("type", "") == "EVENT"
    ]
)

# Features for event node attributes
event_node_attributes = [
    NodeAttributeNumeric(
        node_id=node_id,
        attribute_name=attr_name,
        value_range=selected_event_attributes[attr_name],
    )
    for node_id, attr in target_process_execution.nodes(data="attr")
    if attr.get("type", "") == "EVENT"
    for attr_name in attr.keys()
    if attr_name in selected_event_attributes
]

branch_and_bound = BranchAndBoundCounterFactual(
    process_outcome=process_outcome,
    max_changes=max_changes,
    counterfactual_label=counter_factual_label,
    num_workers=num_workers,
)

# %% Run branch and bound algorithm to find counter factuals
available_features = [
    object_substitution_feature,
    event_deletion_feature,
] + event_node_attributes
for feature in available_features:
    print(feature)

print(
    "Maximum number of actions to evaluate: ",
    branch_and_bound.maximum_number_of_actions(available_features),
)

selected_actions = []
selected_actions = branch_and_bound.find_counterfactuals(
    target_process_execution,
    available_features,
)

# %% Display results
print("Number of selected actions:", end=" ")
print(len(selected_actions))

for selected_action in selected_actions:
    print(
        [
            f"{feature}: {change_value}"
            for feature, change_value in selected_action.node_attributes_modification.items()
            if change_value != 0
        ],
        [
            (subst[0], subst[1][0] if subst[1] else "")
            for v in selected_action.object_substitution.values()
            for subst in v
            if subst
        ],
    )
