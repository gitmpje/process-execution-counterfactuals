# %% Import dependencies
import json
import gzip
import networkx as nx
import os
import pm4py
import torch

from collections import Counter
from networkx import Graph

from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
    HeteroExplanation,
)

from tree_search.feature_helpers import (
    build_object_substitution_features,
    build_event_deletion_features,
    build_node_attribute_features,
    construct_attribute_spec_dict,
)
from tree_search.feature_selection import (
    get_nodes_by_importance,
    get_feature_labels_by_importance,
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
from gnn.hetero_graph_data import build_hetero_data
from gnn.utils import Metadata

target_object_types = ["PackingUnit"]
target_activity = "Aggregation-ADD"
viewpoint = "PackingUnit"
y_key = "class"
dataset_name = f"example_DB1-{viewpoint.replace(' ', '_')}-{y_key.replace(' ', '_')}-pe"

dirname = os.path.dirname(__file__)
tmp_dir = os.path.join(dirname, "tmp")
path_pe_dataset = os.path.join(
    tmp_dir,
    f"{dataset_name}.pt",
)
path_metadata = os.path.join(tmp_dir, f"{dataset_name}-metadata.json")
path_pe_model = os.path.join(tmp_dir, f"{dataset_name}.pth")

path_ocel = os.path.join(dirname, "data/example_DB1_ocel.json.gz")

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

target_process_execution_id = "14602"

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_pe_model, weights_only=False)
model = model.to(device)
model.eval()


def generate_explanation(p: ProcessExecution, event: str) -> HeteroExplanation:
    data, _, _, _, feat_label_dict, node_label_dict = build_hetero_data(
        graph=p,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=ocel.object_type_column,
        event_activity_col=ocel.event_activity,
        viewpoint=metadata.viewpoint,
        node_y_mapping={event: None},
        normalize=metadata.normalized,
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
        node_mask_type="attributes",
        threshold_config=dict(
            threshold_type="topk",
            value=200,
        ),
    )

    data = data.to(device)

    # For a single graph, create a batch vector of zeros (all nodes belong to graph 0)
    batch_dict = {
        node_type: torch.zeros(
            data[node_type].num_nodes, dtype=torch.long, device=device
        )
        for node_type in metadata.node_types
    }

    return (
        explainer(
            x=data.x_dict,
            edge_index=data.edge_index_dict,
            batch_dict=batch_dict,
        ),
        feat_label_dict,
        node_label_dict,
    )


@torch.no_grad()
def process_outcome(p: ProcessExecution) -> bool:
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

num_bins = 2 # number of bins to use for numeric attribute range
max_change_size = 10

selected_attributes = [
    "process_yield",
    "capability",
    # "quantity",
]

attribute_spec_dict = construct_attribute_spec_dict(
    attributes = selected_attributes,
    ocel=ocel,
    node_cat_keys=metadata.node_cat_keys,
    node_num_keys=metadata.node_num_keys,
    num_bins=num_bins,
)

counter_factual_label = not process_executions[target_process_execution_id]["class"]

target_process_execution = process_executions[target_process_execution_id][
    "process_execution"
]

target_explanation, feat_label_dict, node_label_dict = generate_explanation(
    target_process_execution, target_process_execution_id,
)

nodes_ordered = [
    n["label"]
    for n in get_nodes_by_importance(
        explanation=target_explanation, node_label_dict=node_label_dict
    )
]
attr_ordered = {
    node_type: [f["feature"] for f in features]
    for node_type, features in get_feature_labels_by_importance(
        explanation=target_explanation, feat_label_dict=feat_label_dict
    ).items()
}

# Features for object node attributes
object_node_attributes = build_node_attribute_features(
    target_nodes=target_process_execution.nodes(data=True),
    attribute_spec_dict=attribute_spec_dict,
    node_type="OBJECT",
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
    object_type_column=ocel.object_type_column,
)


# Object substitution features
def _check_capability(node_attr, subst_attr):
    return node_attr.get("capability", "") == subst_attr.get("capability", "")


node_data = target_process_execution.nodes(data=True)
target_nodes_for_subst = (
    (n, node_data[n])
    for n in nodes_ordered
    if node_data[n].get("attr", {}).get(ocel.object_type_column, "")
    == "ProductionResource"
)

object_substitution_features = build_object_substitution_features(
    target_nodes=target_nodes_for_subst,
    ocel_nodes=ocel_nx.nodes(data=True),
    graph=target_process_execution,
    check=_check_capability,
    object_type_column=ocel.object_type_column,
    attribute_spec_dict=attribute_spec_dict,
)

# Features for event node attributes
event_node_attributes = build_node_attribute_features(
    target_nodes=target_process_execution.nodes(data=True),
    attribute_spec_dict=attribute_spec_dict,
    node_type="EVENT",
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
)

# Events that can be deleted
event_deletion_features = build_event_deletion_features(
    target_process_execution.nodes(data=True)
)

available_features = (
    object_node_attributes
    # + event_node_attributes + object_substitution_features
    # + event_deletion_features
)
for feature in available_features:
    print(feature)

print(f"counter_factual_label={counter_factual_label}")
print(f"Total number of features: {len(available_features)}")

# %% Run tree search algorithm to find counter factuals
tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counter_factual_label,
    log_file="logs/example_gnn_graph_classification.log",
)

selected_actions = tree_search.search_layer(
    [(Action(), available_features)],
    target_process_execution,
)

# %% Run tree search algorithm in parallel
# tree_search_parallel = TreeSearchCounterFactualParallel(
#     process_outcome=process_outcome,
#     max_change_size=max_change_size,
#     counterfactual_label=counter_factual_label,
#     num_workers=5,
# )

# selected_actions = tree_search_parallel.find_counterfactuals(
#     available_features,
#     target_process_execution,
# )

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
