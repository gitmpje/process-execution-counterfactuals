# %% Import dependencies
import json
import gzip
import os

import pm4py
import torch

from networkx import Graph

from tree_search.action import Action
from tree_search.tree_search import TreeSearchCounterFactual
from tree_search.feature_helpers import (
    build_object_substitution_features,
    build_node_deletion_features,
    build_node_attribute_features,
    construct_attribute_spec_dict,
)
from tree_search.feature_selection import (
    get_nodes_by_importance,
    get_feature_labels_by_importance,
)

from gnn.hetero_graph_data import build_hetero_data
from gnn.utils import Metadata, generate_explanation
from process_execution.process_execution import extract_process_execution
from process_execution.utils import build_ocel_dfg

target_activity = "Register Customer Order"
viewpoint = "Customer Order"
backward = False
y_key = "class"
dataset_name = (
    f"example_logistics-{viewpoint.replace(' ', '_')}-{y_key.replace(' ', '_')}-pe"
)

dirname = os.path.dirname(__file__)
tmp_dir = os.path.join(dirname, "tmp")
path_pe_dataset = os.path.join(
    tmp_dir,
    f"{dataset_name}.pt",
)
path_metadata = os.path.join(tmp_dir, f"{dataset_name}-metadata.json")
path_model = os.path.join(tmp_dir, f"{dataset_name}.pth")

path_ocel = os.path.join(dirname, "data/ContainerLogistics.json.gz")

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

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

# %% Load OCEL and build DFG with aggregation edges
ocel = pm4py.read_ocel2_json(path_ocel)

ocel_nx = build_ocel_dfg(ocel)

# Convert timestamp to epoch
format_string = "%Y-%m-%d %H:%M:%S"
for _, attr in ocel_nx.nodes(data="attr"):
    if attr.get("type", "") == "EVENT":
        attr["epoch"] = attr["ocel:timestamp"].timestamp()

# %% Load model and define process outcome function

target_process_execution_id = "reg_co493"

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
        feature_per_category=metadata.feature_per_category,
    )

    data = data.to(device)

    # For a single graph, create a batch vector of zeros (all nodes belong to graph 0)
    batch_dict = {
        node_type: torch.zeros(
            data[node_type].num_nodes if data[node_type] else 0,
            dtype=torch.long,
            device=device,
        )
        for node_type in metadata.node_types
    }
    out = model(data.x_dict, data.edge_index_dict, batch_dict)

    return bool(out.argmax(dim=-1).cpu().item())


# %% Configure counterfactual generation (tree search)
# Select process execution to generate counterfactual for

num_bins = 1  # number of bins to use for numeric attribute range
max_change_size = 10
node_importance_threshold = 0.1
attr_importance_threshold = 0.0

selected_attributes = [
    "Amount of Containers",
    "Amount of Goods",
    "Amount of Handling Units",
    "Status",
    "ocel:activity",
]

attribute_spec_dict = construct_attribute_spec_dict(
    attributes=selected_attributes,
    ocel=ocel,
    node_cat_keys=metadata.node_cat_keys,
    node_num_keys=metadata.node_num_keys,
    num_bins=num_bins,
)

# Extract target process execution
target_process_execution = extract_process_execution(
    ocel_nx,
    target_process_execution_id,
    [
        "Customer Order",
        "Handling Unit",
        "Container",
        "Transport Document",
    ],
    backward=backward,
)

target_explanation, feat_label_dict, node_label_dict = generate_explanation(
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
        feature_per_category=metadata.feature_per_category,
    ).items()
}

counterfactual_label = not process_outcome(target_process_execution)


# Features for object node attributes
object_node_attributes = build_node_attribute_features(
    target_nodes=target_process_execution.nodes(data=True),
    attribute_spec_dict=attribute_spec_dict,
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
    node_type="OBJECT",
    object_type_column=ocel.object_type_column,
)

# Object substitution features
target_nodes_for_subst = (
    (n, target_process_execution.nodes(data=True)[n])
    for n in nodes_ordered
    if target_process_execution.nodes(data=True)[n]
    .get("attr", {})
    .get(ocel.object_type_column, "")
    in [""]
)

object_substitution_features = build_object_substitution_features(
    target_nodes=target_nodes_for_subst,
    ocel_nodes=ocel_nx.nodes(data=True),
    graph=target_process_execution,
    object_type_column=ocel.object_type_column,
    attribute_spec_dict=attribute_spec_dict,
)

# Features for event node attributes
event_node_attributes = build_node_attribute_features(
    target_nodes=target_process_execution.nodes(data=True),
    attribute_spec_dict=attribute_spec_dict,
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
    node_type="EVENT",
)

# Events that can be deleted
node_deletion_features = build_node_deletion_features(
    target_process_execution.nodes(data=True),
    nodes_order=nodes_ordered,
    viewpoint=metadata.viewpoint,
    object_type_column=ocel.object_type_column,
)

available_features = (
    object_node_attributes
    # + object_substitution_features
    + node_deletion_features
    # + event_node_attributes
)
print(f"Total number of features: {len(available_features)}")
selected_features = []
for feature in available_features:
    if feature.action_space_size() > 0:
        print(feature)
        selected_features.append(feature)

print(f"counterfactual_label={counterfactual_label}")
print(f"Selected number of features (action space size > 0): {len(selected_features)}")

# %% Run tree search algorithm to find counter factuals
from logging import DEBUG

tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counterfactual_label,
    log_file="logs/example_gnn_logistics.log",
    log_level=DEBUG,
)

selected_actions = tree_search.search_layer(
    [(Action(), selected_features)],
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
    graph: Graph, output_file_name: str = "figures/target_process_execution.svg"
):
    from networkx import nx_agraph
    from process_execution.visualization import (
        apply_node_styles_nx,
        apply_edge_styles_nx,
    )

    apply_node_styles_nx(graph)
    apply_edge_styles_nx(graph)

    agraph = nx_agraph.to_agraph(graph)
    agraph.draw(output_file_name, prog="dot")


# target_process_execution.construct_node_label()
# target_process_execution.construct_edge_label()
visualize_trace_graph(target_process_execution)


# %%
from copy import deepcopy
from tree_search.feature import NodeAttributeNumeric

print(process_outcome(target_process_execution))

counterfactual_pe = deepcopy(target_process_execution)
node_attributes_modification = {
    f: min([v for v in f.action_space()])
    for f in object_node_attributes
    if isinstance(f, NodeAttributeNumeric)
}
node_attributes_modification.update(
    {f: f.category_values[0] for f in event_node_attributes}
)

node_deletion = {f: [v for v in f.action_space()][0] for f in node_deletion_features}

a = Action(
    node_attributes_modification=node_attributes_modification,
    node_deletion=node_deletion,
)

a.apply_changes(counterfactual_pe)
print(a.action_size())
print(process_outcome(counterfactual_pe))

visualize_trace_graph(counterfactual_pe, "figures/counterfactual_pe.svg")
