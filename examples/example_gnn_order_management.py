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
    build_event_deletion_features,
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

target_activity = "place order"
viewpoint = "orders"
y_key = "class"
dataset_name = f"example_order_management-{viewpoint.replace(' ', '_')}-{y_key.replace(' ', '_')}-pe"

dirname = os.path.dirname(__file__)
tmp_dir = os.path.join(dirname, "tmp")
path_metadata = os.path.join(tmp_dir, f"{dataset_name}-metadata.json")
path_model = os.path.join(tmp_dir, f"{dataset_name}.pth")

path_ocel = os.path.join(dirname, "ocel/data/order-management.sqlite")

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
ocel = pm4py.read_ocel2_sqlite(path_ocel)

ocel_nx = build_ocel_dfg(ocel)

# Change object type "items" to "_items" to avoid issue with HanConv
for _, d in ocel_nx.nodes(data=True):
    if d["attr"].get(ocel.object_type_column) == "items":
        d["attr"][ocel.object_type_column] = "_items"

# Convert timestamp to epoch
format_string = "%Y-%m-%d %H:%M:%S"
for _, attr in ocel_nx.nodes(data="attr"):
    if attr.get("type", "") == "EVENT":
        # attr["epoch"] = datetime.strptime(
        #     attr["ocel:timestamp"], format_string
        # ).timestamp()
        attr["epoch"] = attr["ocel:timestamp"].timestamp()


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


# %% Load model and define process outcome function

target_process_execution_id = "place_o-990363"

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

num_bins = 2  # number of bins to use for numeric attribute range
max_change_size = 2

selected_attributes = [
    "price",
    # "weight",
    # "role",
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
    ["order", "_items", "packages"],
    backward=False,
)

counterfactual_label = not process_outcome(target_process_execution)

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
]
attr_ordered = {
    node_type: [f["feature"] for f in features]
    for node_type, features in get_feature_labels_by_importance(
        explanation=target_explanation,
        feat_label_dict=feat_label_dict,
        feature_per_category=metadata.feature_per_category,
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
node_data = target_process_execution.nodes(data=True)
target_nodes_for_subst = (
    (n, node_data[n])
    for n in nodes_ordered
    if node_data[n].get("attr", {}).get(ocel.object_type_column, "") in ["products"]
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
    node_type="EVENT",
    nodes_order=nodes_ordered,
    attr_order=attr_ordered,
)

# Events that can be deleted
event_deletion_features = build_event_deletion_features(
    target_process_execution.nodes(data=True),
    nodes_order=nodes_ordered,
)

available_features = (
    object_node_attributes
    # object_substitution_features
    # + event_deletion_features
    # + event_node_attributes
)
for feature in available_features:
    print(feature)
print(f"counterfactual_label={counterfactual_label}")
print(f"Total number of features: {len(available_features)}")

# %% Run tree search algorithm to find counter factuals
from logging import DEBUG

tree_search = TreeSearchCounterFactual(
    process_outcome=process_outcome,
    max_change_size=max_change_size,
    counterfactual_label=counterfactual_label,
    log_level=DEBUG,
    log_file="logs/example_gnn_order_management.log",
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
def visualize_process_execution(
    process_execution: Graph,
    output_file_name: str = "figures/target_process_execution.svg",
):
    from networkx import nx_agraph
    from process_execution.visualization import (
        apply_node_styles_nx,
        apply_edge_styles_nx,
    )

    process_execution.construct_node_label()
    process_execution.construct_edge_label()

    apply_node_styles_nx(process_execution)
    apply_edge_styles_nx(process_execution)

    agraph = nx_agraph.to_agraph(process_execution)
    agraph.draw(output_file_name, prog="dot")


visualize_process_execution(target_process_execution_id)
