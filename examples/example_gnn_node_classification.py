# %% Import dependencies
import gzip
import os

import numpy as np
import pm4py
import torch

from collections import Counter
from networkx import Graph
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddMetaPaths

from branch_and_bound.branch_and_bound import Action, BranchAndBoundCounterFactual
from branch_and_bound.feature import NodeAttributeNumeric, ObjectNodeSubstitution
from gnn.han_node_level import HAN, HANConvTrainer
from gnn.hetero_graph_dataset import build_hetero_dataset
from process_execution.process_execution import (
    extract_process_execution,
    ProcessExecution,
)
from process_execution.utils import build_ocel_dfg

dirname = os.path.dirname(__file__)
path_ocel = os.path.join(dirname, "data/example_DB1_ocel.json.gz")

tmp_dir = os.path.join(dirname, "tmp")
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

events_to_trace = df_events_objects[
    (df_events_objects[ocel.object_type_column].isin(target_object_types))
].index.values

print(f"Number of events selected: {len(events_to_trace)}")


def determine_class_quality(G: Graph, event: str):
    return int(G.nodes()[event]["attr"].get("averageQuality") >= 1.0)


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
        "class": determine_class_quality(ocel_nx, event),
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

metapaths = [
    [
        ("PackingUnit", "DESCENDANTS", "ProductionLot"),
        ("ProductionLot", "INTERACTION", "ProductionResource"),
    ]
]
transforms = [
    AddMetaPaths(
        metapaths=[
            [
                ("PackingUnit", "DESCENDANTS", "ProductionLot"),
                ("ProductionLot", "INTERACTION", "ProductionResource"),
            ],
        ],
        drop_orig_edge_types=True,
        drop_unconnected_node_types=True,
    ),
    AddMetaPaths(
        metapaths=[
            [
                ("PackingUnit", "INTERACTION", "ProductionLot"),
                ("ProductionLot", "INTERACTION", "ProductionResource"),
            ],
        ],
        drop_orig_edge_types=True,
        drop_unconnected_node_types=True,
    ),
    AddMetaPaths(
        metapaths=[
            [
                ("PackingUnit", "CODEATH", "ProductionLot"),
                ("ProductionLot", "INTERACTION", "ProductionResource"),
            ],
        ],
        drop_orig_edge_types=True,
        drop_unconnected_node_types=True,
    ),
]

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
    object_num_keys[t] = (
        ocel.events[ocel.events[ocel.event_activity] == t]
        .select_dtypes(include=[np.number])
        .columns
    )

node_num_keys = {
    NODE_TYPE_OBJECT: object_num_keys,
    NODE_TYPE_EVENT: event_num_keys,
}

dataset, node_types_set, edge_types_set = build_hetero_dataset(
    trace_graphs,
    node_num_keys,
    ocel.object_type_column,
    ocel.event_activity,
    viewpoint,
    y_key,
    activities=activities,
    path_dataset=path_dataset,
)

# Apply transforms
# for i in range(len(dataset)):
#     for t in transforms:
#         try:
#             dataset[i] = t(dataset[i])
#         except AssertionError:
#             print(f"Skipping {t} for dataset item {i}")

# # Add metapaths to edge type set
# for d in dataset:
#     try:
#         edge_types_set.update(d.metapath_dict.keys())
#     except KeyError:
#         continue

# if path_dataset:
#     torch.save(dataset, path_dataset)

# %% Create train/validate/test dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 10

dataset = torch.load(path_dataset, weights_only=False)

train_idx, test_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.4,
    random_state=1,
)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=1)

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
test_ds = Subset(dataset, test_idx)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# %% Define and train model
model = HAN(
    in_channels=-1,
    out_channels=2,
    num_layers=1,
    viewpoint=viewpoint,
    metadata=[node_types_set, edge_types_set],
)
trainer = HANConvTrainer(
    model=model,
    viewpoint=viewpoint,
    device=device,
    criterion=torch.nn.CrossEntropyLoss(),
    output_type="binary",
)
trainer.train(
    train_loader,
    val_loader,
    test_loader,
    n_epochs=200,
    start_patience=50,
    # learning_rate=0.01,
)

torch.save(model, path_model)

# %% Load model and define process outcome function
# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_model, weights_only=False)
model = model.to(device)
model.eval()


@torch.no_grad()
def process_outcome(p: ProcessExecution) -> bool:
    """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

    Args:
        p (ProcessExecution): The process execution to classify.
    Returns:
        float: The predicted value.
    """
    graph_map = {"_tmp": {"process_execution": p, y_key: np.nan}}
    dataset, _, _ = build_hetero_dataset(
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


# %% Configure counterfactual generation (branch & bound)
# Select process execution to generate counterfactual for
target_process_execution_id = "14602"
counterfactual_label = not trace_graphs[target_process_execution_id]["class"]
max_changes = 10

selected_event_attributes = {
    "quantity": range(0, 1001, 500),
}

target_process_execution = trace_graphs[target_process_execution_id][
    "process_execution"
]

# Object substitution features
object_substitution_features = []
for node_id, data in target_process_execution.nodes(data=True):
    if node_id != "DB1":
        continue
    if data["attr"].get("type", "") != "OBJECT":
        continue

    if data["attr"].get(ocel.object_type_column, "") not in [
        "ProductionResource",
    ]:
        continue

    substitution_objects = [
        (subst_id, subst_data)
        for subst_id, subst_data in ocel_nx.nodes(data=True)
        if subst_data["attr"].get(ocel.object_type_column, "")
        == data["attr"].get(ocel.object_type_column, "")
        and subst_data["attr"].get("capability", "") == data["attr"].get("capability", "")
    ]

    object_substitution_features.extend(
        [
            ObjectNodeSubstitution(
                event_id=event_id,
                object_id=node_id,
                substitution_objects=substitution_objects,
            )
            for event_id, _, attr in target_process_execution.in_edges(node_id, data="attr")
            if attr.get("type", "") == "E2O"
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


available_features = event_node_attributes + object_substitution_features
for feature in available_features:
    print(feature)
print(f"Total number of features: {len(available_features)}")

# %% Run branch and bound algorithm to find counter factuals
branch_and_bound = BranchAndBoundCounterFactual(
    process_outcome=process_outcome,
    max_changes=max_changes,
    counterfactual_label=counterfactual_label,
)

branch_and_bound.enumerate(
    Action(),
    available_features,
    target_process_execution,
)
selected_actions = branch_and_bound.selected_actions

# %% Display results
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

# %%
import networkx as nx
from process_execution.visualization import (
    apply_node_styles_nx,
    apply_edge_styles_nx,
)

target_process_execution = trace_graphs["40217"]["process_execution"]
apply_node_styles_nx(target_process_execution)  # apply coloring + tooltip
apply_edge_styles_nx(target_process_execution)  # apply coloring + tooltip

# Draw process execution graph
agraph = nx.nx_agraph.to_agraph(target_process_execution)
agraph.draw("figures/target_process_execution.svg", prog="dot")
