# %%
import json
import matplotlib.pyplot as plt
import os
import pm4py
import yaml

from collections import Counter
from networkx import Graph
from numpy import array, isnan, quantile
from pandas import Series
from torch import save as tsave, tensor

from gnn.hetero_graph_data import build_hetero_data
from gnn.utils import (
    construct_node_cat_keys,
    construct_node_num_keys,
    Metadata,
)

from utils import convert_event_log_ocel

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Node attributes to exclude
exclude_attributes = (
    [
        "ocel:eid",
        "ocel:timestamp",
        "ocel:oid",
        "ocel:type",
        "case:End date",
        "case:Start date",
    ]
    + [f"case:Start date:{i}" for i in range(1, 16)]
    + [f"case:End date:{i}" for i in range(1, 16)]
)

# Dataset
dataset_cfg = cfg.get("dataset", {})
viewpoint = dataset_cfg.get("viewpoint")
path_xes = dataset_cfg.get("path_xes")
path_dataset = dataset_cfg.get("path_dataset")
path_metadata = dataset_cfg.get("path_metadata")

# %%
event_log = pm4py.read_xes(path_xes)

# %% Convert event log to OCEL
ocel, ocel_nx = convert_event_log_ocel(event_log, viewpoint)

# %%
viewpoint_objects = event_log[viewpoint].unique()

NODE_TYPE_OBJECT = "OBJECT"
NODE_TYPE_EVENT = "EVENT"

# Define node types
object_types = [viewpoint]
event_types = []
# event_types = list(event_log["case:concept"].unique())

# Define categoric node attributes
node_cat_keys = construct_node_cat_keys(
    selected_object_types=object_types,
    selected_event_types=event_types,
    df_objects=ocel.objects,
    df_events=ocel.events,
    object_type_column=ocel.object_type_column,
    event_activity_column=ocel.event_activity,
    exclude_attributes=exclude_attributes,
)

# Define numeric node attributes
node_num_keys = construct_node_num_keys(
    selected_object_types=object_types,
    selected_event_types=event_types,
    df_objects=ocel.objects,
    df_events=ocel.events,
    object_type_column=ocel.object_type_column,
    event_activity_column=ocel.event_activity,
    exclude_attributes=exclude_attributes,
)


# %%
def process_time(process_execution_graph: Graph):
    events = [
        d["epoch"]
        for _, d in process_execution_graph.nodes(data="attr")
        if d["type"] == "EVENT"
    ]

    if not (events):
        return float("nan")

    return max(events) - min(events)


normalize = True
feature_per_category = True
dataset = []
node_types_set = set()
edge_types_set = set()
feat_label_dict = {}
for idx, obj in enumerate(viewpoint_objects):
    events = set([e for e, _ in ocel_nx.in_edges(obj)])
    nodes = events | set([obj])

    G = ocel_nx.subgraph(nodes)

    # Build HeteroData graph
    hetero_data, n_types, e_types, _, feat_labels, _ = build_hetero_data(
        graph=G,
        node_num_keys=node_num_keys,
        node_cat_keys=node_cat_keys,
        object_type_col=ocel.object_type_column,
        event_activity_col=ocel.event_activity,
        viewpoint=viewpoint,
        add_reverse_edges=False,
        normalize=normalize,
        feature_per_category=feature_per_category,
    )

    # Set graph-level y if graph_y_function was provided
    y_value = process_time(G)
    hetero_data.y = y_value

    dataset.append(hetero_data)
    node_types_set.update(n_types)
    edge_types_set.update(e_types)

    # Fill global feature/label dict if empty
    for k, v in feat_labels.items():
        if k not in feat_label_dict:
            feat_label_dict[k] = v

    if idx % 50 == 0:
        print(f"Processed {idx} process executions")

# Save dataset
tsave(dataset, path_dataset)

# Create metadata
metadata = Metadata(
    viewpoint=viewpoint,
    node_num_keys=node_num_keys,
    node_cat_keys=node_cat_keys,
    node_types=list(node_types_set),
    edge_types=list(edge_types_set),
    feat_label_dict=feat_label_dict,
    normalized=normalize,
    feature_per_category=feature_per_category,
)

with open(path_metadata, "w") as f:
    json.dump(metadata.to_dict(), f)


# %% Determine threshold and assign final class y values to each HeteroData
p_times = array([data.y for data in dataset if not isnan(data.y)])
threshold = quantile(p_times, 0.5)
print("Classes:", Counter(p_times <= threshold))

Series(p_times).plot(kind="kde")
plt.axvline(
    x=threshold,
    color="red",
    linestyle="--",
)

for data in dataset:
    y_orig = data.y
    y_class = int(y_orig <= threshold)

    # Retain orignal y value on node-level
    data[viewpoint].y = tensor([y_orig] * data[viewpoint].y.size(-1)).reshape(-1, 1)

    # Update graph-level y
    data.y = y_class

# Overwrite dataset
tsave(dataset, path_dataset)
