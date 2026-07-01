# %%
import json
import matplotlib.pyplot as plt
import os
import pm4py
import yaml

from collections import Counter
from networkx import Graph
from numpy import diff, linspace, sign, where
from pandas import DataFrame
from scipy.stats import gaussian_kde
from torch import save as tsave, tensor

from gnn.utils import (
    build_process_execution_dataset,
    construct_node_cat_keys,
    construct_node_num_keys,
)

from utils import clean_ocel_dataset

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Dataset
dataset_cfg = cfg["dataset"]
path_ocel = dataset_cfg["path_ocel"]
path_dataset = dataset_cfg["path_dataset"]
path_metadata = dataset_cfg["path_metadata"]
exclude_attributes = dataset_cfg.get("exclude_attributes", [])
normalize = dataset_cfg.get("normalize", False)
one_hot_encoding = dataset_cfg.get("one_hot_encoding", False)
add_reverse_edges = dataset_cfg.get("add_reverse_edges", False)

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]
viewpoint_activity = process_execution_cfg["viewpoint_activity"]
process_execution_object_types = process_execution_cfg["object_types"]
process_execution_target_activity = process_execution_cfg.get("target_activity")
trace_backward = process_execution_cfg.get("trace_backward", False)

# %% Load OCEL
ocel = pm4py.read_ocel2_json(path_ocel)

ocel = clean_ocel_dataset(ocel)

# %% Convert OCEL to Networkx graph
ocel_nx = pm4py.convert_ocel_to_networkx(ocel)

# %% Define features for dataset

# Extract events related to viewpoint activity
viewpoint_events = ocel.events[
    (ocel.events[ocel.event_activity] == viewpoint_activity)
][ocel.event_id_column].unique()

print(f"Number of viewpoint events selected: {len(viewpoint_events)}")

NODE_TYPE_OBJECT = "OBJECT"
NODE_TYPE_EVENT = "EVENT"

# Define node types
object_types = list(ocel.objects[ocel.object_type_column].unique())
event_types = []
# event_types = list(event_log[ocel.event_activity].unique())

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
reschedule_container = {}


def process_time(process_execution_graph: Graph, event_id: str):
    events = [
        d["epoch"]
        for _, d in process_execution_graph.nodes(data="attr")
        if d["type"] == "EVENT"
    ]

    if not (events):
        return float("nan")

    # Set flag if process execution contains "Reschedule Container" event
    reschedule_container[event_id] = any(
        n
        for n, attr in process_execution_graph.nodes(data="attr")
        if attr.get(ocel.event_activity, "") == "Reschedule Container"
    )
    return max(events) - min(events)


dataset, metadata = build_process_execution_dataset(
    ocel_nx=ocel_nx,
    trace_object_types=process_execution_object_types,
    trace_target_activity_type=process_execution_target_activity,
    trace_backward=trace_backward,
    node_cat_keys=node_cat_keys,
    node_num_keys=node_num_keys,
    viewpoint=viewpoint,
    graph_y_function=process_time,
    events_to_trace=viewpoint_events,
    object_type_col=ocel.object_type_column,
    event_activity_col=ocel.event_activity,
    add_reverse_edges=add_reverse_edges,
    normalize=normalize,
    one_hot_encoding=one_hot_encoding,
    path_pe_dataset=path_dataset,
)

with open(path_metadata, "w") as f:
    json.dump(metadata.to_dict(), f)


# %% Determine threshold
cases = []
for i, data in enumerate(dataset):
    cases.append(
        {
            viewpoint: viewpoint_events[i],
            "process_time": data.y,
        }
    )
df = DataFrame(cases)
df["reschedule_container"] = df[viewpoint].map(reschedule_container)

# Define groups
data_group_1 = df[df["reschedule_container"]]["process_time"]
data_group_2 = df[~df["reschedule_container"]]["process_time"]

# Estimate KDE
kde_group_1 = gaussian_kde(data_group_1)
kde_group_2 = gaussian_kde(data_group_2)

# Define a common x-range for evaluation
x_min = df["process_time"].min() - 1
x_max = df["process_time"].max() + 1
x_vals = linspace(x_min, x_max, 500)

# Evaluate KDEs
y_group_1 = kde_group_1(x_vals)
y_group_2 = kde_group_2(x_vals)

# Find intersection points (where difference changes sign)
diff_groups = y_group_1 - y_group_2
sign_changes = where(diff(sign(diff_groups)) != 0)[0]

# Interpolate intersection points for better accuracy
intersections = []
for idx in sign_changes:
    x0, x1 = x_vals[idx], x_vals[idx + 1]
    y0, y1_diff = diff_groups[idx], diff_groups[idx + 1]

    # Linear interpolation
    x_intersect = x0 - y0 * (x1 - x0) / (y1_diff - y0)
    intersections.append(x_intersect)
threshold = intersections[0]

data_group_1.plot(kind="kde", label="At least one 'Reschedule Container' event")
data_group_2.plot(kind="kde", label="No 'Reschedule Container' event")
plt.axvline(
    x=threshold,
    color="red",
    linestyle="--",
)
plt.legend()

print("Classes:", Counter(df["process_time"].values <= threshold))

# %% Assign final class y values to each HeteroData
for data in dataset:
    y_orig = data.y
    y_class = int(y_orig <= threshold)

    # Retain orignal y value on node-level
    data[viewpoint].y = tensor([y_orig] * data[viewpoint].y.size(-1)).reshape(-1, 1)

    # Update graph-level y
    data.y = y_class

# Overwrite dataset
tsave(dataset, path_dataset)

# %% Store labels per viewpoint object
path_labels = dataset_cfg.get("path_labels")
if path_labels:
    label_dict = {"viewpoint_event_labels": {}}
    for i, data in enumerate(dataset):
        label_dict["viewpoint_event_labels"][viewpoint_events[i]] = data.y

    with open(path_labels, "w") as f:
        label_dict = json.dump(label_dict, f, indent=2)
