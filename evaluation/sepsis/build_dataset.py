# %%
import json
import matplotlib.pyplot as plt
import os
import pm4py
import yaml

from collections import Counter
from networkx import Graph
from numpy import diff, linspace, where, sign
from pandas import DataFrame
from pm4py.objects.ocel.exporter.jsonocel import exporter
from torch import save as tsave, tensor
from scipy.stats import gaussian_kde

from gnn.utils import (
    build_process_execution_dataset,
    construct_node_cat_keys,
    construct_node_num_keys,
    Metadata,
)

from utils import convert_event_log_ocel

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Dataset
dataset_cfg = cfg["dataset"]
path_xes = dataset_cfg["path_xes"]
path_ocel = dataset_cfg.get("path_ocel")
path_dataset = dataset_cfg["path_dataset"]
path_metadata = dataset_cfg["path_metadata"]
exclude_attributes = dataset_cfg.get("exclude_attributes", [])
normalize = dataset_cfg.get("normalize", False)
one_hot_encoding = dataset_cfg.get("one_hot_encoding", False)
add_reverse_edges = dataset_cfg.get("add_reverse_edges", False)

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]
process_execution_object_types = process_execution_cfg["object_types"]
process_execution_target_activity = process_execution_cfg["target_activity"]
exclude_target_activity = process_execution_cfg.get("exclude_target_activity", False)
trace_backward = process_execution_cfg["trace_backward"]

# %% Load event log
event_log = pm4py.read_xes(path_xes)

# %% Convert event log to OCEL and Networkx graph
ocel, ocel_nx = convert_event_log_ocel(event_log, viewpoint)

if path_ocel:
    exporter.apply(ocel, path_ocel, variant=exporter.Variants.OCEL20_STANDARD)

# %% Define features for dataset
viewpoint_objects = event_log[viewpoint].unique()
print(f"Number of viewpoint objects selected: {len(viewpoint_objects)}")

NODE_TYPE_OBJECT = "OBJECT"
NODE_TYPE_EVENT = "EVENT"

# Define node types
object_types = list(ocel.objects[ocel.object_type_column].unique())
event_types = []
# event_types = list(ocel.events[ocel.event_activity].unique())

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
viewpoint_events = []
viewpoint_event_object_map = {}
for obj in viewpoint_objects:
    events = set([e for e, _ in ocel_nx.in_edges(obj)])
    first_event = sorted(
        [ocel_nx.nodes(data="attr")[e] for e in events], key=lambda x: x["epoch"]
    )[0]
    viewpoint_event_object_map[first_event["ocel:eid"]] = obj
    viewpoint_events.append(first_event["ocel:eid"])

object_activities = event_log.groupby(viewpoint)["concept:name"].agg(list)


def process_outcome(process_execution_graph: Graph, event_id: str):
    obj = viewpoint_event_object_map[event_id]
    return int(process_execution_target_activity in object_activities.loc[obj])


def process_time(process_execution_graph: Graph, event_id: str):
    events = [
        d["epoch"]
        for _, d in process_execution_graph.nodes(data="attr")
        if d["type"] == "EVENT"
    ]

    if not (events):
        return float("nan")

    return max(events) - min(events)


# Build HeteroData graph
dataset, metadata = build_process_execution_dataset(
    ocel_nx=ocel_nx,
    trace_object_types=process_execution_object_types,
    trace_target_activity_type=process_execution_target_activity,
    trace_backward=trace_backward,
    node_cat_keys=node_cat_keys,
    node_num_keys=node_num_keys,
    viewpoint=viewpoint,
    graph_y_function=process_outcome,
    events_to_trace=viewpoint_events,
    object_type_col=ocel.object_type_column,
    event_activity_col=ocel.event_activity,
    add_reverse_edges=add_reverse_edges,
    normalize=normalize,
    one_hot_encoding=one_hot_encoding,
    path_pe_dataset=path_dataset,
    exclude_target_activity=exclude_target_activity,
)

# Save dataset
tsave(dataset, path_dataset)

with open(path_metadata, "w") as f:
    json.dump(metadata.to_dict(), f)

# %% Determine threshold
# cases = []
# for i, data in enumerate(dataset):
#     cases.append(
#         {
#             viewpoint: viewpoint_objects[i],
#             "process_time": data.y,
#         }
#     )
# df = DataFrame(cases)

# # mapping = event_log.set_index(viewpoint)["case:Specialism code:2"]
# # mapping = mapping[~mapping.index.duplicated(keep="first")]
# # df["case:Specialism code:2"] = df[viewpoint].map(mapping)

# # Define groups [df["case:Specialism code:2"] == 7.0]
# data_group_1 = df["process_time"]
# data_group_2 = df["process_time"]

# # Estimate KDE
# kde_group_1 = gaussian_kde(data_group_1)
# kde_group_2 = gaussian_kde(data_group_2)

# # Define a common x-range for evaluation
# x_min = df["process_time"].min() - 1
# x_max = df["process_time"].max() + 1
# x_vals = linspace(x_min, x_max, 500)

# # Evaluate KDEs
# y_group_1 = kde_group_1(x_vals)
# y_group_2 = kde_group_2(x_vals)

# # Find intersection points (where difference changes sign)
# diff_groups = y_group_1 - y_group_2
# sign_changes = where(diff(sign(diff_groups)) != 0)[0]

# Interpolate intersection points for better accuracy
# intersections = []
# for idx in sign_changes:
#     x0, x1 = x_vals[idx], x_vals[idx + 1]
#     y0, y1_diff = diff_groups[idx], diff_groups[idx + 1]

#     # Linear interpolation
#     x_intersect = x0 - y0 * (x1 - x0) / (y1_diff - y0)
#     intersections.append(x_intersect)
# threshold = intersections[0]

# from numpy import quantile

# threshold = quantile(df["process_time"].values, 0.85)

# data_group_1.plot(kind="kde", label="")
# data_group_2.plot(kind="kde", label="")
# plt.axvline(
#     x=threshold,
#     color="red",
#     linestyle="--",
# )
# plt.legend()

# print("Classes:", Counter(df["process_time"].values <= threshold))

# %% Assign final class y values to each HeteroData
# for data in dataset:
#     y_orig = data.y
#     y_class = int(y_orig <= threshold)

#     # Retain orignal y value on node-level
#     data[viewpoint].y = tensor([y_orig] * data[viewpoint].y.size(-1)).reshape(-1, 1)

#     # Update graph-level y
#     data.y = y_class

# # Overwrite dataset
# tsave(dataset, path_dataset)

# %% Store labels per viewpoint object
path_labels = dataset_cfg.get("path_labels")
if path_labels:
    label_dict = {"viewpoint_object_labels": {}, "viewpoint_event_labels": {}}
    for i, data in enumerate(dataset):
        label_dict["viewpoint_object_labels"][viewpoint_objects[i]] = data.y
        label_dict["viewpoint_event_labels"][viewpoint_events[i]] = data.y

    with open(path_labels, "w") as f:
        label_dict = json.dump(label_dict, f, indent=2)
