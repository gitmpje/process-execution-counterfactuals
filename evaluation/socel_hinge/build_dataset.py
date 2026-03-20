# %%
import json
import matplotlib.pyplot as plt
import os
import pm4py
import yaml

from collections import Counter
from networkx import Graph
from numpy import array, isnan, linspace
from pandas import Series
from scipy.stats import gaussian_kde
from torch import save as tsave, tensor

from gnn.utils import (
    build_process_execution_dataset,
    construct_node_cat_keys,
    construct_node_num_keys,
)

from utils import clean_ocel_dataset

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config_HingePack.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Dataset
dataset_cfg = cfg["dataset"]
path_ocel = dataset_cfg["path_ocel"]
path_dataset = dataset_cfg["path_dataset"]
path_metadata = dataset_cfg["path_metadata"]
exclude_attributes = dataset_cfg.get("exclude_attributes", [])

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]
viewpoint_activity = process_execution_cfg["viewpoint_activity"]
process_execution_object_types = process_execution_cfg["object_types"]
process_execution_target_activity = process_execution_cfg["target_activity"]

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
def process_time(process_execution_graph: Graph, event_id: str):
    events = [
        d["epoch"]
        for _, d in process_execution_graph.nodes(data="attr")
        if d["type"] == "EVENT"
    ]

    if not (events):
        return float("nan")

    return max(events) - min(events)


def process_outcome(process_execution_graph: Graph, event_id: str) -> bool:
    return int(event_id.split(">")[-1] == "true")


normalize = True
one_hot_encoding = True
dataset, metadata = build_process_execution_dataset(
    ocel_nx=ocel_nx,
    trace_object_types=process_execution_object_types,
    trace_target_activity_type=process_execution_target_activity,
    trace_backward=True,
    node_cat_keys=node_cat_keys,
    node_num_keys=node_num_keys,
    viewpoint=viewpoint,
    graph_y_function=process_time,
    events_to_trace=viewpoint_events,
    object_type_col=ocel.object_type_column,
    event_activity_col=ocel.event_activity,
    add_reverse_edges=False,
    normalize=normalize,
    one_hot_encoding=one_hot_encoding,
    path_pe_dataset=path_dataset,
)

with open(path_metadata, "w") as f:
    json.dump(metadata.to_dict(), f)

# %%
# p_classes = array([data.y for data in dataset if not isnan(data.y)])
# print("Classes:", Counter(p_classes))


# %% Determine threshold and assign final class y values to each HeteroData
def find_valleys(values):
    """
    Find indices and values of valleys in a list.
    A valley is an element strictly less than its immediate neighbors.

    :param values: List of numeric values
    :return: List of tuples (index, value) for each valley
    """
    # Input validation
    if not isinstance(values, list) or not all(
        isinstance(x, (int, float)) for x in values
    ):
        raise ValueError("Input must be a list of numbers.")

    valleys = []
    n = len(values)

    # Need at least 3 points to have a valley
    if n < 3:
        return valleys

    for i in range(1, n - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            valleys.append((i, values[i]))

    return valleys


p_times = array([data.y for data in dataset if not isnan(data.y)])
kde = gaussian_kde(p_times)
x_min = min(p_times)
x_max = max(p_times)
x = linspace(x_min, x_max, 500)
y = kde(x)

# Use first valley in KDE as threshold
threshold = x[find_valleys(list(y))[0][0]]

Series(p_times).plot(kind="kde")
plt.axvline(
    x=threshold,
    color="red",
    linestyle="--",
)
print("Classes:", Counter(p_times <= threshold))

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
