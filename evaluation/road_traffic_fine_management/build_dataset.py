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

# Dataset
dataset_cfg = cfg["dataset"]
path_xes = dataset_cfg["path_xes"]
path_dataset = dataset_cfg["path_dataset"]
path_metadata = dataset_cfg["path_metadata"]
path_sample = dataset_cfg["path_sample"]
exclude_attributes = dataset_cfg.get("exclude_attributes", [])

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]

# %% Load event log
event_log = pm4py.read_xes(path_xes)

# Select subset of data
sample_objects = event_log[viewpoint].sample(n=10000)
with open(path_sample, "w") as f:
    f.write(",".join(sample_objects))
event_log_sample = event_log[event_log[viewpoint].isin(sample_objects)]

# %% Convert event log to OCEL and Networkx graph
ocel, ocel_nx = convert_event_log_ocel(event_log_sample)

# %% Define features for dataset
viewpoint_objects = event_log_sample[viewpoint].unique()
print(f"Number of viewpoint objects selected: {len(viewpoint_objects)}")

NODE_TYPE_OBJECT = "OBJECT"
NODE_TYPE_EVENT = "EVENT"

# Define node types
object_types = list(ocel.objects[ocel.object_type_column].unique())
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
object_activities = event_log.groupby(viewpoint)["concept:name"].agg(list)


def process_outcome(obj: str):
    return int("Payment" in object_activities.loc[obj])


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
one_hot_encoding = True
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
        one_hot_encoding=one_hot_encoding,
    )

    # Set graph-level y if graph_y_function was provided
    # y_value = process_time(G)
    y_value = process_outcome(obj)
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
    one_hot_encoding=one_hot_encoding,
)

with open(path_metadata, "w") as f:
    json.dump(metadata.to_dict(), f)


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

for data in dataset:
    y_orig = data.y
    y_class = int(y_orig <= threshold)

    # Retain orignal y value on node-level
    data[viewpoint].y = tensor([y_orig] * data[viewpoint].y.size(-1)).reshape(-1, 1)

    # Update graph-level y
    data.y = y_class

# Overwrite dataset
tsave(dataset, path_dataset)

print("Classes:", Counter([d.y for d in dataset]))
