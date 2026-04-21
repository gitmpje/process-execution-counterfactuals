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
from torch import save as tsave, tensor
from scipy.stats import gaussian_kde

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
exclude_attributes = dataset_cfg.get("exclude_attributes", [])
normalize = dataset_cfg.get("normalize", False)
one_hot_encoding = dataset_cfg.get("one_hot_encoding", False)
add_reverse_edges = dataset_cfg.get("add_reverse_edges", False)

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]

# %% Load event log
event_log = pm4py.read_xes(path_xes)

# %% Convert event log to OCEL and Networkx graph
ocel, ocel_nx = convert_event_log_ocel(event_log, viewpoint)

# %% Define features for dataset
viewpoint_objects = event_log[viewpoint].unique()
print(f"Number of viewpoint objects selected: {len(viewpoint_objects)}")

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
def process_time(process_execution_graph: Graph):
    events = [
        d["epoch"]
        for _, d in process_execution_graph.nodes(data="attr")
        if d["type"] == "EVENT"
    ]

    if not (events):
        return float("nan")

    return max(events) - min(events)


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
        add_reverse_edges=add_reverse_edges,
        normalize=normalize,
        one_hot_encoding=one_hot_encoding,
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
    one_hot_encoding=one_hot_encoding,
    add_reverse_edges=add_reverse_edges,
)

with open(path_metadata, "w") as f:
    json.dump(metadata.to_dict(), f)

# %% Determine threshold
cases = []
for i, data in enumerate(dataset):
    cases.append(
        {
            viewpoint: viewpoint_objects[i],
            "process_time": data.y,
        }
    )
df = DataFrame(cases)

mapping = event_log.set_index(viewpoint)["case:Specialism code:2"]
mapping = mapping[~mapping.index.duplicated(keep="first")]
df["case:Specialism code:2"] = df[viewpoint].map(mapping)

# Define groups
data_group_1 = df[df["case:Specialism code:2"] == 7.0]["process_time"]
data_group_2 = df[df["case:Specialism code:2"] != 7.0]["process_time"]

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

data_group_1.plot(kind="kde", label="'Specialism code:2' = 7")
data_group_2.plot(kind="kde", label="'Specialism code:2' != 7")
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
    label_dict = {"viewpoint_object_labels": {}}
    for i, data in enumerate(dataset):
        label_dict["viewpoint_object_labels"][viewpoint_objects[i]] = data.y

    with open(path_labels, "w") as f:
        label_dict = json.dump(label_dict, f, indent=2)
