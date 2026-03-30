# %%
import json
import os
import pm4py
import yaml

from collections import Counter
from numpy import array, isnan
from torch import save as tsave

from gnn.utils import (
    build_process_execution_dataset,
    construct_node_cat_keys,
    construct_node_num_keys,
)

from utils import _replace_scenario_prefix

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Replace $SCENARIO_PREFIX tokens in config
scenario_prefix = os.environ.get("SCENARIO_PREFIX")
if scenario_prefix is not None:
    cfg = _replace_scenario_prefix(cfg, scenario_prefix)

# Dataset
dataset_cfg = cfg["dataset"]
path_ocel = dataset_cfg["path_ocel"]
path_dataset = dataset_cfg["path_dataset"]
path_labels = dataset_cfg["path_labels"]
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

# Convert timestamp to epoch
ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

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
dataset, metadata = build_process_execution_dataset(
    ocel_nx=ocel_nx,
    trace_object_types=process_execution_object_types,
    trace_target_activity_type=process_execution_target_activity,
    trace_backward=trace_backward,
    node_cat_keys=node_cat_keys,
    node_num_keys=node_num_keys,
    viewpoint=viewpoint,
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

# %%
with open(path_labels) as f:
    label_dict = json.load(f)

viewpoint_event_labels = {}
for i, data in enumerate(dataset):
    label = label_dict["labels"][str(i + 1)]
    data.y = int(label)
    viewpoint_event_labels[viewpoint_events[i]] = label

p_classes = array([data.y for data in dataset if not isnan(data.y)])
print("Classes:", Counter(p_classes))

# Overwrite dataset
tsave(dataset, path_dataset)

# Store labels per viewpoint event
label_dict["viewpoint_event_labels"] = viewpoint_event_labels
with open(path_labels, "w") as f:
    label_dict = json.dump(label_dict, f, indent=2)
