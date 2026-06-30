# %%
import json
import os
import pm4py
import yaml

from pm4py.objects.ocel.exporter.jsonocel import exporter
from torch import save as tsave

from gnn.utils import (
    build_hetero_data,
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
path_sample = dataset_cfg["path_sample"]
exclude_attributes = dataset_cfg.get("exclude_attributes", [])
normalize = dataset_cfg.get("normalize", False)
one_hot_encoding = dataset_cfg.get("one_hot_encoding", False)
add_reverse_edges = dataset_cfg.get("add_reverse_edges", False)

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]
process_execution_object_types = process_execution_cfg["object_types"]

# %% Load event log
event_log = pm4py.read_xes(path_xes)

# Select subset of data
sample_objects = event_log[viewpoint].sample(n=1000)
with open(path_sample, "w") as f:
    f.write(",".join(sample_objects))
event_log_sample = event_log[event_log[viewpoint].isin(sample_objects)]

# %% Convert event log to OCEL and Networkx graph
ocel, ocel_nx = convert_event_log_ocel(event_log_sample, viewpoint)

if path_ocel:
    exporter.apply(ocel, path_ocel, variant=exporter.Variants.OCEL20_STANDARD)

# %% Define features for dataset
viewpoint_objects = event_log_sample[viewpoint].unique()
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
case_accepted = (
    event_log_sample[["case:concept:name", "case:Accepted"]]
    .drop_duplicates()
    .set_index("case:concept:name")
)


def process_outcome(obj: str):
    return int(case_accepted.loc[obj, "case:Accepted"])


dataset = []
node_types_set = set()
edge_types_set = set()
feat_label_dict = {}
for idx, obj in enumerate(viewpoint_objects):
    events = set([e for e, _ in ocel_nx.in_edges(obj)])
    objects = set([o for e in events for _, o, d in ocel_nx.out_edges(e, data="attr") if d["type"] == "E2O"])
    G = ocel_nx.subgraph(events | objects)

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
    add_reverse_edges=add_reverse_edges,
)

with open(path_metadata, "w") as f:
    json.dump(metadata.to_dict(), f)

# %% Store labels per viewpoint object
path_labels = dataset_cfg.get("path_labels")
if path_labels:
    label_dict = {"viewpoint_object_labels": {}, "viewpoint_event_labels": {}}
    for i, data in enumerate(dataset):
        label_dict["viewpoint_object_labels"][viewpoint_objects[i]] = data.y

    with open(path_labels, "w") as f:
        label_dict = json.dump(label_dict, f, indent=2)
