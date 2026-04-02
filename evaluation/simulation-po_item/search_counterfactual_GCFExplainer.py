# %% Import dependencies
import json
import os
import pm4py
import torch
import yaml

from copy import deepcopy
from networkx import Graph
from random import seed
from torch_geometric.data import HeteroData

from gnn.gcf_counterfactual import gcf_explain, gcf_explain_global

from gnn.hetero_graph_data import build_hetero_data
from gnn.utils import Metadata
from process_execution.process_execution import extract_process_execution

from utils import _replace_scenario_prefix, visualize_process_execution

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Replace $SCENARIO_PREFIX tokens in config
SCENARIO_PREFIX = os.environ.get("SCENARIO_PREFIX", "scenario_03")
if SCENARIO_PREFIX is not None:
    cfg = _replace_scenario_prefix(cfg, SCENARIO_PREFIX)

# Dataset
dataset_cfg = cfg["dataset"]
path_ocel = dataset_cfg["path_ocel"]
path_labels = dataset_cfg["path_labels"]
path_metadata = dataset_cfg["path_metadata"]
normalize = dataset_cfg.get("normalize", False)
one_hot_encoding = dataset_cfg.get("one_hot_encoding", False)
add_reverse_edges = dataset_cfg.get("add_reverse_edges", False)

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]
process_execution_object_types = process_execution_cfg["object_types"]
process_execution_target_activity = process_execution_cfg.get("target_activity")
trace_backward = process_execution_cfg.get("trace_backward", False)

# GNN
gnn_cfg = cfg["gnn"]
path_model = gnn_cfg["path_model"]
random_seed = gnn_cfg.get("random_seed", 0)

torch.manual_seed(random_seed)
seed(random_seed)

# Counterfactual search
counterfactual_cfg = cfg["counterfactual"]
viewpoint_event_label = counterfactual_cfg["viewpoint_event_label"]
max_change_size = counterfactual_cfg["max_change_size"]

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

# %% Load OCEL
ocel = pm4py.read_ocel2_json(path_ocel)

# Convert timestamp to epoch
ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

# %% Convert OCEL to Networkx graph
ocel_nx = pm4py.convert_ocel_to_networkx(ocel)

# %% Load model and define process outcome function
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_model, weights_only=False)
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
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
        add_reverse_edges=metadata.add_reverse_edges,
    )

    data = data.to(device)

    try:
        # For a single graph, create a batch vector of zeros (all nodes belong to graph 0)
        batch_dict = {
            node_type: torch.zeros(
                data[node_type].num_nodes if data[node_type].num_nodes else 0,
                dtype=torch.long,
                device=device,
            )
            for node_type in metadata.node_types
        }
        out = model(data.x_dict, data.edge_index_dict, batch_dict)

    except Exception as e:
        print(f"Error occurred while processing graph: {e}")
        print(data)
        raise e

    return bool(out.argmax(dim=-1).cpu().item())


# Determine viewpoint_event_id from labels file + config label
with open(path_labels, "r") as f:
    labels_data = json.load(f)

if "viewpoint_event_labels" not in labels_data:
    raise KeyError(f"Missing 'viewpoint_event_labels' key in {path_labels}")

viewpoint_event_ids = []
for event_id, event_label in labels_data["viewpoint_event_labels"].items():
    if event_label == viewpoint_event_label:
        process_execution = extract_process_execution(
            ocel_nx,
            event_id,
            object_types=process_execution_object_types,
            target_activity_type=process_execution_target_activity,
            backward=trace_backward,
        )
        if process_outcome(process_execution) == viewpoint_event_label:
            viewpoint_event_ids.append(event_id)

if not viewpoint_event_ids:
    raise ValueError(
        f"No event found in {path_labels} with actual and predicted label {viewpoint_event_label}"
    )
viewpoint_event_id = viewpoint_event_ids[0]

# %% Select graph to explain
# Extract target process execution
data_dict = {}
for event_id in viewpoint_event_ids:
    process_execution = deepcopy(
        extract_process_execution(
            ocel_nx,
            event_id,
            object_types=process_execution_object_types,
            target_activity_type=process_execution_target_activity,
            backward=trace_backward,
        )
    )

    data, _, _, _, feat_label_dict, node_label_dict = build_hetero_data(
        graph=process_execution,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=ocel.object_type_column,
        event_activity_col=ocel.event_activity,
        viewpoint=metadata.viewpoint,
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
        add_reverse_edges=metadata.add_reverse_edges,
    )

    data_dict[event_id] = (data, node_label_dict)


# %% GCFExplainer local
@torch.no_grad
def predict(data: HeteroData) -> int:
    """Return the predicted class index (int) for a single graph."""
    batch_dict = {
        node_type: torch.zeros(
            data[node_type].num_nodes if data[node_type].num_nodes else 0,
            dtype=torch.long,
            device=device,
        )
        for node_type in metadata.node_types
    }
    out = model(data.x_dict, data.edge_index_dict, batch_dict)

    return int(out.argmax(dim=-1).cpu().item())


print("counterfactual_label =", viewpoint_event_label)
data = data_dict[viewpoint_event_id][0]
result = gcf_explain(
    predict,
    data,
    target_class=int(viewpoint_event_label),
    max_distance=3,
    # num_steps=1000,
)


# %% Modify process execution graph
def apply_counterfactual_edits_to_nx_graph(
    nx_graph: Graph,
    edits: list[dict],
    node_label_dict: dict[str, list],
    add_node_id_template: str = "cf_{node_type}_{next_idx}",
) -> Graph:
    """Apply a GCF counterfactual edit list to a NetworkX process execution graph.

    `node_label_dict` maps hetero node type -> list of original nx node labels.
    Each edit in `edits` is one of:
      - remove_node: {'action':'remove_node','node_type', 'node_idx'}
      - add_node: {'action':'add_node','node_type','features'}
      - remove_edge: {'action':'remove_edge','edge_type',(src_type,rel,dst_type),'src',src_idx,'dst',dst_idx}
      - add_edge: similar to remove_edge + 'features'

    Returns a modified copy of `nx_graph`.
    """
    g = deepcopy(nx_graph)

    for edit in edits:
        action = edit.get("action")

        if action == "remove_node":
            ntype = edit["node_type"]
            nidx = edit["node_idx"]
            if ntype not in node_label_dict or nidx >= len(node_label_dict[ntype]):
                raise KeyError(f"Unknown remove_node mapping: {ntype}[{nidx}]")
            nx_node = node_label_dict[ntype][nidx]
            if g.has_node(nx_node):
                g.remove_node(nx_node)

        elif action == "add_node":
            ntype = edit["node_type"]
            # Create a unique synthetic id for a new node.
            existing_ids = [
                n
                for n, d in g.nodes(data=True)
                if d.get("attr", {}).get("type") == ntype
            ]
            next_idx = len(existing_ids)
            nx_node = add_node_id_template.format(node_type=ntype, next_idx=next_idx)
            # If collision, increment until unique.
            while g.has_node(nx_node):
                next_idx += 1
                nx_node = add_node_id_template.format(
                    node_type=ntype, next_idx=next_idx
                )
            g.add_node(nx_node, attr={"type": ntype})

        elif action in {"remove_edge", "add_edge"}:
            edge_type = edit.get("edge_type")
            if not edge_type or len(edge_type) != 3:
                raise KeyError(f"Invalid edge_type in edit: {edit}")
            src_type, rel, dst_type = edge_type
            src_idx = edit.get("src")
            dst_idx = edit.get("dst")
            if src_type not in node_label_dict or src_idx >= len(
                node_label_dict[src_type]
            ):
                raise KeyError(f"Unknown src mapping: {src_type}[{src_idx}]")
            if dst_type not in node_label_dict or dst_idx >= len(
                node_label_dict[dst_type]
            ):
                raise KeyError(f"Unknown dst mapping: {dst_type}[{dst_idx}]")
            src_node = node_label_dict[src_type][src_idx]
            dst_node = node_label_dict[dst_type][dst_idx]

            if action == "remove_edge":
                if g.has_edge(src_node, dst_node):
                    g.remove_edge(src_node, dst_node)
            else:
                # add_edge
                edge_attr = {"attr": {"type": rel}}
                g.add_edge(src_node, dst_node, **edge_attr)

        else:
            raise ValueError(f"Unsupported edit action: {action}")

    return g


edits = result.edits
if edits:
    target_process_execution = deepcopy(
        extract_process_execution(
            ocel_nx,
            viewpoint_event_id,
            object_types=process_execution_object_types,
            target_activity_type=process_execution_target_activity,
            backward=trace_backward,
        )
    )
    visualize_process_execution(
        target_process_execution, f"data/{SCENARIO_PREFIX}-target_pe.svg"
    )

    node_label_dict = data_dict[viewpoint_event_id][1]
    cf_process_execution = apply_counterfactual_edits_to_nx_graph(
        target_process_execution,
        edits,
        node_label_dict,
    )
    visualize_process_execution(
        cf_process_execution, f"data/{SCENARIO_PREFIX}-cf_pe.svg"
    )
    print("Counterfactual process execution graph has been created.")
else:
    print("No counterfactual edits found; no graph modification applied.")

# %% GCFExplainer global
global_result = gcf_explain_global(
    predict,
    list(data_dict.values()),
    target_class=int(viewpoint_event_label),
    max_distance=3,
    summary_size=10,
)
