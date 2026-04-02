# %% Import dependencies
import json
import os
import pm4py
import torch
import yaml

from copy import deepcopy
from networkx import Graph
from random import seed
from torch_geometric.utils import to_dgl

from NSEG.explainer.explainer_NSEG import NSEG
from NSEG.GCN.model import GCN

from gnn.hetero_graph_data import build_hetero_data
from gnn.nseg_counterfactual import generate_counterfactual
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
path_model_gcn = path_model.replace(".pth", "-gcn.pth")
num_layers = gnn_cfg["num_layers"]
random_seed = gnn_cfg.get("random_seed", 0)

torch.manual_seed(random_seed)
seed(random_seed)

# Counterfactual search
counterfactual_cfg = cfg["counterfactual"]
viewpoint_event_label = counterfactual_cfg["viewpoint_event_label"]
depth_first = counterfactual_cfg.get("depth_first")
num_bins = counterfactual_cfg["num_bins"]
max_change_size = counterfactual_cfg["max_change_size"]
node_importance_threshold = counterfactual_cfg["node_importance_threshold"]
attr_importance_threshold = counterfactual_cfg["attr_importance_threshold"]

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

num_classes = 2
feat_dim = 22
hidden_dims = [64, 64]
num_gnn_layers = len(hidden_dims)
model_gcn = GCN(
    dim_input=feat_dim,
    dim_hidden=hidden_dims,
    num_classes=num_classes,
    dropout=0.5,
    num_layers=num_gnn_layers,
    mode="graph",
)
model_gcn_state = torch.load(path_model_gcn, weights_only=True)
model_gcn.load_state_dict(model_gcn_state)

model_gcn = model_gcn.to(device)
model_gcn.eval()


@torch.no_grad()
def process_outcome_dgl(p: Graph) -> bool:
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

    data = to_dgl(data.to_homogeneous())
    data = data.to(device)
    feat = data.ndata["x"].to(device)

    out = model_gcn(data, feat).squeeze(0)
    return bool(out.argmax(dim=-1).cpu().item())


# Determine viewpoint_event_id from labels file + config label
with open(path_labels, "r") as f:
    labels_data = json.load(f)

if "viewpoint_event_labels" not in labels_data:
    raise KeyError(f"Missing 'viewpoint_event_labels' key in {path_labels}")

viewpoint_event_id = None
for event_id, event_label in labels_data["viewpoint_event_labels"].items():
    if event_label == viewpoint_event_label:
        process_execution = extract_process_execution(
            ocel_nx,
            event_id,
            object_types=process_execution_object_types,
            target_activity_type=process_execution_target_activity,
            backward=trace_backward,
        )
        if process_outcome_dgl(process_execution) == viewpoint_event_label:
            viewpoint_event_id = event_id
        break

if viewpoint_event_id is None:
    raise ValueError(
        f"No event found in {path_labels} with actual and predicted label {viewpoint_event_label}"
    )

# %% Select graph to explain
# Extract target process execution
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

counterfactual_label = not process_outcome_dgl(target_process_execution)

data, _, _, _, feat_label_dict, node_label_dict = build_hetero_data(
    graph=target_process_execution,
    node_num_keys=metadata.node_num_keys,
    node_cat_keys=metadata.node_cat_keys,
    object_type_col=ocel.object_type_column,
    event_activity_col=ocel.event_activity,
    viewpoint=metadata.viewpoint,
    normalize=metadata.normalized,
    one_hot_encoding=metadata.one_hot_encoding,
    add_reverse_edges=metadata.add_reverse_edges,
)
homo = data.to_homogeneous()
dgl_graph = to_dgl(homo)

node_labels = node_label_dict["EVENT"] + node_label_dict["item"] + node_label_dict["PO"]
feat_labels = feat_label_dict["EVENT"] + feat_label_dict["item"] + feat_label_dict["PO"]

# %% Explanation
config = {
    "objective": "pns",  # pns, pn, ps
    "type_explanation": "ef",
    "lr": 0.01,
    "num_epochs": 500,
    "alpha_e": 0.0005,
    "beta_e": 1,
    "alpha_f": 0,
    "beta_f": 0,
}
device = "cuda" if torch.cuda.is_available() else "cpu"
explainer = NSEG(
    model=model_gcn,
    num_hops=num_layers,
    alpha_e=config["alpha_e"],
    beta_e=config["beta_e"],
    alpha_f=config["alpha_f"],
    beta_f=config["beta_f"],
    num_epochs=config["num_epochs"],
    objective=config["objective"],
    type_ex=config["type_explanation"],
    lr=config["lr"],
    device=device,
)

dgl_graph = dgl_graph.to(device)
features = dgl_graph.ndata["x"].to(device)
# features_cf provides counterfactual features for feature-based explanations (f/ef)
features_cf = torch.zeros_like(features)

if config["type_explanation"] in ["f", "ef"]:
    explanation = explainer.explain_graph(
        "", dgl_graph, features, features_cf=features_cf
    )
else:
    explanation = explainer.explain_graph("", dgl_graph, features)

# %%
mask_e = explanation[0]
mask_f = explanation[2]
{node_labels[i]: round(mask_f[i].item(), 3) for i in range(len(mask_f))}

# %% NSEG for HeteroData
model = torch.load(path_model, weights_only=False)

batch_dict = {
    node_type: torch.zeros(
        data[node_type].num_nodes if data[node_type].num_nodes else 0,
        dtype=torch.long,
        device=device,
    )
    for node_type in metadata.node_types
}
result = generate_counterfactual(
    hetero_data=data,
    model=model,
    objective="necessity",
    alpha_e=config["alpha_e"],
    beta_e=config["beta_e"],
    alpha_f=config["alpha_f"],
    beta_f=config["beta_f"],
    num_epochs=config["num_epochs"],
    lr=config["lr"],
    edge_threshold=0.5,
    feature_threshold=0.5,
    explain_features=True,
    log_every=0,
    run_diagnostics=False,
)
