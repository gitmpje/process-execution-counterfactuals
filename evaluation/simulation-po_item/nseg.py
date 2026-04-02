# %%
import os
import torch
import yaml

from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_dgl

from NSEG.explainer.explainer_NSEG import NSEG
from NSEG.GCN.model import GCN

from utils import _replace_scenario_prefix

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Replace $SCENARIO_PREFIX tokens in config
SCENARIO_PREFIX = os.environ.get("SCENARIO_PREFIX", "scenario_01")
if SCENARIO_PREFIX is not None:
    cfg = _replace_scenario_prefix(cfg, SCENARIO_PREFIX)

# Dataset
dataset_cfg = cfg["dataset"]
path_dataset = dataset_cfg["path_dataset"]

# GNN
gnn_cfg = cfg["gnn"]
path_model = gnn_cfg["path_model"]
num_layers = gnn_cfg["num_layers"]

# %% Load dataset
dataset = torch.load(path_dataset, weights_only=False)

dataset_dgl = []
labels = []
for data in dataset:
    dataset_dgl.append(to_dgl(data.to_homogeneous()))
    labels.append(torch.tensor([data.y], dtype=torch.long))

# %% Train model
# graph classification training based on dataset_dgl + labels
num_classes = len(torch.unique(torch.cat(labels)).to(torch.long))
feat_dim = dataset_dgl[0].ndata["x"].shape[1]
hidden_dims = [64, 64]
num_gnn_layers = len(hidden_dims)

model = GCN(
    dim_input=feat_dim,
    dim_hidden=hidden_dims,
    num_classes=num_classes,
    dropout=0.5,
    num_layers=num_gnn_layers,
    mode="graph",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# split indices
all_idx = list(range(len(dataset_dgl)))
all_labels = [int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in labels]
train_idx, test_idx = train_test_split(
    all_idx, test_size=0.1, random_state=42, stratify=all_labels
)
train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.1,
    random_state=42,
    stratify=[all_labels[i] for i in train_idx],
)

train_graphs = [dataset_dgl[i] for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]
val_graphs = [dataset_dgl[i] for i in val_idx]
val_labels = [all_labels[i] for i in val_idx]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

n_epochs = 50
for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss = 0.0
    for g, y in zip(train_graphs, train_labels):
        g = g.to(device)
        feat = g.ndata["x"].to(device)
        out = model(g, feat).squeeze(0)
        loss = model.loss(
            out.unsqueeze(0), torch.tensor([y], dtype=torch.long, device=device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        correct_val = 0
        for g, y in zip(val_graphs, val_labels):
            g = g.to(device)
            feat = g.ndata["x"].to(device)
            out = model(g, feat).squeeze(0)
            pred = out.argmax(dim=-1).item()
            correct_val += int(pred == y)
        val_acc = correct_val / len(val_graphs)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | train_loss {total_loss / len(train_graphs):.4f} | val_acc {val_acc:.4f}"
        )

# save trained model
try:
    torch.save(model.state_dict(), path_model)
except Exception:
    print("could not save model at", path_model)

# %% Explain
config = {
    "objective": "pns",
    "type_explanation": "f",
    "lr": 0.01,
    "num_epochs": 500,
    "alpha_e": 0.0005,
    "beta_e": 1,
    "alpha_f": 0,
    "beta_f": 0,
}
device = "cuda" if torch.cuda.is_available() else "cpu"
explainer = NSEG(
    model=model,
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

graph_idx = 0
graph = dataset_dgl[graph_idx].to(device)
features = graph.ndata["x"].to(device)
# features_cf provides counterfactual features for feature-based explanations (f/ef)
features_cf = torch.zeros_like(features)

if config["type_explanation"] in ["f", "ef"]:
    mask_explanation, ids = explainer.explain_graph(
        graph_idx, graph, features, features_cf=features_cf
    )
else:
    mask_explanation, ids = explainer.explain_graph(graph_idx, graph, features)

# %% Evaluate explanation
