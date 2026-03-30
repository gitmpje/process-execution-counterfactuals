# %% Import dependencies
import json
import os
import torch
import yaml

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from gnn.han_graph_level import HANGraphLevel, HANConvTrainerGraphLevel
from gnn.utils import Metadata

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
path_dataset = dataset_cfg["path_dataset"]
path_metadata = dataset_cfg["path_metadata"]

# GNN
gnn_cfg = cfg["gnn"]
path_model = gnn_cfg["path_model"]
num_layers = gnn_cfg["num_layers"]
dropout = gnn_cfg["dropout"]
n_epochs = gnn_cfg["n_epochs"]
start_patience = gnn_cfg["start_patience"]
learning_rate = gnn_cfg["learning_rate"]
random_seed = gnn_cfg.get("random_seed", 0)

torch.manual_seed(random_seed)

# %% Create train/validate/test dataset for graph-level training
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 10

dataset = torch.load(path_dataset, weights_only=False)

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

labels = [data.y for data in dataset]
class_weights = torch.tensor([len(labels) / (len(labels) - sum(labels)), 1.0])
train_idx, test_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.1,
    random_state=random_seed,
    stratify=labels,
)

train_labels = [labels[i] for i in train_idx]
train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.1,
    random_state=random_seed,
    stratify=train_labels,
)

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
test_ds = Subset(dataset, test_idx)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# %% Define and train model
model = HANGraphLevel(
    in_channels=-1,
    out_channels=2,
    num_layers=num_layers,
    dropout=dropout,
    viewpoint=metadata.viewpoint,
    metadata=[metadata.node_types, metadata.edge_types],
)
trainer = HANConvTrainerGraphLevel(
    model=model,
    viewpoint=metadata.viewpoint,
    device=device,
    criterion=torch.nn.CrossEntropyLoss(),  # weight=class_weights
    output_type="binary",
)
trainer.train(
    train_loader,
    val_loader,
    test_loader,
    n_epochs=n_epochs,
    start_patience=start_patience,
    learning_rate=learning_rate,
)

torch.save(model, path_model)
