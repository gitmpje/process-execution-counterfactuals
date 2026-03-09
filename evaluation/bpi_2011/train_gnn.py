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

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# GNN
num_layers = 1
dropout = 0.1

# Training
n_epochs = 100
start_patience = 25
learning_rate = 0.005

# Dataset
dataset_cfg = cfg.get("dataset", {})
path_dataset = dataset_cfg.get("path_dataset")
path_metadata = dataset_cfg.get("path_metadata")

# %% Create train/validate/test dataset for graph-level training
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 10

dataset = torch.load(path_dataset, weights_only=False)

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

train_idx, test_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.1,
    random_state=1,
)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=1)

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
    criterion=torch.nn.CrossEntropyLoss(),
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
