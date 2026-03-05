import torch

from numpy import inf
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.nn import HANConv, global_mean_pool
from typing import Dict, List, Tuple, Union


class HANGraphLevel(nn.Module):
    """HAN model for graph-level predictions."""

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        viewpoint: str,
        metadata: Tuple[List[str], List[Tuple[str]]],
        hidden_channels=128,
        num_layers=1,
        heads=8,
        dropout=0.6,
        pooling_method: str = "mean",
    ):
        super().__init__()

        self.viewpoint = viewpoint
        self.metadata = metadata
        self.pooling_method = pooling_method

        self.han_convs = nn.ModuleList()
        current_in = in_channels
        for _ in range(num_layers):
            self.han_convs.append(
                HANConv(
                    in_channels=current_in,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    metadata=self.metadata,
                )
            )

            if isinstance(current_in, dict):
                # Map each node type to hidden_channels for subsequent layers
                current_in = {nt: hidden_channels for nt in current_in.keys()}
            else:
                current_in = hidden_channels
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        """
        Forward pass for graph-level predictions.

        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            batch_dict: Dictionary of batch vectors indicating which graph each node belongs to

        Returns:
            Graph-level predictions (one per graph in batch)
        """
        for han_conv in self.han_convs:
            x_dict = han_conv(x_dict, edge_index_dict)

            # HANConv may return `None` for node types with no nodes or edges.
            # Substitute an empty tensor matching the conv's output dimension.
            out_dim = han_conv.out_channels
            device = next(han_conv.parameters()).device
            for nt, v in list(x_dict.items()):
                if v is None:
                    x_dict[nt] = torch.zeros((0, out_dim), device=device)

        # Determine the expected batch size (number of graphs) across all node types.
        batch_size = 0
        for b in batch_dict.values():
            if b.numel() > 0:
                batch_size = max(batch_size, int(b.max().item() + 1))

        # Pool over all node types
        pooled = []
        for node_type, x in x_dict.items():
            # HANConv may return `None` for a type with no nodes or no incoming messages.
            # also skip tensors with zero elements (no nodes of this type)
            if x is None or x.numel() == 0:
                continue

            # Global pool result should contain one row per graph even
            # if some graphs have no nodes of this type.
            pooled.append(global_mean_pool(x, batch_dict[node_type], size=batch_size))

        if pooled:
            # Average over node types
            graph_emb = torch.stack(pooled).mean(dim=0)
        else:
            # If no valid node-type embeddings were produced fall back to a zero vector.
            graph_emb = torch.zeros(batch_size, self.lin.in_features)

        out = self.lin(graph_emb)

        return out


class HANConvTrainerGraphLevel:
    """Trainer for graph-level predictions using HAN architecture."""

    def __init__(
        self,
        model,
        viewpoint,
        device,
        criterion,
        output_type: str = "binary",
    ):
        self.model = model
        self.viewpoint = viewpoint
        self.device = device
        self.criterion = criterion
        self.output_type = output_type

    def train_epoch(self, data_loader, learning_rate=0.005):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=0.001
        )

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            optimizer.zero_grad()
            out = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            loss = self.criterion(out, batch.y)

            loss.backward()
            optimizer.step()

            total_count += batch.num_graphs
            total_loss += loss.item() * batch.num_graphs

        return total_loss / total_count if total_count > 0 else 0.0

    @torch.no_grad()
    def test(self, data_loader):
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            out = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            loss = self.criterion(out, batch.y)

            total_loss += loss.item() * batch.num_graphs
            total_count += batch.num_graphs

        return total_loss / total_count if total_count > 0 else 0.0

    @torch.no_grad()
    def evaluate_acc(self, data_loader) -> Tuple[float, float]:
        """Evaluate accuracy and F1 score on batched graph-level data."""
        self.model.eval()
        y_true, y_pred = [], []
        for batch in data_loader:
            batch = batch.to(self.device)
            out = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            y_pred.extend(out.argmax(dim=-1).cpu().numpy().tolist())
            y_true.extend(batch.y.view(-1).cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, average="binary") if len(set(y_true)) > 1 else 0.0

        return acc, f1

    @torch.no_grad()
    def evaluate_mae(self, data_loader) -> float:
        """
        Compute Mean Absolute Error (MAE) over a data_loader with heterogeneous graph data.

        Args:
            data_loader: DataLoader yielding batched heterogeneous graphs.

        Returns:
            float: Mean Absolute Error over the dataset.
        """
        self.model.eval()
        total_abs_error = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            # Forward pass
            out = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)

            # Extract target
            target = batch.y

            # Compute absolute error sum for this batch
            abs_error = torch.abs(out - target).sum().item()
            total_abs_error += abs_error
            total_count += batch.num_graphs

        return total_abs_error / total_count if total_count > 0 else float("nan")

    def train(
        self,
        train_loader,
        val_loader,
        test_loader,
        n_epochs: int = 100,
        start_patience: int = 10,
        learning_rate: float = 0.005,
    ):
        self.model.to(self.device)
        self.model.train()

        min_loss = inf
        patience = start_patience
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader, learning_rate)
            val_loss = self.test(val_loader)

            val_acc, val_f1 = None, None
            if self.output_type == "binary":
                val_acc, val_f1 = self.evaluate_acc(val_loader)

            if epoch % 5 == 0:
                val_f1_str = (
                    f", val_acc={val_acc:.4f}, val_f1={val_f1:.4f}" if val_acc else ""
                )
                print(
                    f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, {val_loss:.4f}{val_f1_str}"
                )

            if val_loss < min_loss:
                min_loss = val_loss
                patience = start_patience
            else:
                patience -= 1

            if patience <= 0:
                print(
                    "Stopping training as validation Loss did not improve "
                    f"for {start_patience} epochs"
                )
                break
        test_loss = self.test(test_loader)
        print(f"Final test Loss: {test_loss:.4f}")
        if self.output_type == "binary":
            test_acc, test_f1 = self.evaluate_acc(test_loader)
            print("Final test acc:", test_acc, ", test f1:", test_f1)

        if self.output_type == "continuous":
            test_mae = self.evaluate_mae(test_loader)
            print("Final test MAE:", test_mae)
