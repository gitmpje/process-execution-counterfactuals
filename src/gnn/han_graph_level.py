import copy

import torch

from numpy import inf
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.nn import HANConv, global_max_pool, global_mean_pool
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
        pool_max: bool = False,
    ):
        super().__init__()

        self.dropout = dropout
        self.viewpoint = viewpoint
        self.metadata = metadata
        self.pool_max = pool_max

        self.han_convs = nn.ModuleList()
        current_in = in_channels
        for _ in range(num_layers):
            self.han_convs.append(
                HANConv(
                    in_channels=current_in,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=self.dropout,
                    metadata=self.metadata,
                )
            )

            current_in = hidden_channels

        pool_dim = hidden_channels * 2 if self.pool_max else hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, pool_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(pool_dim // 2, out_channels),
        )

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

            out_dim = han_conv.out_channels
            device = next(han_conv.parameters()).device
            x_dict_clean = {}
            for node_type, x in x_dict.items():
                if x is None or x.numel() == 0:
                    x_dict_clean[node_type] = torch.zeros((0, out_dim), device=device)
                else:
                    x = torch.nn.functional.elu(x)
                    x = torch.nn.functional.dropout(
                        x, p=self.dropout, training=self.training
                    )
                    x_dict_clean[node_type] = x
            x_dict = x_dict_clean

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

            # Global pool results
            pool = [global_mean_pool(x, batch_dict[node_type], size=batch_size)]
            if self.pool_max:
                pool.append(global_max_pool(x, batch_dict[node_type], size=batch_size))

            pooled.append(torch.cat(pool, dim=-1))

        if pooled:
            # Average over node types
            graph_emb = torch.stack(pooled).mean(dim=0)
        else:
            # If no valid node-type embeddings were produced fall back to a zero vector.
            graph_emb = torch.zeros(batch_size, self.classifier[0].in_features)

        out = self.classifier(graph_emb)

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
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.viewpoint = viewpoint
        self.device = device
        self.criterion = criterion
        self.output_type = output_type
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # LR scheduler: halve LR when val_loss plateaus for 15 epochs.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )

    def train_epoch(self, data_loader):
        self.model.train()

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            loss = self.criterion(out, batch.y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

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
        f1 = f1_score(y_true, y_pred, average="macro") if len(set(y_true)) > 1 else 0.0

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
        n_epochs: int = 200,
        start_patience: int = 30,
    ):
        self.model.to(self.device)

        # Track best weights by val_f1
        best_val_f1 = 0.0
        best_weights = None
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test(val_loader)

            # Step LR scheduler on validation loss
            self.scheduler.step(val_loss)

            val_acc, val_f1 = None, None
            if self.output_type == "binary":
                val_acc, val_f1 = self.evaluate_acc(val_loader)

            if epoch % 5 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                val_f1_str = (
                    f", val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
                    if val_acc is not None
                    else ""
                )
                print(
                    f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, {val_loss:.4f}"
                    f"{val_f1_str}, lr={current_lr:.2e}"
                )

            # Checkpoint on best F1 and reset patience
            if self.output_type == "binary" and val_f1 is not None:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                # For non-binary output fall back to tracking val_loss
                if val_loss < getattr(self, "_min_loss", inf):
                    self._min_loss = val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= start_patience:
                print(
                    f"Stopping training as val_f1 did not improve "
                    f"for {start_patience} epochs (best val_f1={best_val_f1:.4f})"
                )
                break

        # Restore the best checkpoint before final evaluation
        if best_weights is not None:
            print(f"Restoring best model weights (val_f1={best_val_f1:.4f})")
            self.model.load_state_dict(best_weights)

        test_loss = self.test(test_loader)
        print(f"Final test Loss: {test_loss:.4f}")
        if self.output_type == "binary":
            test_acc, test_f1 = self.evaluate_acc(test_loader)
            print("Final test acc:", test_acc, ", test f1:", test_f1)

        if self.output_type == "continuous":
            test_mae = self.evaluate_mae(test_loader)
            print("Final test MAE:", test_mae)
