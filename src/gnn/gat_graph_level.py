import copy

import torch

from numpy import inf
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from typing import Tuple


class GATGraphLevel(nn.Module):
    """GAT model for graph-level predictions on homogeneous graphs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 1,
        heads: int = 8,
        dropout: float = 0.6,
        pool_max: bool = False,
    ):
        super().__init__()

        self.dropout = dropout
        self.pool_max = pool_max

        self.gat_convs = nn.ModuleList()
        current_in = in_channels
        for i in range(num_layers):
            # Concatenate heads on all but the last layer; average on the last.
            is_last = i == num_layers - 1
            self.gat_convs.append(
                GATConv(
                    in_channels=current_in,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=not is_last,
                )
            )
            current_in = hidden_channels if is_last else hidden_channels * heads

        pool_dim = hidden_channels * 2 if self.pool_max else hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, pool_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(pool_dim // 2, out_channels),
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass for graph-level predictions.

        Args:
            x:          Node feature matrix  [num_nodes, in_channels]
            edge_index: Graph connectivity    [2, num_edges]
            batch:      Batch vector          [num_nodes]  — maps each node to its graph index

        Returns:
            Graph-level predictions, shape [num_graphs, out_channels]
        """
        for gat_conv in self.gat_convs:
            x = gat_conv(x, edge_index)
            x = torch.nn.functional.elu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Pool all node embeddings into one vector per graph.
        pool = [global_mean_pool(x, batch)]
        if self.pool_max:
            pool.append(global_max_pool(x, batch))
        graph_emb = torch.cat(pool, dim=-1)

        return self.classifier(graph_emb)


class GATConvTrainerGraphLevel:
    """Trainer for graph-level predictions using a homogeneous GAT architecture."""

    def __init__(
        self,
        model: GATGraphLevel,
        device,
        criterion,
        output_type: str = "binary",
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        grad_clip: float = 1.0,
    ):
        self.model = model
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

    def _forward(self, batch):
        """Run a forward pass on a batch from a homogeneous DataLoader."""
        return self.model(batch.x, batch.edge_index, batch.batch)

    def train_epoch(self, data_loader) -> float:
        self.model.train()

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            out = self._forward(batch)
            loss = self.criterion(out, batch.y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_count += batch.num_graphs
            total_loss += loss.item() * batch.num_graphs

        return total_loss / total_count if total_count > 0 else 0.0

    @torch.no_grad()
    def test(self, data_loader) -> float:
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            out = self._forward(batch)
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
            out = self._forward(batch)
            y_pred.extend(out.argmax(dim=-1).cpu().numpy().tolist())
            y_true.extend(batch.y.view(-1).cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, average="macro") if len(set(y_true)) > 1 else 0.0
        return acc, f1

    @torch.no_grad()
    def evaluate_mae(self, data_loader) -> float:
        """Compute Mean Absolute Error over a data_loader."""
        self.model.eval()
        total_abs_error = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            out = self._forward(batch)
            abs_error = torch.abs(out - batch.y).sum().item()
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

        # Track best weights by val_f1 (the goal metric)
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
        print(f"Final test loss: {test_loss:.4f}")

        if self.output_type == "binary":
            test_acc, test_f1 = self.evaluate_acc(test_loader)
            print("Final test acc:", test_acc, ", test f1:", test_f1)

        if self.output_type == "continuous":
            test_mae = self.evaluate_mae(test_loader)
            print("Final test MAE:", test_mae)
