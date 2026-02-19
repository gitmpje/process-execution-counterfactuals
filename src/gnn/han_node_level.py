import torch

from numpy import inf
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HANConv
from typing import Dict, List, Tuple, Union


class HAN(nn.Module):
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
    ):
        super().__init__()

        self.viewpoint = viewpoint
        self.metadata = metadata

        self.han_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.han_convs.append(
                HANConv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    metadata=self.metadata,
                )
            )
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for han_conv in self.han_convs:
            x_dict = han_conv(x_dict, edge_index_dict)
        out = self.lin(x_dict[self.viewpoint])

        return out


class HANConvTrainer:
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
        self.model.to(self.device)  # Move the model to the GPU (or CPU)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=0.001
        )

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            optimizer.zero_grad()

            out = self.model(batch.x_dict, batch.edge_index_dict)

            y_target = batch.y_dict[self.viewpoint]
            if self.output_type == "binary":
                y_target = y_target.reshape(-1)

            loss = self.criterion(out, y_target)

            loss.backward()
            optimizer.step()

            total_count += batch.batch_size
            total_loss += total_count

        return (total_loss / total_count).item()

    @torch.no_grad()
    def test(self, data_loader):
        self.model.to(self.device)  # Move the model to the GPU (or CPU)
        self.model.eval()

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            out = self.model(batch.x_dict, batch.edge_index_dict)

            y_target = batch.y_dict[self.viewpoint]
            if self.output_type == "binary":
                y_target = y_target.reshape(-1)

            loss = self.criterion(out, y_target)

            total_loss += loss
            total_count += batch.batch_size

        return (total_loss / total_count).item()

    @torch.no_grad()
    def evaluate_acc(self, data_loader) -> float:
        self.model.eval()
        y_true, y_pred = [], []
        for batch in data_loader:
            batch = batch.to(self.device)
            out = self.model(batch.x_dict, batch.edge_index_dict)

            y_pred.extend(out.argmax(dim=-1).cpu().numpy().tolist())
            y_true.extend(batch.y_dict[self.viewpoint].view(-1).cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, average="binary") if len(set(y_true)) > 1 else 0.0

        return acc, f1

    @torch.no_grad()
    def evaluate_mae(self, data_loader):
        """
        Compute Mean Absolute Error (MAE) over a test_loader with heterogeneous data.

        Args:
            data_loader: DataLoader yielding HeteroData or dict-like batches.

        Returns:
            float: Mean Absolute Error over the dataset.
        """
        self.model.eval()
        total_abs_error = 0.0
        total_count = 0
        for batch in data_loader:
            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass
            out = self.model(batch.x_dict, batch.edge_index_dict)

            # Extract target
            target = batch.y_dict[self.viewpoint]
            target = target.to(self.device)

            # Compute absolute error sum for this batch
            abs_error = torch.abs(out - target).sum().item()
            total_abs_error += abs_error
            total_count += batch.batch_size  # total number of predictions

        return total_abs_error / total_count if total_count > 0 else float("nan")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        n_epochs: int = 100,
        start_patience: int = 10,
        learning_rate: float = 0.005,
    ):
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


class HANConvTrainerWithMasks(HANConvTrainer):
    def train_epoch(self, data_loader, learning_rate=0.005):
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=0.001
        )

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            optimizer.zero_grad()
            out = self.model(batch.x_dict, batch.edge_index_dict)

            train_mask = batch[self.viewpoint].train_mask
            if train_mask.sum() > 0:
                y_target = batch.y_dict[self.viewpoint][train_mask]
                if self.output_type == "binary":
                    y_target = y_target.reshape(-1)

                loss = self.criterion(out[train_mask], y_target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * train_mask.sum().item()
                total_count += train_mask.sum().item()

        return total_loss / total_count if total_count > 0 else 0.0

    @torch.no_grad()
    def test(self, data_loader, mask_type="val"):
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            batch = batch.to(self.device)

            out = self.model(batch.x_dict, batch.edge_index_dict)

            mask = getattr(batch[self.viewpoint], f"{mask_type}_mask")
            if mask.sum() > 0:
                y_target = batch.y_dict[self.viewpoint][mask]
                if self.output_type == "binary":
                    y_target = y_target.reshape(-1)

                loss = self.criterion(out[mask], y_target)
                total_loss += loss.item() * mask.sum().item()
                total_count += mask.sum().item()

        return total_loss / total_count if total_count > 0 else 0.0

    @torch.no_grad()
    def evaluate_acc(self, data_loader, mask_type="test"):
        self.model.eval()
        y_true, y_pred = [], []
        for batch in data_loader:
            batch = batch.to(self.device)
            out = self.model(batch.x_dict, batch.edge_index_dict)

            mask = getattr(batch[self.viewpoint], f"{mask_type}_mask")
            if mask.sum() > 0:
                y_pred.extend(out[mask].argmax(dim=-1).cpu().numpy().tolist())
                y_true.extend(
                    batch.y_dict[self.viewpoint][mask].view(-1).cpu().numpy().tolist()
                )

        acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, average="binary") if len(set(y_true)) > 1 else 0.0

        return acc, f1

    @torch.no_grad()
    def evaluate_mae(self, data_loader, mask_type="test"):
        self.model.eval()
        total_abs_error = 0.0
        total_count = 0
        for batch in data_loader:
            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass
            out = self.model(batch.x_dict, batch.edge_index_dict)

            # Extract target
            mask = getattr(batch[self.viewpoint], f"{mask_type}_mask")
            if mask.sum() > 0:
                target = batch.y_dict[self.viewpoint][mask]
                abs_error = torch.abs(out[mask] - target).sum().item()
                total_abs_error += abs_error
                total_count += mask.sum().item()

        return total_abs_error / total_count if total_count > 0 else float("nan")

    def train(
        self,
        data_loader: DataLoader,
        n_epochs: int = 100,
        start_patience: int = 10,
        learning_rate: float = 0.005,
    ):
        min_loss = inf
        patience = start_patience
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(data_loader, learning_rate)
            val_loss = self.test(data_loader, mask_type="val")

            val_acc, val_f1 = None, None
            if self.output_type == "binary":
                val_acc, val_f1 = self.evaluate_acc(data_loader, mask_type="val")

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

        test_loss = self.test(data_loader, mask_type="test")
        print(f"Final test loss: {test_loss:.4f}")
        if self.output_type == "binary":
            test_acc, test_f1 = self.evaluate_acc(data_loader, mask_type="test")
            print("Final test acc:", test_acc, ", test f1:", test_f1)
        elif self.output_type == "continuous":
            test_mae = self.evaluate_mae(data_loader, mask_type="test")
            print("Final test MAE:", test_mae)
