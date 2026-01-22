import torch

from numpy import inf
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.nn import HANConv
from typing import Dict, List, Tuple, Union


class HANConvTrainer:
    def __init__(
        self,
        model,
        viewpoint,
        device,
        criterion,
        output_type: str = "binary",
        start_patience: int = 100,
    ):
        self.model = model
        self.viewpoint = viewpoint
        self.device = device
        self.criterion = criterion
        self.output_type = output_type
        self.start_patience = start_patience

    def train_epoch(self, data_loader, learning_rate=0.005):
        self.model.to(self.device)  # Move the model to the GPU (or CPU)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=0.001
        )

        to_div = 0
        train_loss = 0
        for batch in data_loader:
            # Move the batch to the appropriate device
            batch = batch.to(self.device)

            optimizer.zero_grad()
            out = self.model(batch.x_dict, batch.edge_index_dict)
            loss = self.criterion(out, batch.y_dict[self.viewpoint])

            loss.backward()
            optimizer.step()

            to_div += batch.batch_size
            train_loss += loss

        return (train_loss / to_div).item()

    @torch.no_grad()
    def test(self, data_loader):
        self.model.to(self.device)  # Move the model to the GPU (or CPU)

        to_div = 0
        test_loss = 0
        self.model.eval()
        for batch in data_loader:
            batch = batch.to(self.device)

            out = self.model(batch.x_dict, batch.edge_index_dict)
            loss = self.criterion(out, batch.y_dict[self.viewpoint])

            test_loss += loss
            to_div += batch.batch_size

        return (test_loss / to_div).item()

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

        # Avoid division by zero
        if total_count == 0:
            return float("nan")

        return total_abs_error / total_count

    def train(self, train_loader, val_loader, test_loader):
        min_loss = inf
        patience = self.start_patience
        for epoch in range(1, 200):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test(val_loader)

            val_acc, val_f1 = None, None
            if self.output_type == "binary":
                val_acc, val_f1 = self.evaluate_acc(val_loader)

            if epoch % 5 == 0 or epoch == 1:
                val_f1_str = (
                    f", val_acc={val_acc:.4f}, val_f1={val_f1:.4f}" if val_acc else ""
                )
                print(
                    f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, {val_loss:.4f}{val_f1_str}"
                )

            if val_loss < min_loss:
                min_loss = val_loss
                patience = self.start_patience
            else:
                patience -= 1

            if patience <= 0:
                print(
                    "Stopping training as validation Loss did not improve "
                    f"for {self.start_patience} epochs"
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
    ):
        super().__init__()

        self.viewpoint = viewpoint
        self.metadata = metadata

        self.han_convs = []
        for _ in range(num_layers):
            self.han_convs.append(
                HANConv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    dropout=0.6,
                    metadata=self.metadata,
                )
            )
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for han_conv in self.han_convs:
            x_dict = han_conv(x_dict, edge_index_dict)
        out = self.lin(x_dict[self.viewpoint])

        return out


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
                loss = self.criterion(
                    out[train_mask], batch.y_dict[self.viewpoint][train_mask]
                )
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
                loss = self.criterion(out[mask], batch.y_dict[self.viewpoint][mask])
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
            batch = batch.to(self.device)

            out = self.model(batch.x_dict, batch.edge_index_dict)

            mask = getattr(batch[self.viewpoint], f"{mask_type}_mask")
            if mask.sum() > 0:
                target = batch.y_dict[self.viewpoint][mask]
                abs_error = torch.abs(out[mask] - target).sum().item()
                total_abs_error += abs_error
                total_count += mask.sum().item()

        return total_abs_error / total_count if total_count > 0 else float("nan")

    def train(self, data_loader):
        min_loss = inf
        patience = self.start_patience
        for epoch in range(1, 200):
            train_loss = self.train_epoch(data_loader)
            val_loss = self.test(data_loader, mask_type="val")

            val_acc, val_f1 = None, None
            if self.output_type == "binary":
                val_acc, val_f1 = self.evaluate_acc(data_loader, mask_type="val")

            if epoch % 5 == 0 or epoch == 1:
                val_f1_str = (
                    f", val_acc={val_acc:.4f}, val_f1={val_f1:.4f}" if val_acc else ""
                )
                print(
                    f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, {val_loss:.4f}{val_f1_str}"
                )

            if val_loss < min_loss:
                min_loss = val_loss
                patience = self.start_patience
            else:
                patience -= 1

            if patience <= 0:
                print(
                    "Stopping training as validation Loss did not improve "
                    f"for {self.start_patience} epochs"
                )
                break
        test_loss = self.test(data_loader, mask_type="test")
        print(f"Final test Loss: {test_loss:.4f}")
        if self.output_type == "binary":
            test_acc, test_f1 = self.evaluate_acc(data_loader, mask_type="test")
            print("Final test acc:", test_acc, ", test f1:", test_f1)

        if self.output_type == "continuous":
            test_mae = self.evaluate_mae(data_loader, mask_type="test")
            print("Final test MAE:", test_mae)
