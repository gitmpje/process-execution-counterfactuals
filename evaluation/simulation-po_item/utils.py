import torch
import torch.nn.functional as F
import torch.nn as nn

from networkx import Graph
from torch_geometric.data import Data, HeteroData

from typing import List, Tuple

from gnn.graph_cfe import (
    GraphCFEDatasetStats,
    _to_dense,
)
from gnn.utils import build_hetero_data, Metadata


def _replace_scenario_prefix(item: dict | list | str, scenario_prefix: str):
    if isinstance(item, str):
        return item.replace("$SCENARIO_PREFIX", scenario_prefix)
    if isinstance(item, dict):
        return {
            k: _replace_scenario_prefix(v, scenario_prefix) for k, v in item.items()
        }
    if isinstance(item, list):
        return [_replace_scenario_prefix(v, scenario_prefix) for v in item]
    return item


# Build dense training tensors for GraphCFE and train the VAE on the available dataset.
def _build_graphcfe_training_tensors(dataset, labels, stats, device):
    features = []
    adjs = []
    ys = []
    us = []
    for data, label in zip(dataset, labels, strict=False):
        feat, adj, _ = _to_dense(data, stats.max_num_nodes, stats.x_dim, device)
        features.append(feat)
        adjs.append(adj)
        ys.append(
            torch.tensor([[float(label.item())]], dtype=torch.float32, device=device)
        )

        num_edges = int(data.edge_index.shape[1])
        num_nodes = max(int(data.num_nodes), 1)
        u_value = float(num_edges / num_nodes)
        us.append(torch.tensor([[u_value]], dtype=torch.float32, device=device))

    return (
        torch.cat(features, dim=0),
        torch.cat(adjs, dim=0),
        torch.cat(ys, dim=0),
        torch.cat(us, dim=0),
    )


def train_graphcfe_model(
    graphcfe_model,
    features,
    adjs,
    ys,
    us,
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    device="cpu",
):
    """Train GraphCFE model using CLEAR loss from https://github.com/jma712/GraphCFE."""
    graphcfe_model.to(device)
    optimizer = torch.optim.Adam(graphcfe_model.parameters(), lr=learning_rate)
    num_samples = features.size(0)

    def distance_feature(feat_1, feat_2):
        """Computes mean pairwise L2 distance between feature tensors."""
        pdist = nn.PairwiseDistance(p=2)
        output = pdist(feat_1, feat_2) / 4.0
        return torch.mean(output)

    def distance_graph_prob(adj_1, adj_2_prob):
        """Computes binary cross entropy distance between adjacency matrices."""
        return F.binary_cross_entropy(adj_2_prob, adj_1)

    for epoch in range(1, num_epochs + 1):
        graphcfe_model.train()
        perm = torch.randperm(num_samples, device=device)
        total_loss = 0.0

        for start in range(0, num_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_x = features[batch_idx]
            batch_adj = adjs[batch_idx]
            batch_y = ys[batch_idx]
            batch_u = us[batch_idx]

            outputs = graphcfe_model(batch_x, batch_u, batch_adj, batch_y)
            recon_adj = outputs["adj_reconst"]
            recon_x = outputs["features_reconst"]
            z_mu = outputs["z_mu"]
            z_logvar = outputs["z_logvar"]
            z_u_mu = outputs["z_u_mu"]
            z_u_logvar = outputs["z_u_logvar"]

            # KL loss: divergence between encoder and prior distributions
            loss_kl = 0.5 * (
                (z_u_logvar - z_logvar)
                + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())
                - 1.0
            )
            loss_kl = torch.mean(loss_kl)

            # Similarity loss: weighted distance on features and adjacency
            batch_size_actual = batch_x.size(0)
            dist_x = distance_feature(
                batch_x.view(batch_size_actual, -1),
                recon_x.view(batch_size_actual, -1),
            )
            dist_a = distance_graph_prob(batch_adj, recon_adj)

            beta = 10.0
            loss_sim = beta * dist_x + 10.0 * dist_a

            # Total loss (without CFE loss since no classifier during training)
            loss = loss_sim + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size_actual

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"[GraphCFE] epoch {epoch}/{num_epochs} "
                f"loss={(total_loss / num_samples):.4f} "
                f"loss_sim={loss_sim.item():.4f} "
                f"loss_kl={loss_kl.item():.4f}"
            )

    graphcfe_model.eval()
    return graphcfe_model


def to_homogeneous(
    dataset: List[HeteroData],
) -> Tuple[List[Data], GraphCFEDatasetStats]:
    homogeneous_dataset = []
    labels = []
    for data in dataset:
        homogeneous_dataset.append(data.to_homogeneous())
        labels.append(torch.tensor([data.y], dtype=torch.long))

    return homogeneous_dataset, labels


def get_dense_representation(
    process_execution: Graph,
    metadata: Metadata,
    stats: GraphCFEDatasetStats,
    object_type_column: str,
    event_activity: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    hetero_data, _, _, _, _, _ = build_hetero_data(
        graph=process_execution,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=object_type_column,
        event_activity_col=event_activity,
        viewpoint=metadata.viewpoint,
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
        add_reverse_edges=metadata.add_reverse_edges,
    )
    homogeneous_data = hetero_data.to_homogeneous()

    return _to_dense(homogeneous_data, stats.max_num_nodes, stats.x_dim, device)


def visualize_process_execution(
    process_execution: Graph,
    output_file_name: str,
):
    from networkx import nx_agraph
    from process_execution.process_execution import ProcessExecution
    from process_execution.visualization import (
        apply_node_styles_nx,
        apply_edge_styles_nx,
    )

    process_execution = ProcessExecution(process_execution)
    process_execution.construct_node_label()
    process_execution.construct_edge_label()

    apply_node_styles_nx(process_execution)
    apply_edge_styles_nx(process_execution)

    agraph = nx_agraph.to_agraph(process_execution)
    agraph.draw(output_file_name, prog="dot")
