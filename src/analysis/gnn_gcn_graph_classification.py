# Convert trace_graphs -> PyG Data objects (include node and edge attributes),
# define a GNN that uses node features + aggregated edge features, and train it.
import json
import numpy as np
import networkx as nx
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Dict, List, Optional, Tuple


def build_vocab_and_numeric_keys(
    trace_graphs: Dict[str, nx.Graph],
    output_path: str = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Build node label vocabulary and identify numeric attribute keys for nodes and edges.

    Args:
        trace_graphs (Dict[str, nx.Graph]): Dictionary mapping event IDs to their corresponding process execution graphs.
            {"process_execution": nx.Graph, "class": bool}

    Returns:
        Tuple[List[str], List[str], List[str]]:
            - List of unique node labels.
            - List of numeric attribute keys for nodes.
            - List of numeric attribute keys for edges.
    """
    node_labels = set()
    node_numeric_keys = set()
    edge_numeric_keys = set()
    for trace_graph in trace_graphs.values():
        G = trace_graph["process_execution"]
        for _, d in G.nodes(data=True):
            lab = d.get("label") or d.get("node_label") or d.get("nlabel")
            if lab is not None:
                node_labels.add(str(lab))
            # check attr dict for numeric keys
            attr = d.get("attr") or {}
            if isinstance(attr, dict):
                for k, v in attr.items():
                    if isinstance(v, (int, float)):
                        node_numeric_keys.add(k)
        for _, _, ed in G.edges(data=True):
            eattr = ed.get("attr") or {}
            if isinstance(eattr, dict):
                for k, v in eattr.items():
                    if isinstance(v, (int, float)):
                        edge_numeric_keys.add(k)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "node_labels": sorted(node_labels),
                        "node_numeric_keys": sorted(node_numeric_keys),
                        "edge_numeric_keys": sorted(edge_numeric_keys),
                    },
                    f,
                )
    return sorted(node_labels), sorted(node_numeric_keys), sorted(edge_numeric_keys)


def convert_trace_graphs_to_pyg(
    trace_graphs, node_label_vocab, node_num_keys, edge_num_keys
) -> List[Data]:
    """
    Convert trace graphs to PyTorch Geometric Data objects.

    Args:
        trace_graphs (Dict[str, nx.Graph]): Dictionary mapping event IDs to their corresponding process execution graphs.
            {"process_execution": nx.Graph, "class": bool}
        node_label_vocab (List[str]): List of node labels for one-hot encoding.
        node_num_keys (List[str]): List of numeric attribute keys for nodes.
        edge_num_keys (List[str]): List of numeric attribute keys for edges.

    Returns:
        List[Data]: List of PyTorch Geometric Data objects.
    """

    label_to_idx = {label: i for i, label in enumerate(node_label_vocab)}

    data_list = []
    for trace_graph in trace_graphs.values():
        G = trace_graph["process_execution"]
        # node features: one-hot label + numeric attrs
        node_list = list(G.nodes())
        node_feat = []
        for n in node_list:
            d = G.nodes[n]
            lab = d.get("label") or d.get("node_label") or d.get("nlabel")
            if lab is not None and str(lab) in label_to_idx:
                onehot = np.zeros(len(node_label_vocab), dtype=np.float32)
                onehot[label_to_idx[str(lab)]] = 1.0
            else:
                onehot = np.zeros(len(node_label_vocab), dtype=np.float32)
            # numeric attrs
            nums = []
            attr = d.get("attr") or {}
            for k in node_num_keys:
                nums.append(float(attr.get(k, 0.0)))
            vec = (
                np.concatenate([onehot, np.array(nums, dtype=np.float32)])
                if len(nums) > 0
                else onehot
            )
            node_feat.append(vec)

        if len(node_feat) == 0:
            x = torch.zeros((0, max(1, len(node_label_vocab))), dtype=torch.float)
        else:
            x = torch.tensor(np.vstack(node_feat), dtype=torch.float)

        # edges
        edge_index_src = []
        edge_index_dst = []
        edge_attr_list = []
        for u, v, ed in G.edges(data=True):
            edge_index_src.append(node_list.index(u))
            edge_index_dst.append(node_list.index(v))
            eattr = ed.get("attr") or {}
            vals = [float(eattr.get(k, 0.0)) for k in edge_num_keys]
            edge_attr_list.append(vals)

        if len(edge_index_src) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None
        else:
            edge_index = torch.tensor(
                [edge_index_src, edge_index_dst], dtype=torch.long
            )
            edge_attr = (
                torch.tensor(np.vstack(edge_attr_list), dtype=torch.float)
                if len(edge_num_keys) > 0
                else None
            )

        # aggregate edge attributes per node (sum of incident edge attrs)
        if edge_attr is not None:
            agg = np.zeros((len(node_list), edge_attr.shape[1]), dtype=np.float32)
            for ei in range(edge_index.shape[1]):
                src = int(edge_index[0, ei].item())
                dst = int(edge_index[1, ei].item())
                agg[src] += edge_attr[ei].numpy()
                agg[dst] += edge_attr[ei].numpy()
            agg = torch.tensor(agg, dtype=torch.float)
            # concat to node features (pad if necessary)
            if x.shape[0] == 0:
                x = agg
            else:
                x = torch.cat([x, agg], dim=1)

        data = Data(x=x, edge_index=edge_index)
        data.y = torch.tensor([1 if trace_graph.get("class") else 0], dtype=torch.long)
        data_list.append(data)
    return data_list


class GCNWithEdgeAgg(torch.nn.Module):
    """
    A simple Graph Convolutional Network (GCN) for graph classification that uses node features
    augmented with aggregated edge attributes.

    Arguments:
        in_channels (int): Number of input features per node.
        hidden (int): Number of hidden units in the GCN layers. Default is 64.
        num_classes (int): Number of output classes for classification. Default is 2.
    """

    def __init__(self, in_channels, hidden=64, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        """
        Forward pass of the GCN.
        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge index tensor.
            batch (torch.Tensor): Batch vector, which assigns each node to a specific graph in the batch.
        Returns:
            torch.Tensor: Output logits for each graph in the batch.
        """
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)


def train_epoch(model, loader, opt, device) -> float:
    """
    Train the GCN model for one epoch.
    Args:
        model (torch.nn.Module): The GCN model to train.
        loader (DataLoader): DataLoader for the training data.
        opt (torch.optim.Optimizer): Optimizer for training.
        device (str): Device to run the training on ('cpu' or 'cuda').
    Returns:
        float: Average loss over the epoch.
    """
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = torch.nn.functional.cross_entropy(out, batch.y.view(-1))
        loss.backward()
        opt.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def evaluate_acc(model, loader, device) -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluate the GCN model on the given data loader.
    Args:
        model (torch.nn.Module): The GCN model to evaluate.
        loader (DataLoader): DataLoader for the evaluation data.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').
    Returns:
        Tuple[float, float, List[int], List[int]]: Accuracy, F1 score, true labels, predicted labels.
    """
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            p = out.argmax(dim=-1).cpu().numpy().tolist()
            ps.extend(p)
            ys.extend(batch.y.view(-1).cpu().numpy().tolist())

    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(ys, ps) if len(ys) > 0 else 0.0
    f1 = f1_score(ys, ps, average="binary") if len(set(ys)) > 1 else 0.0
    return acc, f1, ys, ps


def build_dataset_from_graph_dict(graphs: Dict[str, nx.Graph]) -> List[Data]:
    # Build dataset
    node_label_vocab, node_num_keys, edge_num_keys = build_vocab_and_numeric_keys(
        graphs
    )
    print("node_label_vocab:", node_label_vocab)
    print("node_num_keys:", node_num_keys)
    print("edge_num_keys:", edge_num_keys)

    data_list = convert_trace_graphs_to_pyg(
        graphs, node_label_vocab, node_num_keys, edge_num_keys
    )
    if len(data_list) == 0:
        raise RuntimeError("No graphs found in graphs")

    return data_list


def train_gnn_graph_classification(
    graphs: Dict[str, nx.Graph], model_path: str = None, model_weights_path: str = None
):
    """
    Train a GNN for graph classification on process execution graphs.
    Args:
        graphs (Dict[str, nx.Graph]): Dictionary of process execution graphs.
        model_path (str): Path to save the full model.
        model_weights_path (str): Path to save the model weights.
    """

    data_list = build_dataset_from_graph_dict(graphs)

    labels = [int(d.y.item()) for d in data_list]
    train_idx, test_idx = train_test_split(
        list(range(len(data_list))),
        test_size=0.2,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    train_ds = Subset(data_list, train_idx)
    val_ds = Subset(data_list, val_idx)
    test_ds = Subset(data_list, test_idx)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    test_loader = DataLoader(test_ds, batch_size=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = data_list[0].x.shape[1]
    model = GCNWithEdgeAgg(in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 31):
        loss = train_epoch(model, train_loader, opt, device)
        val_acc, val_f1, val_y, val_p = evaluate_acc(model, val_loader, device)
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d} loss={loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            )

    test_acc, test_f1, test_y, test_p = evaluate_acc(model, test_loader, device)
    print("Final test acc:", test_acc, "test f1:", test_f1)

    # Save model (weights)
    if model_path:
        torch.save(model, model_path)
    if model_weights_path:
        torch.save(model.state_dict(), model_weights_path)


def explain_gnn_graph_classification(
    graph: nx.Graph,
    node_label_vocab: List[str],
    node_num_keys: List[str],
    edge_num_keys: List[str],
    output_path_feature_importance: Optional[str],
    output_path_subgraph: Optional[str],
):
    feat_labels = node_label_vocab + node_num_keys

    graphs = {"tmp": {"process_execution": graph}}
    data_list = convert_trace_graphs_to_pyg(
        graphs,
        node_label_vocab,
        node_num_keys,
        edge_num_keys,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data_list[0]
    in_ch = data.x.shape[1]
    model = GCNWithEdgeAgg(in_ch).to(device)

    batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    data = data.to(device)

    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )
    explanation = explainer(data.x, data.edge_index, batch=batch_vec)

    if output_path_feature_importance:
        explanation.visualize_feature_importance(
            output_path_feature_importance,
            feat_labels=feat_labels,
            top_k=10,
        )

    if output_path_subgraph:
        explanation.visualize_graph(
            output_path_subgraph, node_labels=[v for n, v in graph.nodes(data="label")]
        )

    return explainer, bool(
        explainer.get_prediction(data.x, data.edge_index, batch=batch_vec)
        .argmax()
        .item()
    )
