# Convert trace_graphs -> PyG Data objects (include node and edge attributes),
# define a GNN that uses node features + aggregated edge features, and train it.
import json
import networkx as nx
import torch

from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HANConv
from typing import Dict, List, Tuple


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
    """
    node_labels = set()
    node_numeric_keys = set()
    node_type_attributes = dict()
    for trace_graph in trace_graphs.values():
        G = trace_graph["process_execution"]
        for _, d in G.nodes(data=True):
            # lab = d.get("label") or d.get("node_label") or d.get("nlabel")
            # if lab is not None:
            #     node_labels.add(str(lab))

            # check attr dict for numeric keys
            attr = d.get("attr") or {}
            if isinstance(attr, dict):
                if attr.get("type") == "OBJECT":
                    node_type = attr.get("ocel:type", "OBJECT")
                else:
                    node_type = "EVENT"
                node_labels.add(node_type)
                node_type_attributes[node_type] = set()
                for k, v in attr.items():
                    if isinstance(v, (int, float)):
                        node_numeric_keys.add(k)
                        node_type_attributes[node_type].add(k)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "node_labels": sorted(node_labels),
                        "node_numeric_keys": sorted(node_numeric_keys),
                    },
                    f,
                )
    return sorted(node_labels), sorted(node_numeric_keys), {k: sorted(v) for k, v in node_type_attributes.items()}


def convert_trace_graphs_to_hetero_pyg(trace_graphs, node_types, node_num_keys):
    data_list = []
    for trace_graph in trace_graphs.values():
        G = trace_graph["process_execution"]

        hetero_data = HeteroData()

        # Collect nodes per type
        node_id_to_type = {}
        type_to_nodes = defaultdict(list)
        for node, attr in G.nodes(data="attr"):
            # label = data.get("label") or data.get("node_label") or data.get("nlabel")
            label = (
                attr.get("ocel:type", "OBJECT")
                if attr.get("type") == "OBJECT"
                else "EVENT"
            )
            if label is not None:
                node_id_to_type[node] = label
                type_to_nodes[label].append(node)

        # Assign local indices and features
        type_to_idx = {}
        for nt in node_types:
            nodes = type_to_nodes.get(nt, [])
            type_to_idx[nt] = {n: i for i, n in enumerate(nodes)}
            feats = []
            for n in nodes:
                attr = G.nodes[n].get("attr") or {}
                feats.append([float(attr.get(k, 0.0)) for k in node_num_keys])
            if feats:
                hetero_data[nt].x = torch.tensor(feats, dtype=torch.float)
            else:
                hetero_data[nt].x = torch.empty(
                    (0, len(node_num_keys)), dtype=torch.float
                )

        # Collect edges
        edge_dict = defaultdict(list)
        for u, v, ed in G.edges(data=True):
            if u not in node_id_to_type or v not in node_id_to_type:
                continue
            u_type = node_id_to_type[u]
            v_type = node_id_to_type[v]
            e_type = ed.get("attr", {}).get("type", "default")
            edge_type = (u_type, e_type, v_type)
            edge_dict[edge_type].append(
                (type_to_idx[u_type][u], type_to_idx[v_type][v])
            )

        for et, edges in edge_dict.items():
            if edges:
                hetero_data[et].edge_index = torch.tensor(edges, dtype=torch.long).t()
            else:
                hetero_data[et].edge_index = torch.empty((2, 0), dtype=torch.long)

        hetero_data.y = torch.tensor(
            [1 if trace_graph.get("class") else 0], dtype=torch.long
        )
        data_list.append(hetero_data)

    return data_list


class HAN(torch.nn.Module):
    def __init__(
        self, in_channels_dict, out_channels, num_classes, metadata, num_layers=1
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HANConv(
                    in_channels_dict,
                    out_channels,
                    heads=8,
                    dropout=0.6,
                    metadata=metadata,
                )
            )
        self.lin = torch.nn.Linear(out_channels, num_classes)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                k: torch.nn.functional.relu(v)
                for k, v in x_dict.items()
                if v is not None
            }
        # Concatenate all node embeddings for graph-level pooling
        x_all = torch.cat(
            [
                x_dict[nt]
                for nt in x_dict.keys()
                if x_dict[nt] is not None and x_dict[nt].shape[0] > 0
            ],
            dim=0,
        )
        if x_all.shape[0] == 0:
            x_pooled = torch.zeros(
                self.lin.in_features,
                device=list(x_dict.values())[0].device
                if x_dict
                else torch.device("cpu"),
            )
        else:
            x_pooled = torch.mean(x_all, dim=0)
        return self.lin(x_pooled.unsqueeze(0))


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = torch.nn.functional.cross_entropy(out, batch.y.view(-1))
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


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
            out = model(batch.x_dict, batch.edge_index_dict)
            p = out.argmax(dim=-1).cpu().numpy().tolist()
            ps.extend(p)
            ys.extend(batch.y.view(-1).cpu().numpy().tolist())

    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(ys, ps) if len(ys) > 0 else 0.0
    f1 = f1_score(ys, ps, average="binary") if len(set(ys)) > 1 else 0.0
    return acc, f1, ys, ps


def build_dataset_from_graph_dict(graphs: Dict[str, nx.Graph]) -> List[Data]:
    # Build dataset
    node_label_vocab, node_num_keys, node_type_attributes = (
        build_vocab_and_numeric_keys(graphs)
    )
    print("node_label_vocab:", node_label_vocab)
    print("node_num_keys:", node_num_keys)
    print("node_type_attributes:", node_type_attributes)

    data_list = convert_trace_graphs_to_hetero_pyg(
        graphs, node_label_vocab, node_num_keys
    )
    if len(data_list) == 0:
        raise RuntimeError("No graphs found in graphs")

    return data_list, node_label_vocab, node_num_keys, node_type_attributes


def train_gnn_graph_classification(
    graphs: Dict[str, nx.Graph],
    model_type: str,
    model_path: str = None,
    model_weights_path: str = None,
):
    """
    Train a GNN for graph classification on process execution graphs.
    Args:
        graphs (Dict[str, nx.Graph]): Dictionary of process execution graphs.
        model_path (str): Path to save the full model.
        model_weights_path (str): Path to save the model weights.
    """

    data_list, node_labels, node_num_keys, _ = build_dataset_from_graph_dict(graphs)

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
    in_channels_dict = {nt: len(node_num_keys) for nt in node_labels}
    all_node_types = set()
    all_edge_types = set()
    for data in data_list:
        all_node_types.update(data.node_types)
        all_edge_types.update(data.edge_types)
    metadata = (sorted(all_node_types), sorted(all_edge_types))
    model = HAN(
        in_channels_dict,
        out_channels=64,
        num_classes=2,
        metadata=metadata,
        num_layers=1,
    ).to(device)

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
