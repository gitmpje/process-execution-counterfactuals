# Convert trace_graphs -> PyG Data objects (include node and edge attributes),
# define a GNN that uses node features + aggregated edge features, and train it.
import torch
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception:
    raise ImportError(
        "Install torch_geometric to run this cell: https://pytorch-geometric.readthedocs.io/"
    )


def _unwrap_graph(obj):
    if isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph)):
        return obj
    if hasattr(obj, "g"):
        return getattr(obj, "g")
    if hasattr(obj, "graph"):
        return getattr(obj, "graph")
    # as a last resort, try to use the object itself if it behaves like a graph
    raise TypeError("Unsupported graph object")


def build_vocab_and_numeric_keys(trace_graphs):
    node_labels = set()
    node_numeric_keys = set()
    edge_numeric_keys = set()
    for k, v in trace_graphs.items():
        proc = v.get("process_execution")
        if proc is None:
            continue
        G = _unwrap_graph(proc)
        for _, d in G.nodes(data=True):
            lab = d.get("label") or d.get("node_label") or d.get("nlabel")
            if lab is not None:
                node_labels.add(str(lab))
            # check attr dict for numeric keys
            attr = d.get("attr") or {}
            if isinstance(attr, dict):
                for kk, vv in attr.items():
                    if isinstance(vv, (int, float)):
                        node_numeric_keys.add(kk)
        for _, _, ed in G.edges(data=True):
            eattr = ed.get("attr") or {}
            if isinstance(eattr, dict):
                for kk, vv in eattr.items():
                    if isinstance(vv, (int, float)):
                        edge_numeric_keys.add(kk)
    return sorted(node_labels), sorted(node_numeric_keys), sorted(edge_numeric_keys)

def convert_trace_graphs_to_pyg(trace_graphs, node_label_vocab, node_num_keys, edge_num_keys):
    label_to_idx = {l: i for i, l in enumerate(node_label_vocab)}

    data_list = []
    for gid, v in trace_graphs.items():
        G = _unwrap_graph(v["process_execution"])
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
            for kk in node_num_keys:
                nums.append(float(attr.get(kk, 0.0)))
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
        for u, w, ed in G.edges(data=True):
            edge_index_src.append(node_list.index(u))
            edge_index_dst.append(node_list.index(w))
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
        data.y = torch.tensor([1 if v.get("class") else 0], dtype=torch.long)
        data_list.append(data)
    return data_list


class GCNWithEdgeAgg(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)


def train_epoch(model, loader, opt, device):
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


def evaluate_acc(model, loader, device):
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


import os
import json


def load_graphml_with_json_attrs(path: str) -> nx.Graph:
    """Read a GraphML file and attempt to JSON-decode any string attributes back into Python objects.

    Only replaces attribute values when json.loads returns a dict or list (to avoid converting plain strings).
    Works for Graph/DiGraph and MultiGraph/MultiDiGraph edge representations.
    """
    G = nx.read_graphml(path)

    # Nodes
    for n, d in G.nodes(data=True):
        for k, v in list(d.items()):
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, (dict, list)):
                        d[k] = parsed
                except Exception:
                    # leave as string if it isn't JSON
                    pass

    # Edges (handle keyed MultiGraphs and non-keyed graphs)
    try:
        edges = list(G.edges(keys=True, data=True))
        keyed = True
    except TypeError:
        edges = list(G.edges(data=True))
        keyed = False

    if keyed:
        for u, v, key, ed in edges:
            for k, val in list(ed.items()):
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            ed[k] = parsed
                    except Exception:
                        pass
    else:
        for u, v, ed in edges:
            for k, val in list(ed.items()):
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            ed[k] = parsed
                    except Exception:
                        pass

    return G


if __name__ == "__main__":
    path = "data/scenario_test_ocel.json"

    # Load the graphml and parse JSON attributes back
    graphml_path = path.replace(".json", ".graphml")
    if os.path.exists(graphml_path):
        ocel_nx = load_graphml_with_json_attrs(graphml_path)
        print(
            f"Loaded GraphML from {graphml_path} — nodes={ocel_nx.number_of_nodes()} edges={ocel_nx.number_of_edges()}"
        )
    else:
        print(f"GraphML file not found: {graphml_path}")


    from collections import Counter
    from pm4py import read_ocel2_json

    from process_execution import extract_process_execution


    ocel = read_ocel2_json(path)

    df_events = ocel.events.copy()
    df_events.set_index("ocel:eid", inplace=True)
    df_relations = ocel.relations.copy()
    df_relations.set_index("ocel:eid", inplace=True)

    df_events_objects = df_events.join(df_relations, rsuffix="_relations")
    object_types = ["PackingUnit"]

    events_to_trace = df_events_objects[
        (df_events_objects["ocel:type"].isin(object_types))
    ].index.values

    print(
        f"Number of events selected to extract process execution for: {len(events_to_trace)}"
    )


    def determine_class_quality(event: str):
        return ocel_nx.nodes()[event]["attr"].get("averageQuality") >= 1.0


    def determine_class_attribute(trace_graph: nx.Graph):
        selected_activity = "Object-departing-WB"
        selected_attribute = "a"
        for _, data in trace_graph.nodes(data="attr"):
            if (
                data.get("ocel:activity", "") == selected_activity
                and data.get(selected_attribute, 1) < 0.25
            ):
                return False
        return True


    # Extract process executions
    trace_graphs = {}
    for event in events_to_trace:
        trace_graph = extract_process_execution(
            ocel_nx,
            event,
            ["ProductionLot", "PackingUnit"],
            "Object-creating_class_instance",
        )
        trace_graph.construct_node_label()
        trace_graph.construct_edge_label()

        trace_graphs[event] = {
            "process_execution": trace_graph,
            # "class": determine_class_quality(event),
            "class": determine_class_attribute(trace_graph),
        }
    print(Counter([d["class"] for d in trace_graphs.values()]))


    # Build dataset
    node_label_vocab, node_num_keys, edge_num_keys = build_vocab_and_numeric_keys(
        trace_graphs
    )
    print("node_label_vocab:", node_label_vocab)
    print("node_num_keys:", node_num_keys)
    print("edge_num_keys:", edge_num_keys)

    data_list = convert_trace_graphs_to_pyg(trace_graphs, node_label_vocab, node_num_keys, edge_num_keys)
    if len(data_list) == 0:
        raise RuntimeError("No graphs found in trace_graphs")

    labels = [int(d.y.item()) for d in data_list]
    train_idx, test_idx = train_test_split(
        list(range(len(data_list))),
        test_size=0.2,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    from torch.utils.data import Subset

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

    train_acc, train_f1, train_y, train_p = evaluate_acc(model, train_loader, device)

    test_acc, test_f1, test_y, test_p = evaluate_acc(model, test_loader, device)
    print("Final test acc:", test_acc, "test f1:", test_f1)

    # Save model weights
    torch.save(model.state_dict(), path.replace(".json", "-model_weights.pth"))

    # Save predictions
    train_events = [list(trace_graphs.keys())[i] for i in train_idx]
    val_events = [list(trace_graphs.keys())[i] for i in val_idx]
    test_events = [list(trace_graphs.keys())[i] for i in test_idx]

    from pandas import Series

    Series(
        index=train_events + val_events + test_events, data=train_p + val_p + test_p
    ).to_csv(path.replace(".json", "-gnn_predictions.csv"))
