import numpy as np
import torch

from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from typing import Dict


def construct_graph_dict(
    trace_graph,
    node_num_keys,
    object_type_col,
    event_activity,
    y_key,
    node_type_object="OBJECT",
    node_type_event="EVENT",
):
    graph_dict = {}
    event_object_types = list(node_num_keys[node_type_object].keys()) + list(
        node_num_keys[node_type_event].keys()
    )

    G = trace_graph["process_execution"]

    # Collect nodes per type
    node_id_to_type = {}
    type_to_nodes = defaultdict(list)
    for node, attr in G.nodes(data="attr"):
        if attr.get("type") == node_type_object:
            object_type = attr.get(object_type_col)
            t = object_type if object_type in event_object_types else node_type_object
        if attr.get("type") == node_type_event:
            event_type = attr.get(event_activity)
            t = event_type if event_type in event_object_types else node_type_event

        node_id_to_type[node] = t
        type_to_nodes[t].append(node)

    # Collect nodes and features
    type_to_idx = {}
    for t in type_to_nodes.keys():
        nodes = type_to_nodes.get(t, [])
        type_to_idx[t] = {n: i for i, n in enumerate(nodes)}
        feats = []
        for n in nodes:
            attr = G.nodes[n].get("attr") or {}
            node_type = attr["type"]
            if node_type not in node_num_keys.keys():
                continue

            node_feats = [float(attr.get(k, 0.0)) for k in node_num_keys[node_type][t]]
            if not node_feats:
                node_feats = [float(0.0)]
            feats.append(node_feats)

        graph_dict[t] = feats

    # Collect edges
    edge_dict = defaultdict(list)
    for u, v, ed in G.edges(data=True):
        if u not in node_id_to_type or v not in node_id_to_type:
            continue
        u_type = node_id_to_type[u]
        v_type = node_id_to_type[v]
        e_type = ed.get("attr", {}).get("type", "default")
        edge_type = (u_type, e_type, v_type)
        edge_dict[edge_type].append((type_to_idx[u_type][u], type_to_idx[v_type][v]))

    graph_dict |= edge_dict

    graph_dict["y"] = trace_graph.get(y_key)

    return graph_dict, type_to_nodes.keys(), edge_dict.keys()


def build_hetero_dataset(
    graphs,
    node_num_keys,
    object_type_col,
    event_activity,
    viewpoint: str,
    y_key: str,
    normalize: bool = False,
    path_dataset: str = None,
):
    dataset = []
    node_types_set = set()
    edge_types_set = set()
    for trace_graph in graphs.values():
        graph_dict, node_types, edge_types = construct_graph_dict(
            trace_graph, node_num_keys, object_type_col, event_activity, y_key
        )

        node_types_set.update(node_types)
        edge_types_set.update(edge_types)

        if viewpoint not in node_types:
            raise ValueError(
                f"No nodes of viewpoint type {viewpoint} occur in the graph"
            )
        if len(graph_dict[viewpoint]) > 1:
            raise ValueError(
                f"More than one node of viewpoint type {viewpoint} occurs in the graph"
            )

        hetero_data = HeteroData()
        for t in node_types:
            hetero_data[t].x = torch.tensor(graph_dict[t], dtype=torch.float32).reshape(
                -1, 1
            )

        hetero_data[viewpoint].y = torch.tensor(
            graph_dict["y"], dtype=torch.float32
        ).reshape(-1, 1)

        for t in edge_types:
            hetero_data[t].edge_index = torch.tensor(
                graph_dict[t], dtype=torch.int64
            ).t()

        dataset.append(hetero_data)

    if normalize:
        y_all = []
        for graph in dataset:
            y_all.extend(graph[viewpoint].y)
        y_all = [t.item() for t in y_all]
        y_mean = np.mean(y_all)
        y_std = np.std(y_all)
        print(f"y_mean={y_mean}, y_std={y_std}")

        for graph in dataset:
            graph[viewpoint].y = (graph[viewpoint].y - y_mean) / y_std

    if path_dataset:
        torch.save(dataset, path_dataset)

    return dataset, node_types_set, edge_types_set


def construct_graph_dict_multiple_viewpoint_nodes(
    G,
    node_num_keys,
    object_type_col,
    event_activity,
    viewpoint,
    node_y_mapping,
    node_type_object="OBJECT",
    node_type_event="EVENT",
):
    graph_dict = {}
    event_object_types = list(node_num_keys[node_type_object].keys()) + list(
        node_num_keys[node_type_event].keys()
    )

    # Collect nodes per type
    node_id_to_type = {}
    type_to_nodes = defaultdict(list)
    for node, attr in G.nodes(data="attr"):
        t = None
        if attr.get("type") == node_type_object:
            object_type = attr.get(object_type_col)
            t = object_type if object_type in event_object_types else node_type_object
        if attr.get("type") == node_type_event:
            event_type = attr.get(event_activity)
            t = event_type if event_type in event_object_types else node_type_event

        if t:
            node_id_to_type[node] = t
            type_to_nodes[t].append(node)

    # Collect nodes and features
    graph_dict["y"] = []
    type_to_idx = {}
    for t in type_to_nodes.keys():
        nodes = type_to_nodes.get(t, [])
        type_to_idx[t] = {n: i for i, n in enumerate(nodes)}
        feats = []
        for n in nodes:
            attr = G.nodes[n].get("attr") or {}
            node_type = attr["type"]
            if node_type not in node_num_keys.keys():
                continue

            node_feats = [float(attr.get(k, 0.0)) for k in node_num_keys[node_type][t]]
            if not node_feats:
                node_feats = [float(0.0)]
            feats.append(node_feats)

            if t == viewpoint:
                if not np.isnan(node_y_mapping.get(n, np.nan)):
                    print(n)
                graph_dict["y"].append(node_y_mapping.get(n, np.nan))

        graph_dict[t] = feats

    # Collect edges
    edge_dict = defaultdict(list)
    for u, v, ed in G.edges(data=True):
        if u not in node_id_to_type or v not in node_id_to_type:
            continue
        u_type = node_id_to_type[u]
        v_type = node_id_to_type[v]
        e_type = ed.get("attr", {}).get("type", "default")
        edge_type = (u_type, e_type, v_type)
        edge_dict[edge_type].append((type_to_idx[u_type][u], type_to_idx[v_type][v]))

    graph_dict |= edge_dict

    return graph_dict, type_to_nodes.keys(), edge_dict.keys()


def build_hetero_data(
    graph,
    node_num_keys,
    object_type_col,
    event_activity,
    viewpoint: str,
    node_y_mapping: Dict[str, float],
    normalize: bool = False,
    path_dataset: str = None,
):
    hetero_data = HeteroData()
    graph_dict, node_types, edge_types = construct_graph_dict_multiple_viewpoint_nodes(
        graph,
        node_num_keys,
        object_type_col,
        event_activity,
        viewpoint,
        node_y_mapping,
    )

    if viewpoint not in node_types:
        raise ValueError(f"No nodes of viewpoint type {viewpoint} occur in the graph")

    for t in node_types:
        hetero_data[t].x = torch.tensor(graph_dict[t], dtype=torch.float32).reshape(
            -1, 1
        )

    hetero_data[viewpoint].y = torch.tensor(
        graph_dict["y"], dtype=torch.float32
    ).reshape(-1, 1)

    for t in edge_types:
        hetero_data[t].edge_index = torch.tensor(graph_dict[t], dtype=torch.int64).t()

    if normalize:
        y_all = [t.item() for t in hetero_data[viewpoint].y if not np.isnan(t.item())]
        y_mean = np.mean(y_all)
        y_std = np.std(y_all)
        print(f"y_mean={y_mean}, y_std={y_std}")
        hetero_data[viewpoint].y = (hetero_data[viewpoint].y - y_mean) / y_std

    if path_dataset:
        torch.save(hetero_data, path_dataset)

    return hetero_data, set(node_types), set(edge_types)


def add_train_val_test_masks(
    dataset, viewpoint, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42
):
    """
    Add train/validation/test masks for the nodes of the specified viewpoint type in a heterogeneous dataset.

    Args:
        dataset: List of HeteroData objects.
        viewpoint: The node type (str) for which to add masks.
        train_ratio: Proportion of nodes for training.
        val_ratio: Proportion of nodes for validation.
        test_ratio: Proportion of nodes for testing.
        random_state: Random state for splitting.
    """
    # Collect all viewpoint node positions as (graph_idx, node_idx)
    all_positions = []
    for graph_idx, data in enumerate(dataset):
        num_nodes = data[viewpoint].x.size(0)
        for node_idx in range(num_nodes):
            # Only include nodes for which y is not NaN
            if not np.isnan(data[viewpoint].y[node_idx]):
                all_positions.append((graph_idx, node_idx))

    # Split the positions
    train_val, test = train_test_split(
        all_positions, test_size=test_ratio, random_state=random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=random_state,
    )

    # Set masks for each graph
    for graph_idx, data in enumerate(dataset):
        num_nodes = data[viewpoint].x.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        for pos in train:
            if pos[0] == graph_idx:
                train_mask[pos[1]] = True
        for pos in val:
            if pos[0] == graph_idx:
                val_mask[pos[1]] = True
        for pos in test:
            if pos[0] == graph_idx:
                test_mask[pos[1]] = True

        data[viewpoint].train_mask = train_mask
        data[viewpoint].val_mask = val_mask
        data[viewpoint].test_mask = test_mask
