import numpy as np
import torch

from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Set


def construct_graph_dict(
    trace_dict: dict,
    node_num_keys: dict,
    object_type_col: str,
    event_activity_col: str,
    y_key: str,
    node_type_object: str = "OBJECT",
    node_type_event: str = "EVENT",
    activities: List[str] = None,
    add_reverse_edges: bool = False,
):
    graph_dict = {}
    feat_label_dict = {}
    node_label_dict = {}
    event_object_types = list(node_num_keys[node_type_object].keys()) + list(
        node_num_keys[node_type_event].keys()
    )

    G = trace_dict["process_execution"]

    # Collect nodes per type
    node_id_to_type = {}
    type_to_nodes = defaultdict(list)
    for node, attr in G.nodes(data="attr"):
        if attr.get("type") == node_type_object:
            object_type = attr.get(object_type_col)
            t = object_type if object_type in event_object_types else node_type_object
        if attr.get("type") == node_type_event:
            event_type = attr.get(event_activity_col)
            t = event_type if event_type in event_object_types else node_type_event

        node_id_to_type[node] = t
        type_to_nodes[t].append(node)

    # Collect nodes and features
    type_to_idx = {}
    for t in type_to_nodes.keys():
        nodes = type_to_nodes.get(t, [])
        type_to_idx[t] = {n: i for i, n in enumerate(nodes)}
        feat_values = []
        node_labels = []
        for n in nodes:
            attr = G.nodes[n].get("attr") or {}
            node_type = attr["type"]
            if node_type not in node_num_keys.keys():
                continue

            node_feats = [float(attr.get(k, 0.0)) for k in node_num_keys[node_type][t]]
            feat_labels = node_num_keys[node_type][t].copy()

            # Add activity (event type) embedding
            if activities and node_type == node_type_event:
                activity_feats = [0] * len(activities)
                activity_feats[activities.index(attr.get(event_activity_col))] = 1
                node_feats.extend(activity_feats)
                feat_labels.extend(activities)

            if not node_feats:
                node_feats = [float(0.0)]

            feat_values.append(node_feats)
            node_labels.append(n)

        graph_dict[t] = feat_values
        feat_label_dict[t] = feat_labels
        node_label_dict[t] = node_labels

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

        # Add reverse edges
        if add_reverse_edges and u_type != v_type:
            rev_edge_type = (v_type, f"rev_{e_type}", u_type)
            edge_dict[rev_edge_type].append(
                (type_to_idx[v_type][v], type_to_idx[u_type][u])
            )

    graph_dict |= edge_dict

    graph_dict["y"] = trace_dict.get(y_key)

    return (
        graph_dict,
        type_to_nodes.keys(),
        edge_dict.keys(),
        feat_label_dict,
        node_label_dict,
    )


def build_hetero_dataset(
    graphs,
    node_num_keys,
    object_type_col,
    event_activity_col,
    viewpoint: str,
    y_key: str,
    activities: List[str] = None,
    add_reverse_edges: bool = False,
    path_dataset: str = None,
    allow_multiple_viewpoint_nodes: bool = False,
) -> List[HeteroData]:
    dataset = []
    node_types_set = set()
    edge_types_set = set()
    for id, trace_dict in graphs.items():
        try:
            graph_dict, node_types, edge_types, feat_label_dict, node_label_dict = (
                construct_graph_dict(
                    trace_dict,
                    node_num_keys,
                    object_type_col,
                    event_activity_col,
                    y_key,
                    activities=activities,
                    add_reverse_edges=add_reverse_edges,
                )
            )

            node_types_set.update(node_types)
            edge_types_set.update(edge_types)

            if viewpoint not in node_types:
                raise ValueError(
                    f"No nodes of viewpoint type {viewpoint} occur in graph {id}"
                )
            if len(graph_dict[viewpoint]) > 1 and not allow_multiple_viewpoint_nodes:
                raise ValueError(
                    f"More than one node of viewpoint type {viewpoint} occurs in graph {id}"
                )

            hetero_data = HeteroData()

            # Add x data
            for t in node_types:
                hetero_data[t].x = torch.tensor(
                    graph_dict[t], dtype=torch.float32
                ).reshape(len(graph_dict[t]), -1)

            # Add y data
            hetero_data[viewpoint].y = torch.tensor(graph_dict["y"]).reshape(-1, 1)

            # Also store viewpoint y on dataset level
            hetero_data.y = graph_dict["y"]

            # Add edge indices
            for t in edge_types:
                hetero_data[t].edge_index = (
                    torch.tensor(graph_dict[t], dtype=torch.int64).t().contiguous()
                )

            dataset.append(hetero_data)
        except Exception as e:
            print(f"Failed converting graph {id} to HeteroData")
            raise e

    if path_dataset:
        torch.save(dataset, path_dataset)

    return dataset, node_types_set, edge_types_set, feat_label_dict, node_label_dict


def construct_graph_dict_multiple_viewpoint_nodes(
    graph,
    node_num_keys,
    object_type_col,
    event_activity_col,
    viewpoint,
    node_y_mapping: Dict[str, int | float] = None,
    node_type_object="OBJECT",
    node_type_event="EVENT",
    activities: List[str] = None,
    add_reverse_edges: bool = False,
):
    graph_dict = {}
    feat_label_dict = {}
    node_label_dict = {}
    event_object_types = list(node_num_keys[node_type_object].keys()) + list(
        node_num_keys[node_type_event].keys()
    )

    # Collect nodes per type
    node_id_to_type = {}
    type_to_nodes = defaultdict(list)
    for node, attr in graph.nodes(data="attr"):
        t = None
        if attr.get("type") == node_type_object:
            object_type = attr.get(object_type_col)
            t = object_type if object_type in event_object_types else node_type_object
        if attr.get("type") == node_type_event:
            event_type = attr.get(event_activity_col)
            t = event_type if event_type in event_object_types else node_type_event

        if t:
            node_id_to_type[node] = t
            type_to_nodes[t].append(node)

    # Collect nodes and features
    graph_dict["y"] = []
    graph_dict["y_nodes"] = []
    type_to_idx = {}
    for t in type_to_nodes.keys():
        nodes = type_to_nodes.get(t, [])
        type_to_idx[t] = {n: i for i, n in enumerate(nodes)}
        feat_values = []
        node_labels = []
        for n in nodes:
            attr = graph.nodes[n].get("attr") or {}
            node_type = attr["type"]
            if node_type not in node_num_keys.keys():
                continue

            node_feats = [float(attr.get(k, 0.0)) for k in node_num_keys[node_type][t]]
            if not node_feats:
                node_feats = [float(0.0)]
            feat_values.append(node_feats)
            feat_labels = node_num_keys[node_type][t].copy()
            node_labels.append(n)

            # Add activity (event type) embedding
            if activities and node_type == node_type_event:
                activity_feats = [0] * len(activities)
                activity_feats[activities.index(attr.get(event_activity_col))] = 1
                node_feats.extend(activity_feats)
                feat_labels.extend(activities)

            if t == viewpoint:
                graph_dict["y_nodes"].append(n)
                if node_y_mapping:
                    y = node_y_mapping.get(n)
                    if y is not None:
                        graph_dict["y"].append(y)
                    else:
                        print(f"No y value found for node {n}")
                        graph_dict["y"].append(np.nan)

        graph_dict[t] = feat_values
        feat_label_dict[t] = feat_labels
        node_label_dict[t] = node_labels

    # Collect edges
    edge_dict = defaultdict(list)
    for u, v, ed in graph.edges(data=True):
        if u not in node_id_to_type or v not in node_id_to_type:
            continue
        u_type = node_id_to_type[u]
        v_type = node_id_to_type[v]
        e_type = ed.get("attr", {}).get("type", "default")
        edge_type = (u_type, e_type, v_type)
        edge_dict[edge_type].append((type_to_idx[u_type][u], type_to_idx[v_type][v]))

        # Add reverse edges
        if add_reverse_edges and u_type != v_type:
            rev_edge_type = (v_type, f"rev_{e_type}", u_type)
            edge_dict[rev_edge_type].append(
                (type_to_idx[v_type][v], type_to_idx[u_type][u])
            )

    graph_dict |= edge_dict

    return (
        graph_dict,
        type_to_nodes.keys(),
        edge_dict.keys(),
        feat_label_dict,
        node_label_dict,
    )


def build_hetero_data(
    graph,
    node_num_keys,
    object_type_col,
    event_activity_col,
    viewpoint: str,
    node_y_mapping: Dict[str, float] = None,
    activities: List[str] = None,
    add_reverse_edges: bool = False,
    path_dataset: str = None,
) -> Tuple[HeteroData, Set[str], Set[str]]:
    hetero_data = HeteroData()
    graph_dict, node_types, edge_types, feat_label_dict, node_label_dict = (
        construct_graph_dict_multiple_viewpoint_nodes(
            graph=graph,
            node_num_keys=node_num_keys,
            object_type_col=object_type_col,
            event_activity_col=event_activity_col,
            viewpoint=viewpoint,
            node_y_mapping=node_y_mapping,
            activities=activities,
            add_reverse_edges=add_reverse_edges,
        )
    )

    if viewpoint not in node_types:
        raise ValueError(f"No nodes of viewpoint type {viewpoint} occur in the graph")

    for t in node_types:
        hetero_data[t].x = torch.tensor(graph_dict[t], dtype=torch.float32).reshape(
            len(graph_dict[t]), -1
        )

    hetero_data[viewpoint].y = torch.tensor(graph_dict["y"]).reshape(-1, 1)

    for t in edge_types:
        hetero_data[t].edge_index = (
            torch.tensor(graph_dict[t], dtype=torch.int64).t().contiguous()
        )

    if path_dataset:
        torch.save(hetero_data, path_dataset)

    return (
        hetero_data,
        set(node_types),
        set(edge_types),
        graph_dict["y_nodes"],
        feat_label_dict,
        node_label_dict,
    )


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
