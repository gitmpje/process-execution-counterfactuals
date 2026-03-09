import numpy as np
import torch

from collections import defaultdict
from networkx import Graph
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Set


def construct_graph_dict(
    graph: Graph,
    node_num_keys: dict,
    node_cat_keys: dict,
    object_type_col: str,
    event_activity_col: str,
    viewpoint: str,
    node_y_mapping: Dict[str, int | float] = None,
    node_type_object: str = "OBJECT",
    node_type_event: str = "EVENT",
    feature_per_category: bool = False,
    add_reverse_edges: bool = False,
    normalize: bool = False,
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

            # Construct x, including normalization if specified
            num_spec = node_num_keys[node_type][t]
            if isinstance(num_spec, dict):
                num_keys = list(num_spec.keys())
            else:
                num_keys = list(num_spec)

            # Build numeric features, optionally normalizing using (min,max) from num_spec
            node_feats = []
            for k in num_keys:
                try:
                    v = float(attr.get(k, 0.0))
                except Exception as inner_e:
                    raise RuntimeError(
                        f"Error converting numeric attribute '{k}' for node '{n}' of type '{node_type}': {inner_e}"
                    ) from inner_e

                if normalize and isinstance(num_spec, dict) and k in num_spec:
                    vmin, vmax = num_spec[k]
                    denom = float(vmax) - float(vmin)
                    if denom == 0.0:
                        node_feats.append(0.0)
                    else:
                        node_feats.append((v - float(vmin)) / denom)
                else:
                    node_feats.append(v)

            feat_labels = num_keys.copy()

            # Encode categorical attributes by their index in the provided unique-values list
            cat_info = node_cat_keys.get(node_type, {}).get(t, {})
            for col, unique_vals in cat_info.items():
                try:
                    val = attr.get(col)
                    idx = 0
                    try:
                        if val is not None:
                            raw_idx = int(unique_vals.index(val))
                            if (
                                normalize
                                and len(unique_vals) > 1
                                and not feature_per_category
                            ):
                                idx = float(raw_idx) / float(len(unique_vals) - 1)
                            else:
                                idx = int(raw_idx)
                        else:
                            idx = -1
                    except ValueError:
                        idx = 0

                    if feature_per_category:
                        col_x = [0] * len(unique_vals)
                        if idx != -1:
                            col_x[idx] = 1
                        node_feats.extend(col_x)
                        feat_labels.extend([f"{col}[{v}]" for v in unique_vals])
                    else:
                        node_feats.append(idx)
                        feat_labels.append(col)

                except Exception as inner_e:
                    raise RuntimeError(
                        f"Error processing categorical attribute '{col}' for node '{n}' of type '{node_type}': {inner_e}"
                    ) from inner_e

            if not node_feats:
                node_feats = [float(0.0)]

            feat_values.append(node_feats)
            node_labels.append(n)

            # Construct y
            if t == viewpoint:
                graph_dict["y_nodes"].append(n)
                if node_y_mapping:
                    y = node_y_mapping.get(n)
                    if y is not None:
                        graph_dict["y"].append(y)
                    else:
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
    graph: Graph,
    node_num_keys: dict,
    node_cat_keys: dict,
    object_type_col: str,
    event_activity_col: str,
    viewpoint: str,
    node_y_mapping: Dict[str, float] = None,
    add_reverse_edges: bool = False,
    path_dataset: str = None,
    normalize: bool = False,
    feature_per_category: bool = False,
) -> Tuple[HeteroData, Set[str], Set[str]]:
    hetero_data = HeteroData()
    graph_dict, node_types, edge_types, feat_label_dict, node_label_dict = (
        construct_graph_dict(
            graph=graph,
            node_num_keys=node_num_keys,
            node_cat_keys=node_cat_keys,
            object_type_col=object_type_col,
            event_activity_col=event_activity_col,
            viewpoint=viewpoint,
            node_y_mapping=node_y_mapping,
            add_reverse_edges=add_reverse_edges,
            normalize=normalize,
            feature_per_category=feature_per_category,
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
