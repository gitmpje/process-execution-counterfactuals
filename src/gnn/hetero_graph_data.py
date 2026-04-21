import numpy as np
import torch

from collections import defaultdict
from networkx import Graph
from torch_geometric.data import Data, HeteroData
from typing import Any, Dict, List, Tuple, Set


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
    one_hot_encoding: bool = False,
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
    type_to_nodes = {t: [] for t in event_object_types}
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

        if t in node_num_keys.get(node_type_object, {}):
            node_type_key = node_type_object
        elif t in node_num_keys.get(node_type_event, {}):
            node_type_key = node_type_event
        else:
            node_type_key = None

        if node_type_key is not None:
            num_spec = node_num_keys[node_type_key].get(t, {})
            if isinstance(num_spec, dict):
                num_keys = list(num_spec.keys())
            else:
                num_keys = list(num_spec)

            feat_labels = num_keys.copy()
            cat_info = node_cat_keys.get(node_type_key, {}).get(t, {})
            if one_hot_encoding:
                for col, unique_vals in cat_info.items():
                    feat_labels.extend([f"{col}[{v}]" for v in unique_vals])
            else:
                feat_labels.extend(cat_info.keys())
        else:
            feat_labels = []

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
                                and not one_hot_encoding
                            ):
                                idx = float(raw_idx) / float(len(unique_vals) - 1)
                            else:
                                idx = int(raw_idx)
                        else:
                            idx = -1
                    except ValueError:
                        idx = 0

                    if one_hot_encoding:
                        col_x = [0] * len(unique_vals)
                        if idx != -1:
                            col_x[idx] = 1
                        node_feats.extend(col_x)
                    else:
                        node_feats.append(idx)

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
        list(type_to_nodes.keys()),
        list(edge_dict.keys()),
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
    one_hot_encoding: bool = False,
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
            one_hot_encoding=one_hot_encoding,
        )
    )

    if viewpoint not in node_types:
        raise ValueError(f"No nodes of viewpoint type {viewpoint} occur in the graph")

    for t in node_types:
        values = graph_dict[t]
        if not values:
            continue
        hetero_data[t].x = torch.tensor(values, dtype=torch.float32).reshape(
            len(values), -1
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


def _feature_dimension_for_type(
    type_name: str,
    node_num_keys: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]],
    one_hot_encoding: bool = False,
) -> int:
    if type_name in node_num_keys["OBJECT"]:
        node_type_key = "OBJECT"
    elif type_name in node_num_keys["EVENT"]:
        node_type_key = "EVENT"
    else:
        raise ValueError(
            f"Cannot determine node kind for feature dimension of type '{type_name}'"
        )

    num_spec = node_num_keys[node_type_key].get(type_name, {})
    if isinstance(num_spec, dict):
        num_keys = list(num_spec.keys())
    else:
        num_keys = list(num_spec)

    dim = len(num_keys)
    cat_info = node_cat_keys.get(node_type_key, {}).get(type_name, {})
    if one_hot_encoding:
        dim += sum(len(unique_vals) for unique_vals in cat_info.values())
    else:
        dim += len(cat_info)

    return dim if dim > 0 else 1


def _pad_node_features_to_width(
    hetero_data: HeteroData,
    node_types: List[str],
    feature_dim: int,
) -> None:
    # Infer device from existing tensors in hetero_data
    device = None
    for node_type in hetero_data.node_types:
        if (
            hasattr(hetero_data[node_type], "x")
            and hetero_data[node_type].x is not None
        ):
            device = hetero_data[node_type].x.device
            break
    if device is None:
        device = torch.device("cpu")

    for node_type in node_types:
        if node_type not in hetero_data.node_types:
            hetero_data[node_type].x = torch.zeros(
                (0, feature_dim), dtype=torch.float32, device=device
            )
            continue

        x = hetero_data[node_type].x
        if x.size(1) < feature_dim:
            pad = x.new_zeros((x.size(0), feature_dim - x.size(1)))
            hetero_data[node_type].x = torch.cat([x, pad], dim=1)


def _pad_node_features_to_unique_width(
    hetero_data: HeteroData,
    node_types: List[str],
    feature_dim_by_type: Dict[str, int],
    global_feature_dim: int,
) -> None:
    # Infer device from existing tensors in hetero_data
    device = None
    for node_type in hetero_data.node_types:
        if (
            hasattr(hetero_data[node_type], "x")
            and hetero_data[node_type].x is not None
        ):
            device = hetero_data[node_type].x.device
            break
    if device is None:
        device = torch.device("cpu")

    offset = 0
    for node_type in node_types:
        width = feature_dim_by_type[node_type]
        if node_type not in hetero_data.node_types:
            hetero_data[node_type].x = torch.zeros(
                (0, global_feature_dim), dtype=torch.float32, device=device
            )
            offset += width
            continue

        x = hetero_data[node_type].x
        if x.size(1) < width:
            pad = x.new_zeros((x.size(0), width - x.size(1)))
            x = torch.cat([x, pad], dim=1)
        elif x.size(1) > width:
            raise ValueError(
                f"Feature dimension for node type '{node_type}' is larger than expected. "
                f"expected={width}, found={x.size(1)}"
            )

        full = x.new_zeros((x.size(0), global_feature_dim))
        full[:, offset : offset + width] = x
        hetero_data[node_type].x = full
        offset += width


def to_homogeneous_data(
    hetero_data: HeteroData,
    node_num_keys: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]],
    node_types: List[str] | None = None,
    one_hot_encoding: bool = False,
    unique_node_type_attribute_columns: bool = False,
) -> Data:
    all_node_types = (
        list(node_types) if node_types is not None else list(hetero_data.node_types)
    )
    if not all_node_types:
        raise ValueError("No node types provided for homogeneous conversion")

    if unique_node_type_attribute_columns:
        feature_dim_by_type = {
            node_type: _feature_dimension_for_type(
                type_name=node_type,
                node_num_keys=node_num_keys,
                node_cat_keys=node_cat_keys,
                one_hot_encoding=one_hot_encoding,
            )
            for node_type in all_node_types
        }
        total_feature_dim = sum(feature_dim_by_type.values())
        _pad_node_features_to_unique_width(
            hetero_data,
            all_node_types,
            feature_dim_by_type,
            total_feature_dim,
        )
    else:
        feature_dim = max(
            _feature_dimension_for_type(
                type_name=node_type,
                node_num_keys=node_num_keys,
                node_cat_keys=node_cat_keys,
                one_hot_encoding=one_hot_encoding,
            )
            for node_type in all_node_types
        )

        _pad_node_features_to_width(hetero_data, all_node_types, feature_dim)

    data = hetero_data.to_homogeneous()
    if hasattr(hetero_data, "y"):
        data.y = hetero_data.y

    if not hasattr(data, "edge_index") or data.edge_index is None:
        data.edge_index = torch.empty((2, 0), dtype=torch.int64)

    return data


def build_data(
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
    one_hot_encoding: bool = False,
    unique_node_type_attribute_columns: bool = False,
    node_types: List[str] | None = None,
) -> Tuple[
    Data, Set[str], Set[str], List[str], Dict[str, List[str]], Dict[str, List[str]]
]:
    hetero_data, type_names, edge_types, y_nodes, feat_label_dict, node_label_dict = (
        build_hetero_data(
            graph=graph,
            node_num_keys=node_num_keys,
            node_cat_keys=node_cat_keys,
            object_type_col=object_type_col,
            event_activity_col=event_activity_col,
            viewpoint=viewpoint,
            node_y_mapping=node_y_mapping,
            add_reverse_edges=add_reverse_edges,
            normalize=normalize,
            one_hot_encoding=one_hot_encoding,
        )
    )

    data = to_homogeneous_data(
        hetero_data,
        node_num_keys=node_num_keys,
        node_cat_keys=node_cat_keys,
        node_types=node_types or sorted(type_names),
        one_hot_encoding=one_hot_encoding,
        unique_node_type_attribute_columns=unique_node_type_attribute_columns,
    )

    if path_dataset:
        torch.save(data, path_dataset)

    return data, type_names, edge_types, y_nodes, feat_label_dict, node_label_dict
