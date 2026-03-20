from typing import Any, Callable, Dict, Iterable, List, Tuple

from networkx import Graph
from torch import Tensor
from torch_geometric.explain import HeteroExplanation

from tree_search.action import (
    EventNodeDeletion,
    NodeAttributeCategorical,
    NodeAttributeNumeric,
    ObjectNodeDeletion,
    ObjectNodeSubstitution,
)


def _extract_attr(node_data: Any) -> Dict:
    """Return the attribute dict for a node entry from NetworkX.

    Accepts either the form (node_id, {"attr": {...}}) or
    (node_id, {...}) (i.e. already the attr mapping).
    """
    if isinstance(node_data, dict):
        return node_data.get("attr", node_data)
    return {}


def get_nodes_by_importance(
    explanation: HeteroExplanation, node_label_dict: Dict[str, List[str]], top_k=None
):
    """Return a list of nodes ordered by importance (descending).

    Each entry is a dict with keys: `node_type`, `node_index`, `label`, `importance`.
    """
    results = []
    for node_type in explanation.node_types:
        info = explanation[node_type]
        if info is None:
            continue
        mask = info.get("node_mask")
        if mask is None:
            continue
        if not isinstance(mask, Tensor):
            continue

        # Reduce per-feature masks to a scalar per node by averaging features
        if mask.dim() > 1 and mask.size(-1) > 1:
            scalar = mask.mean(dim=-1)
        else:
            scalar = mask.squeeze(-1) if mask.dim() > 1 else mask

        scalar = scalar.detach().cpu()
        labels = node_label_dict.get(node_type, []) or [
            f"{node_type}_{i}" for i in range(scalar.size(0))
        ]

        for i, v in enumerate(scalar.tolist()):
            lbl = labels[i] if i < len(labels) else f"{node_type}_{i}"
            results.append(
                {
                    "node_type": node_type,
                    "node_index": i,
                    "label": lbl,
                    "importance": float(v),
                }
            )

    results_sorted = sorted(results, key=lambda x: x["importance"], reverse=True)
    if top_k is not None:
        return results_sorted[:top_k]
    return results_sorted


def get_feature_labels_by_importance(
    explanation: HeteroExplanation,
    feat_label_dict: Dict[str, List[str]],
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]] = None,
    one_hot_encoding: bool = False,
    top_k=None,
):
    """Return per-node-type feature labels ordered by importance.

    For each node type, features are aggregated across nodes (mean) and
    returned as a list of dicts with keys: `feature`, `importance`.
    """
    out = {}
    category_labels = (
        {
            f"{k}[{label}]": k
            for d1 in node_cat_keys.values()
            for d2 in d1.values()
            for k, v in d2.items()
            for label in v
        }
        if node_cat_keys
        else {}
    )

    for node_type in explanation.node_types:
        node_expl = explanation[node_type]
        if node_expl is None:
            continue
        mask = node_expl.get("node_mask")
        if mask is None or not isinstance(mask, Tensor):
            continue

        # Expect mask shape [num_nodes, num_features] for per-feature importances
        num_feats = explanation.get_node_store(node_type)["x"].size(1)
        if mask.dim() == 1 or (mask.dim() == 2 and mask.size(-1) != num_feats):
            # No per-feature importance available for this node type
            continue

        # Aggregate across nodes -> per-feature importance
        per_feat = mask.mean(dim=0).detach().cpu()
        feat_labels = feat_label_dict.get(node_type, []) or [
            f"feat_{i}" for i in range(per_feat.size(0))
        ]

        if one_hot_encoding:
            # group feature importances by base category extracted from label
            # e.g. 'ocel:activity[Register Customer Order]' -> 'ocel:activity'
            category_vals: Dict[str, List[float]] = {}
            for i, v in enumerate(per_feat.tolist()):
                fname = feat_labels[i] if i < len(feat_labels) else f"feat_{i}"
                base = category_labels[fname] if fname in category_labels else fname
                category_vals.setdefault(base, []).append(v)

            pairs = [
                {"feature": cat, "importance": float(sum(vals) / len(vals))}
                for cat, vals in category_vals.items()
            ]
        else:
            pairs = []
            for i, v in enumerate(per_feat.tolist()):
                fname = feat_labels[i] if i < len(feat_labels) else f"feat_{i}"
                pairs.append({"feature": fname, "importance": float(v)})

        pairs_sorted = sorted(pairs, key=lambda x: x["importance"], reverse=True)
        out[node_type] = pairs_sorted[:top_k] if top_k is not None else pairs_sorted

    return out


def build_object_substitution_actions(
    target_nodes: Iterable[Tuple[Any, Any]],
    ocel_nodes: Iterable[Tuple[Any, Any]],
    graph: Graph,
    object_type_column: str,
    check: Callable[[Dict, Dict], bool] = lambda a, b: True,
    attribute_spec_dict: Dict[str, Any] = None,
) -> List[ObjectNodeSubstitution]:
    """Construct ObjectNodeSubstitution actions from target graph nodes.

    Parameters
    - target_nodes: iterable of (node_id, node_data) from the target process execution
    - ocel_nodes: iterable of (node_id, node_data) from the OCEL graph (e.g. ocel_nx.nodes(data=True))
    - graph: the target process execution NetworkX graph (used to query in_edges for event ids)
    - object_type_column: name of the object type column used to match object classes
    - check: callable(node_attr, subst_attr) -> bool used to determine valid substitutions

    Returns a list of `ObjectNodeSubstitution` instances.
    """
    actions: List[ObjectNodeSubstitution] = []

    discretized_attributes = {}
    for attr, spec in attribute_spec_dict.items():
        if (
            isinstance(spec, (list, tuple))
            and len(spec) == 2
            and all(isinstance(x, (int, float)) for x in spec)
        ):
            discretized_attributes[attr] = spec

    # Normalize ocel_nodes into a list to allow multiple iterations
    ocel_list = list(ocel_nodes)

    target_map = dict(target_nodes)
    for node_id, node_data in target_map.items():
        attr = _extract_attr(node_data)
        if attr.get("type", "") != "OBJECT":
            continue

        # Build candidate substitutions from OCEL nodes
        substitution_objects: List[Tuple[Any, Any]] = []
        for subst_id, subst_data in ocel_list:
            subst_attr = _extract_attr(subst_data)
            if subst_id == node_id:
                continue
            if subst_attr.get(object_type_column, "") != attr.get(
                object_type_column, ""
            ):
                continue
            if not check(attr, subst_attr):
                continue
            substitution_objects.append((subst_id, subst_data))

        # Collect event ids that link to this object (E2O edges), if graph provided
        event_ids: List[Any] = []
        if graph is not None:
            try:
                event_ids = [
                    e
                    for e, _, a in graph.in_edges(node_id, data="attr")
                    if a.get("type") == "E2O"
                ]
            except Exception:
                event_ids = []

        actions.append(
            ObjectNodeSubstitution(
                object_id=node_id,
                object_data=node_data,
                substitution_objects=substitution_objects,
                event_ids=event_ids,
                discretized_attributes=discretized_attributes,
            )
        )

    return actions


def build_node_deletion_actions(
    target_nodes: Iterable[Tuple[Any, Any]],
    nodes_order: Iterable[Any] = None,
    viewpoint: str = None,
    object_type_column: str = "",
) -> List[EventNodeDeletion]:
    """Construct EventNodeDeletion or ObjectNodeDeletion actions for respectively event and object nodes.

    Expects `target_nodes` as an iterable of (node_id, node_data) where node_data
    may be the attribute dict or a mapping containing an `attr` key.
    """
    actions: List[EventNodeDeletion | ObjectNodeDeletion] = []
    target_map = dict(target_nodes)
    nodes_order = nodes_order if nodes_order is not None else target_map.keys()

    for node_id in nodes_order:
        node_data = target_map[node_id]
        attr = _extract_attr(node_data)
        if attr.get("type", "") == "EVENT":
            actions.append(EventNodeDeletion(deletion_options=[[node_id]]))
        elif attr.get("type", "") == "OBJECT":
            # Skip viewpoint nodes
            if attr.get(object_type_column, "") == viewpoint:
                continue
            actions.append(ObjectNodeDeletion(deletion_options=[[node_id]]))
    return actions


def _parse_value_spec(v):
    # Accept (min, max, step) tuples or range objects

    # (min, max, step)
    if (
        isinstance(v, tuple)
        and len(v) == 3
        and all(isinstance(x, (int, float)) for x in v)
    ):
        return v[0], v[1], v[2]

    # range
    if isinstance(v, range):
        return v.start, v.stop, v.step

    raise ValueError(f"Unsupported value specification: {v}")


def build_node_attribute_actions(
    target_nodes: Iterable[Tuple[Any, Any]],
    attribute_spec_dict: Dict[str, Any],
    node_type: str = "EVENT",
    nodes_order: Iterable[Any] = None,
    attr_order: Dict[str, List[str]] = None,
    object_type_column: str = "ocel:type",
) -> List:
    """Construct NodeAttributeNumeric and NodeAttributeCategorical actions.

    `attribute_spec_dict` maps attribute name -> spec where spec is either:
      - a tuple/list `(value_step, value_max)` (numeric)
      - a list/iterable of category values (categorical)

    The function returns a list containing instances of `NodeAttributeNumeric`
    and `NodeAttributeCategorical` as appropriate.
    """
    actions: List = []

    target_map = dict(target_nodes)
    if nodes_order is None:
        iter_nodes = target_map.items()
    else:
        iter_nodes = ((n, target_map.get(n)) for n in nodes_order if n in target_map)

    for node_id, node_data in iter_nodes:
        attr = _extract_attr(node_data)
        if attr.get("type", "") != node_type:
            continue

        # Determine attribute ordering for this node
        if attr_order is not None:
            if node_type == "EVENT":
                ordered_attrs = attr_order.get("EVENT", [])
            else:
                node_obj_type = attr.get(object_type_column, "")
                ordered_attrs = attr_order.get(node_obj_type, [])

            attr_iter = (a for a in ordered_attrs if a in attr)
        else:
            attr_iter = (a for a in list(attr.keys()) if a in attribute_spec_dict)

        for attr_name in attr_iter:
            if attr_name not in attribute_spec_dict:
                continue
            spec = attribute_spec_dict[attr_name]

            # Numeric spec: tuple/list of two numeric values (step, max)
            is_numeric = False
            try:
                if (
                    isinstance(spec, (list, tuple))
                    and len(spec) == 3
                    and all(isinstance(x, (int, float)) for x in spec)
                ):
                    is_numeric = True
            except Exception:
                is_numeric = False

            if is_numeric:
                try:
                    vmin, vmax, step = _parse_value_spec(spec)
                except ValueError:
                    continue
                actions.append(
                    NodeAttributeNumeric(
                        node_id=node_id,
                        attribute_name=attr_name,
                        value_original=attr.get(attr_name),
                        value_step=step,
                        value_min=vmin,
                        value_max=vmax,
                    )
                )
            else:
                # Categorical: expect an iterable/list of category values
                if spec is None:
                    continue
                # Convert to list of values
                try:
                    category_values = list(spec)
                except Exception:
                    continue
                if not category_values:
                    continue
                value_original = attr.get(attr_name)
                actions.append(
                    NodeAttributeCategorical(
                        node_id=node_id,
                        attribute_name=attr_name,
                        value_original=value_original,
                        category_values=category_values,
                    )
                )

    return actions


def construct_attribute_spec_dict(
    attributes: List[str],
    ocel,
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]],
    node_num_keys: Dict[str, Dict[str, Any]],
    num_bins: int = 2,
) -> Dict[str, Any]:
    """Construct a mapping of attribute -> spec for selected attributes.

    - Categorical attributes map to a list of category values (from node_cat_keys).
    - Numeric attributes map to a tuple `(max, min, step)` where `step = (max-min)/num_bins`.

    The function searches `node_cat_keys` and `node_num_keys` for occurrences of
    each attribute across node types and object/event types. If `node_num_keys`
    contains lists rather than min/max pairs, this helper will query `ocel`
    DataFrames to compute min/max.
    """
    specs: Dict[str, Any] = {}

    # Helper to add categorical values preserving order
    def _add_category(attr_name: str, vals: List[Any]):
        if attr_name not in specs:
            specs[attr_name] = []
        existing = set(specs[attr_name])
        for v in vals:
            if v not in existing:
                specs[attr_name].append(v)
                existing.add(v)

    # Gather categorical values from node_cat_keys
    if isinstance(node_cat_keys, dict):
        for type_map in node_cat_keys.values():
            if not isinstance(type_map, dict):
                continue
            for col_map in type_map.values():
                if not isinstance(col_map, dict):
                    continue
                for attr in attributes:
                    if attr in col_map and col_map[attr]:
                        _add_category(attr, list(col_map[attr]))

    # Gather numeric ranges from node_num_keys
    # Collect min/max values across all occurrences and then compute step
    numeric_ranges: Dict[str, List[Tuple[float, float]]] = {}
    if isinstance(node_num_keys, dict):
        for node_type_key, type_map in node_num_keys.items():
            if not isinstance(type_map, dict):
                continue
            for obj_type, spec in type_map.items():
                # spec may be a dict mapping col->(min,max) or a list of col names
                for attr in attributes:
                    if isinstance(spec, dict) and attr in spec:
                        vmin, vmax = spec[attr]
                        numeric_ranges.setdefault(attr, []).append(
                            (float(vmin), float(vmax))
                        )
                    elif isinstance(spec, (list, tuple)) and attr in spec:
                        # fall back to ocel DataFrames to compute min/max per obj/event type
                        try:
                            if node_type_key.upper().startswith("OBJECT"):
                                df = ocel.objects
                                df_t = df[df[ocel.object_type_column] == obj_type]
                            else:
                                df = ocel.events
                                df_t = df[df[ocel.event_activity] == obj_type]
                            if not df_t.empty and attr in df_t.columns:
                                vmin = float(df_t[attr].min())
                                vmax = float(df_t[attr].max())
                                numeric_ranges.setdefault(attr, []).append((vmin, vmax))
                        except Exception:
                            continue

    # If any numeric ranges found, compute global min/max and step
    for attr, ranges in numeric_ranges.items():
        if not ranges:
            continue
        global_min = min(r[0] for r in ranges)
        global_max = max(r[1] for r in ranges)
        if global_max == global_min:
            step = 1.0
        else:
            step = (global_max - global_min) / float(max(1, num_bins))
        specs[attr] = (global_min, global_max, step)

    # Note: categorical specs (if any) will already have been added above
    return specs


__all__ = [
    "get_nodes_by_importance",
    "get_feature_labels_by_importance",
    "build_object_substitution_actions",
    "build_node_deletion_actions",
    "build_node_attribute_actions",
    "construct_attribute_spec_dict",
]
