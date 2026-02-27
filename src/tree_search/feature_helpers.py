from typing import Any, Callable, Dict, Iterable, List, Tuple

from networkx import Graph

from tree_search.feature import (
    EventNodeDeletion,
    NodeAttributeCategorical,
    NodeAttributeNumeric,
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


def build_object_substitution_features(
    target_nodes: Iterable[Tuple[Any, Any]],
    ocel_nodes: Iterable[Tuple[Any, Any]],
    graph: Graph,
    object_type_column: str,
    check: Callable[[Dict, Dict], bool] = lambda a, b: True,
    nodes_order: Iterable[Any] = None,
    attribute_spec_dict: Dict[str, Any] = None,
) -> List[ObjectNodeSubstitution]:
    """Construct ObjectNodeSubstitution features from target graph nodes.

    Parameters
    - target_nodes: iterable of (node_id, node_data) from the target process execution
    - ocel_nodes: iterable of (node_id, node_data) from the OCEL graph (e.g. ocel_nx.nodes(data=True))
    - graph: the target process execution NetworkX graph (used to query in_edges for event ids)
    - object_type_column: name of the object type column used to match object classes
    - check: callable(node_attr, subst_attr) -> bool used to determine valid substitutions

    Returns a list of `ObjectNodeSubstitution` instances.
    """
    features: List[ObjectNodeSubstitution] = []

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

    # Normalize target nodes into a mapping so we can honor ordering if requested
    target_map = dict(target_nodes)
    if nodes_order is None:
        iter_nodes = target_map.items()
    else:
        iter_nodes = ((n, target_map.get(n)) for n in nodes_order if n in target_map)

    for node_id, node_data in iter_nodes:
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

        features.append(
            ObjectNodeSubstitution(
                object_id=node_id,
                object_data=node_data,
                substitution_objects=substitution_objects,
                event_ids=event_ids,
                discretized_attributes=discretized_attributes,
            )
        )

    return features


def build_event_deletion_features(
    target_nodes: Iterable[Tuple[Any, Any]],
) -> List[EventNodeDeletion]:
    """Construct EventNodeDeletion features for event nodes.

    Expects `target_nodes` as an iterable of (node_id, node_data) where node_data
    may be the attribute dict or a mapping containing an `attr` key.
    """
    features: List[EventNodeDeletion] = []
    for node_id, node_data in target_nodes:
        attr = _extract_attr(node_data)
        if attr.get("type", "") == "EVENT":
            features.append(EventNodeDeletion(deletion_options=[[node_id]]))
    return features


def _parse_value_spec(v):
    # Accept (step, max) tuples, range objects, numpy arange-like arrays, or lists
    try:
        import numpy as _np
    except Exception:
        _np = None

    # (step, max)
    if (
        isinstance(v, tuple)
        and len(v) == 2
        and all(isinstance(x, (int, float)) for x in v)
    ):
        return v[0], v[1]

    # range
    if isinstance(v, range):
        return v.step, v.stop

    # numpy arrays or array-likes
    if _np is not None and isinstance(v, _np.ndarray):
        arr = v.tolist()
        if len(arr) >= 2:
            step = arr[1] - arr[0]
            return step, arr[-1] + step

    # list/tuple
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        step = v[1] - v[0]
        return step, v[-1] + step

    raise ValueError(f"Unsupported value specification: {v}")


def build_node_attribute_features(
    target_nodes: Iterable[Tuple[Any, Any]],
    attribute_spec_dict: Dict[str, Any],
    node_type: str = "EVENT",
    nodes_order: Iterable[Any] = None,
    attr_order: Dict[str, List[str]] = None,
    object_type_column: str = None,
) -> List:
    """Construct NodeAttributeNumeric and NodeAttributeCategorical features.

    `attribute_spec_dict` maps attribute name -> spec where spec is either:
      - a tuple/list `(value_step, value_max)` (numeric)
      - a list/iterable of category values (categorical)

    The function returns a list containing instances of `NodeAttributeNumeric`
    and `NodeAttributeCategorical` as appropriate.
    """
    features: List = []

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
        if attr_order:
            if node_type == "EVENT":
                ordered_attrs = attr_order.get("EVENT", [])
            else:
                if object_type_column:
                    node_obj_type = attr.get(object_type_column, "")
                    ordered_attrs = attr_order.get(node_obj_type, [])
                else:
                    ordered_attrs = []
        else:
            ordered_attrs = []

        if ordered_attrs:
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
                    and len(spec) == 2
                    and all(isinstance(x, (int, float)) for x in spec)
                ):
                    is_numeric = True
            except Exception:
                is_numeric = False

            if is_numeric:
                try:
                    step, vmax = _parse_value_spec(spec)
                except ValueError:
                    continue
                features.append(
                    NodeAttributeNumeric(
                        node_id=node_id,
                        attribute_name=attr_name,
                        value_original=attr.get(attr_name),
                        value_step=step,
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
                features.append(
                    NodeAttributeCategorical(
                        node_id=node_id,
                        attribute_name=attr_name,
                        value_original=value_original,
                        category_values=category_values,
                    )
                )

    return features


def construct_attribute_spec_dict(
    attributes: List[str],
    ocel,
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]],
    node_num_keys: Dict[str, Dict[str, Any]],
    num_bins: int = 2,
) -> Dict[str, Any]:
    """Construct a mapping of attribute -> spec for selected attributes.

    - Categorical attributes map to a list of category values (from node_cat_keys).
    - Numeric attributes map to a tuple `(step, max)` where `step = (max-min)/num_bins`.

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
        specs[attr] = (step, global_max)

    # Note: categorical specs (if any) will already have been added above
    return specs


__all__ = [
    "build_object_substitution_features",
    "build_event_deletion_features",
    "build_node_attribute_features",
    "construct_attribute_spec_dict",
]
