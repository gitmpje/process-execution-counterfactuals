from typing import Any, Callable, Dict, Iterable, List, Tuple

from networkx import Graph

from tree_search.feature import (
    EventNodeDeletion,
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
    discretized_event_attributes: Dict[str, Any] = None,
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
                discretized_event_attributes=discretized_event_attributes,
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


def build_node_attribute_numeric(
    target_nodes: Iterable[Tuple[Any, Any]],
    selected_event_attributes: Dict[str, Any],
    node_type: str = "EVENT",
    nodes_order: Iterable[Any] = None,
    attr_order: Dict[str, List[str]] = None,
    object_type_column: str = None,
) -> List[NodeAttributeNumeric]:
    """Construct NodeAttributeNumeric features for numeric node attributes.

    - `selected_event_attributes` maps attribute name -> value spec. The value spec can be:
      - (value_step, value_max)
      - a `range` object
      - a numpy `arange`/array or list/tuple of values
    - `node_type` filters nodes by their `type` attribute ("EVENT" or "OBJECT").
    - `nodes_order` optional iterable of node ids to control ordering of generated features.
    - `attr_order` optional mapping used to order attributes per node key. For `EVENT` nodes use key 'EVENT',
       for `OBJECT` nodes keys should be object type names and `object_type_column` must be provided.
    """
    features: List[NodeAttributeNumeric] = []

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
            attr_iter = (a for a in list(attr.keys()) if a in selected_event_attributes)

        for attr_name in attr_iter:
            if attr_name in selected_event_attributes:
                try:
                    step, vmax = _parse_value_spec(selected_event_attributes[attr_name])
                except ValueError:
                    continue
                features.append(
                    NodeAttributeNumeric(
                        node_id=node_id,
                        attribute_name=attr_name,
                        value_original=attr[attr_name],
                        value_step=step,
                        value_max=vmax,
                    )
                )
    return features


__all__ = [
    "build_object_substitution_features",
    "build_event_deletion_features",
    "build_node_attribute_numeric",
]
