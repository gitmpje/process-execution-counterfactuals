from networkx import graph_edit_distance
from pm4py.objects.ocel.constants import DEFAULT_EVENT_ACTIVITY, DEFAULT_OBJECT_TYPE
from typing import Any, Dict, List


def node_type_diff(n1, n2):
    """
    Compute type difference score.
    If node type (EVENT or OBJECT) is different return 1.
    Otherwise, compare the event/object type.
        EVENT: ocel:activity
        OBJECT: ocel:type
    """
    attr1 = n1.get("attr", {})
    attr2 = n2.get("attr", {})
    type1 = attr1.get("type", "")
    type2 = attr2.get("type", "")

    if type1 != type2:
        return 1

    if type1 == "OBJECT":
        return int(
            attr1.get(DEFAULT_OBJECT_TYPE, "") != attr2.get(DEFAULT_OBJECT_TYPE, "")
        )
    elif type1 == "EVENT":
        return int(
            attr1.get(DEFAULT_EVENT_ACTIVITY, "")
            != attr2.get(DEFAULT_EVENT_ACTIVITY, "")
        )
    else:
        raise NotImplementedError(f"No difference implemented for type {type1}")


def attribute_diff_numeric(v1, v2, interval_size: int | float = None):
    # Numeric comparison: relative difference
    try:
        a = float(v1)
        b = float(v2)
        if interval_size:
            return abs(a - b) / interval_size
        else:
            denom = max(abs(a), abs(b), 1e-9)
            rel_diff = abs(a - b) / denom
            return rel_diff
    except Exception:
        # Fallback to equality
        return 0.0 if v1 == v2 else 1.0


def attribute_diff(v1, v2, interval_size: int | float = None):
    # Treat booleans as exact match
    if isinstance(v1, bool) or isinstance(v2, bool):
        return 0.0 if v1 == v2 else 1.0

    # String comparison: exact match
    if isinstance(v1, str) or isinstance(v2, str):
        return 0.0 if v1 == v2 else 1.0

    return attribute_diff_numeric(v1, v2, interval_size)


def node_attribute_diffs(
    n1,
    n2,
    exclude_attributes: List[str] = None,
    discretized_event_attributes: Dict[str, Any] = None,
):
    """
    Compute node attribute differences.
    """
    attr1 = n1.get("attr", {})
    attr2 = n2.get("attr", {})

    # Compare only common attribute keys
    common_keys = set(attr1.keys()) & set(attr2.keys())
    if not common_keys:
        return 0.0

    attribute_labels = [k for k in common_keys if k not in exclude_attributes]

    diffs = []
    for k in attribute_labels:
        v1 = attr1.get(k)
        v2 = attr2.get(k)
        discr = discretized_event_attributes.get(k)
        if discr:
            diffs.append(attribute_diff(v1, v2, interval_size=discr[0]))
        else:
            diffs.append(attribute_diff(v1, v2))

    return diffs


def node_subst_cost(
    n1,
    n2,
    w_type: float = 2.0,
    w_attr: float = 1.0,
    exclude_attributes: List[str] = None,
    aggregation_type: str = "average",
    discretized_event_attributes: Dict[str, Any] = None,
):
    # Type difference
    t_diff = node_type_diff(n1, n2)

    # Only consider attribute difference if nodes are of the same type
    if t_diff == 0:
        attr_diffs = node_attribute_diffs(
            n1,
            n2,
            exclude_attributes=exclude_attributes,
            discretized_event_attributes=discretized_event_attributes,
        )
        if aggregation_type == "average":
            # Return average difference across attributes (0..1)
            attr_diff = float(sum(attr_diffs)) / len(attr_diffs)
        elif aggregation_type == "count":
            # Count differences > 0
            attr_diff = len([v for v in attr_diffs if v != 0])
        elif aggregation_type == "sum":
            # Return sum of difference across attributes
            attr_diff = float(sum(attr_diffs))
        else:
            raise NotImplementedError(f"{aggregation_type} not implemented")

        return w_type * t_diff + w_attr * attr_diff
    else:
        return t_diff


def edge_subst_cost(e1, e2):
    return 0 if e1 == e2 else 1


def node_del_cost(n):
    return 1


def edge_del_cost(e):
    return 1


def node_ins_cost(n):
    return 1


def edge_ins_cost(e):
    return 1


def ocel_graph_edit_distance(g1, g2):
    graph_edit_distance(
        g1,
        g2,
        node_subst_cost=node_subst_cost,
        edge_subst_cost=edge_subst_cost,
        node_del_cost=node_del_cost,
        edge_del_cost=edge_del_cost,
        node_ins_cost=node_ins_cost,
        edge_ins_cost=edge_ins_cost,
    )
