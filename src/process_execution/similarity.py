from networkx import graph_edit_distance


def node_type_similarity(n1, n2):
    """
    Compute type similarity score.
    If node type (EVENT or OBJECT) is different return 0.
    Otherwise, compare the event/object type.
        EVENT: ocel:activity
        OBJECT: ocel:type
    """
    attr1 = n1.get("attr", {})
    attr2 = n2.get("attr", {})
    attr1_t = attr1.get("type", "")
    attr2_t = attr2.get("type", "")

    if attr1_t != attr2_t:
        return 0

    if attr1_t == "OBJECT":
        return int(attr1.get("ocel:type", "") == attr2.get("ocel:type", ""))
    elif attr1_t == "EVENT":
        return int(attr1.get("ocel:activity", "") == attr2.get("ocel:activity", ""))
    else:
        raise NotImplementedError(f"No similarity implemented for type {attr1_t}")


def attribute_similarity(v1, v2):
    # Treat booleans as exact match
    if isinstance(v1, bool) or isinstance(v2, bool):
        return 1.0 if v1 == v2 else 0.0

    # String comparison: exact match
    if isinstance(v1, str) or isinstance(v2, str):
        return 1.0 if v1 == v2 else 0.0

    # Numeric comparison: relative difference
    try:
        a = float(v1)
        b = float(v2)
        denom = max(abs(a), abs(b), 1e-9)
        rel_diff = abs(a - b) / denom
        return max(0.0, 1.0 - rel_diff)
    except Exception:
        # Fallback to equality
        return 1.0 if v1 == v2 else 0.0


def node_attribute_similarity(n1, n2):
    """
    Compute node attribute similarity score.
    """
    attr1 = n1.get("attr", {})
    attr2 = n2.get("attr", {})

    # Compare only common attribute keys
    common_keys = set(attr1.keys()) & set(attr2.keys())
    if not common_keys:
        return 0.0

    sims = []
    for k in common_keys:
        v1 = attr1.get(k)
        v2 = attr2.get(k)

        sims.append(attribute_similarity(v1, v2))

    # Return average similarity across attributes (0..1)
    return float(sum(sims)) / len(sims)


def node_subst_cost(n1, n2):
    # Define weights
    w_type = 0.9
    w_attr = 1 - w_type

    t_sim = node_type_similarity(n1, n2)

    # Only consider attribute similarity if nodes are of the same type
    if t_sim > 0:
        a_sim = node_attribute_similarity(n1, n2)

        return w_type * (1 - t_sim) + w_attr * (1 - a_sim)
    else:
        return 1 - t_sim


def node_del_cost(n1):
    return 1


def edge_subst_cost(e1, e2):
    return 0 if e1 == e2 else 1


def ocel_graph_similarity(g1, g2):
    graph_edit_distance(
        g1,
        g2,
        node_subst_cost=node_subst_cost,
    )
