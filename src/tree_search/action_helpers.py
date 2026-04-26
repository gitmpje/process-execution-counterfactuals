from copy import deepcopy
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from networkx import Graph
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explanation, HeteroExplanation

from gnn.hetero_graph_data import to_homogeneous_data
from gnn.utils import Metadata
from tree_search.action import (
    EventNodeDeletion,
    EventNodeInsertion,
    EventNodeMove,
    EventNodeSubstitution,
    NodeAttributeCategorical,
    NodeAttributeNumeric,
    ObjectNodeDeletion,
    ObjectNodeInsertion,
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
    explanation: Explanation | HeteroExplanation,
    node_label_dict: Dict[str, List[str]],
    top_k=None,
    metadata: Metadata = None,
    hetero_data: HeteroData = None,
) -> List[Dict]:
    """Return a list of nodes ordered by importance (descending).

    Each entry is a dict with keys: `node_type`, `node_index`, `label`, `importance`.

    For a homogeneous ``Explanation``, nodes are mapped back to their original
    hetero node types using ``hetero_data.to_homogeneous().node_type``, so
    ``node_type`` in each result reflects the original type (e.g. ``"PO"``,
    ``"item"``, ``"EVENT"``) and ``node_index`` is the index within that type.

    Args:
        explanation:     A PyG ``Explanation`` (homogeneous) or ``HeteroExplanation``.
        node_label_dict: Maps original hetero node-type name → list of
                         human-readable node labels.  Pass the original hetero
                         dict from ``build_hetero_data`` for both paths.
        top_k:           If set, return only the top-k most important nodes
                         across all types.
        hetero_data:     [Homogeneous only] The original ``HeteroData`` object
                         (before ``to_homogeneous()``).  Required when
                         ``explanation`` is a homogeneous ``Explanation``.
    """
    results = []

    if isinstance(explanation, HeteroExplanation):
        # --- heterogeneous explanation ---
        for nt in explanation.node_types:
            info = explanation[nt]
            if info is None:
                continue
            mask = info.get("node_mask")
            if mask is None or not isinstance(mask, Tensor):
                continue

            scalar = _reduce_node_mask(mask)
            labels = node_label_dict.get(nt, []) or []
            for i, v in enumerate(scalar.tolist()):
                results.append(
                    {
                        "node_type": nt,
                        "node_index": i,
                        "label": labels[i] if i < len(labels) else f"{nt}_{i}",
                        "importance": float(v),
                    }
                )

    else:
        # --- homogeneous explanation ---
        if (hetero_data is None) or (metadata is None):
            raise ValueError(
                "hetero_data must be provided for homogeneous Explanation so that "
                "nodes can be mapped back to their original node types."
            )

        mask = explanation.get("node_mask")
        if mask is None or not isinstance(mask, Tensor):
            return results

        homo_data = to_homogeneous_data(
            hetero_data,
            metadata.node_num_keys,
            metadata.node_cat_keys,
            metadata.node_types,
            metadata.one_hot_encoding,
            metadata.unique_node_type_attribute_columns,
        )
        orig_node_types = hetero_data.node_types  # str list, index = type int
        node_type_tensor = homo_data.node_type  # shape [num_nodes]

        scalar = _reduce_node_mask(mask)  # shape [num_nodes]

        for type_idx, nt in enumerate(orig_node_types):
            node_mask_idx = (node_type_tensor == type_idx).nonzero(as_tuple=True)[0]
            if node_mask_idx.numel() == 0:
                continue

            labels = node_label_dict.get(nt, []) or []
            for i, global_idx in enumerate(node_mask_idx.tolist()):
                results.append(
                    {
                        "node_type": nt,
                        "node_index": i,
                        "label": labels[i] if i < len(labels) else f"{nt}_{i}",
                        "importance": float(scalar[global_idx].item()),
                    }
                )

    results_sorted = sorted(results, key=lambda x: x["importance"], reverse=True)
    return results_sorted[:top_k] if top_k is not None else results_sorted


def _reduce_node_mask(mask: Tensor) -> Tensor:
    """Reduce a node mask to a scalar importance per node by averaging features."""
    if mask.dim() > 1 and mask.size(-1) > 1:
        return mask.mean(dim=-1).detach().cpu()
    return (mask.squeeze(-1) if mask.dim() > 1 else mask).detach().cpu()


def get_feature_labels_by_importance(
    explanation: Explanation | HeteroExplanation,
    feat_label_dict: Dict[str, List[str]],
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]] = None,
    one_hot_encoding: bool = False,
    top_k=None,
    metadata: Metadata = None,
    hetero_data: HeteroData = None,
) -> Dict[str, List]:
    """Return per-node-type feature labels ordered by importance.

    For each node type, features are aggregated across nodes (mean) and
    returned as a list of dicts with keys: ``feature``, ``importance``.

    For a homogeneous ``Explanation``, importances are mapped back to the
    original hetero node types using ``hetero_data.to_homogeneous().node_type``,
    so the returned dict has the same keys as the original ``feat_label_dict``
    (e.g. ``{'PO': [...], 'item': [...], 'EVENT': [...]}``) rather than a
    flat ``"node"`` key.

    Args:
        explanation:      A PyG ``Explanation`` (homogeneous) or ``HeteroExplanation``.
        feat_label_dict:  Maps original hetero node-type name → list of feature
                          names.  Pass the original hetero dict from
                          ``build_hetero_data`` for both hetero and homogeneous
                          explanations.
        node_cat_keys:    Optional categorical-feature metadata used when
                          ``one_hot_encoding=True``.
        one_hot_encoding: If ``True``, group one-hot expanded features back to
                          their base category and average their importances.
        top_k:            If set, return only the top-k features per node type.
        hetero_data:      [Homogeneous only] The original ``HeteroData`` object
                          (before ``to_homogeneous()``).  Required when
                          ``explanation`` is a homogeneous ``Explanation`` so
                          that the flat node/feature indices can be mapped back
                          to their original node types.
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

    if isinstance(explanation, HeteroExplanation):
        # --- heterogeneous explanation ---
        node_types = explanation.node_types

        def get_mask_and_x(nt):
            node_expl = explanation[nt]
            if node_expl is None:
                return None, None
            return node_expl.get("node_mask"), explanation.get_node_store(nt).get("x")

        def get_feat_labels(nt, size):
            return feat_label_dict.get(nt, []) or [f"feat_{i}" for i in range(size)]

        for nt in node_types:
            mask, x = get_mask_and_x(nt)
            if mask is None or not isinstance(mask, Tensor) or x is None:
                continue

            num_feats = x.size(1)
            if mask.dim() == 1 or (mask.dim() == 2 and mask.size(-1) != num_feats):
                continue

            per_feat = mask.mean(dim=0).detach().cpu()
            feat_labels = get_feat_labels(nt, per_feat.size(0))
            out[nt] = _make_pairs(
                per_feat, feat_labels, category_labels, one_hot_encoding, top_k
            )

    else:
        # --- homogeneous explanation ---
        if (hetero_data is None) or (metadata is None):
            raise ValueError(
                "hetero_data must be provided for homogeneous Explanation so that "
                "feature importances can be mapped back to the original node types."
            )

        mask = explanation.get("node_mask")
        x = explanation.get("x")
        if mask is None or not isinstance(mask, Tensor) or x is None:
            return out

        num_feats = x.size(1)
        if mask.dim() == 1 or (mask.dim() == 2 and mask.size(-1) != num_feats):
            return out

        homo_data = to_homogeneous_data(
            hetero_data,
            metadata.node_num_keys,
            metadata.node_cat_keys,
            metadata.node_types,
            metadata.one_hot_encoding,
            metadata.unique_node_type_attribute_columns,
        )
        orig_node_types = hetero_data.node_types  # str list, index = type int
        node_type_tensor = homo_data.node_type  # shape [num_nodes]

        # Average mask over nodes, but only for rows belonging to each type,
        # and only over that type's own feature columns.
        for type_idx, nt in enumerate(orig_node_types):
            node_store = hetero_data[nt]
            if not hasattr(node_store, "x") or node_store.x is None:
                continue

            nt_num_feats = node_store.x.size(1)
            # Row indices in the homogeneous graph that belong to this type.
            node_mask_idx = (node_type_tensor == type_idx).nonzero(as_tuple=True)[0]
            if node_mask_idx.numel() == 0:
                continue

            # Slice rows for this type, then only the columns it owns.
            per_feat = mask[node_mask_idx, :nt_num_feats].mean(dim=0).detach().cpu()
            feat_labels = feat_label_dict.get(nt, []) or [
                f"feat_{i}" for i in range(nt_num_feats)
            ]
            out[nt] = _make_pairs(
                per_feat, feat_labels, category_labels, metadata.one_hot_encoding, top_k
            )

    return out


def _make_pairs(
    per_feat: Tensor,
    feat_labels: List[str],
    category_labels: Dict[str, str],
    one_hot_encoding: bool,
    top_k,
) -> List[Dict]:
    """Turn a per-feature importance vector into a sorted list of dicts."""
    if one_hot_encoding:
        category_vals: Dict[str, List[float]] = {}
        for i, v in enumerate(per_feat.tolist()):
            fname = feat_labels[i] if i < len(feat_labels) else f"feat_{i}"
            base = category_labels.get(fname, fname)
            category_vals.setdefault(base, []).append(v)
        pairs = [
            {"feature": cat, "importance": float(sum(vals) / len(vals))}
            for cat, vals in category_vals.items()
        ]
    else:
        pairs = [
            {
                "feature": feat_labels[i] if i < len(feat_labels) else f"feat_{i}",
                "importance": float(v),
            }
            for i, v in enumerate(per_feat.tolist())
        ]

    pairs_sorted = sorted(pairs, key=lambda x: x["importance"], reverse=True)
    return pairs_sorted[:top_k] if top_k is not None else pairs_sorted


def get_dataset_category_importances(
    explanation: Explanation | HeteroExplanation,
    feat_label_dict: Dict[str, List[str]],
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]],
    metadata: Metadata = None,
    hetero_data: HeteroData = None,
) -> Dict[str, Dict[str, Dict[str, Dict[Any, float]]]]:
    """Return a copy of node_cat_keys extended with dataset-level importance scores.

    For each categorical attribute, the list of categories is replaced with a dict
    mapping category value to its importance score (averaged across nodes and features).
    Categories not present in the explanation receive an importance of 0.0.
    """
    result = {}

    nt_ocel_type = {
        nt: ocel_type for ocel_type, d in node_cat_keys.items() for nt in d.keys()
    }
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

    # Compute per-feature importances
    if isinstance(explanation, HeteroExplanation):
        node_types = explanation.node_types
        for nt in node_types:
            node_expl = explanation[nt]
            if node_expl is None:
                continue
            mask = node_expl.get("node_mask")
            x = explanation.get_node_store(nt).get("x")
            if mask is None or not isinstance(mask, Tensor) or x is None:
                continue
            num_feats = x.size(1)
            if mask.dim() == 1 or (mask.dim() == 2 and mask.size(-1) != num_feats):
                continue
            per_feat = mask.mean(dim=0).detach().cpu()
            feat_labels = feat_label_dict.get(nt, []) or [
                f"feat_{i}" for i in range(num_feats)
            ]
            for i, imp in enumerate(per_feat.tolist()):
                fname = feat_labels[i] if i < len(feat_labels) else f"feat_{i}"
                base = category_labels.get(fname, None)
                if base is not None and "[" in fname and "]" in fname:
                    val = fname.split("[")[1].split("]")[0]
                    ocel_type = nt_ocel_type[nt]
                    cat_keys = node_cat_keys[ocel_type].get(nt, {})
                    categories = cat_keys.get(base, [])
                    if val in categories:
                        result.setdefault(ocel_type, {}).setdefault(nt, {}).setdefault(
                            base, {}
                        )[val] = float(imp)
    else:
        if (hetero_data is None) or (metadata is None):
            pass  # result remains empty
        else:
            mask = explanation.get("node_mask")
            x = explanation.get("x")
            if mask is None or not isinstance(mask, Tensor) or x is None:
                pass
            else:
                num_feats = x.size(1)
                if not (
                    mask.dim() == 1 or (mask.dim() == 2 and mask.size(-1) != num_feats)
                ):
                    homo_data = to_homogeneous_data(
                        hetero_data,
                        metadata.node_num_keys,
                        metadata.node_cat_keys,
                        metadata.node_types,
                        metadata.one_hot_encoding,
                        metadata.unique_node_type_attribute_columns,
                    )
                    orig_node_types = hetero_data.node_types
                    node_type_tensor = homo_data.node_type
                    for type_idx, nt in enumerate(orig_node_types):
                        node_store = hetero_data[nt]
                        if not hasattr(node_store, "x") or node_store.x is None:
                            continue
                        nt_num_feats = node_store.x.size(1)
                        node_mask_idx = (node_type_tensor == type_idx).nonzero(
                            as_tuple=True
                        )[0]
                        if node_mask_idx.numel() == 0:
                            continue
                        per_feat = (
                            mask[node_mask_idx, :nt_num_feats]
                            .mean(dim=0)
                            .detach()
                            .cpu()
                        )
                        feat_labels = feat_label_dict.get(nt, []) or [
                            f"feat_{i}" for i in range(nt_num_feats)
                        ]
                        for i, imp in enumerate(per_feat.tolist()):
                            fname = (
                                feat_labels[i] if i < len(feat_labels) else f"feat_{i}"
                            )
                            base = category_labels.get(fname, None)
                            if base is not None and "[" in fname and "]" in fname:
                                val = fname.split("[")[1].split("]")[0]
                                ocel_type = nt_ocel_type[nt]
                                cat_keys = node_cat_keys[ocel_type].get(nt, {})
                                categories = cat_keys.get(base, [])
                                if val in categories:
                                    result.setdefault(ocel_type, {}).setdefault(
                                        nt, {}
                                    ).setdefault(base, {})[val] = float(imp)

    # Build the full result by copying node_cat_keys and replacing lists with dicts
    for ocel_type, type_dict in node_cat_keys.items():
        result.setdefault(ocel_type, {})
        for nt, attrs in type_dict.items():
            result[ocel_type].setdefault(nt, {})
            for attr, cats in attrs.items():
                if isinstance(cats, list):
                    imp_dict = {
                        cat: result.get(ocel_type, {})
                        .get(nt, {})
                        .get(attr, {})
                        .get(cat, 0.0)
                        for cat in cats
                    }
                    result[ocel_type][nt][attr] = imp_dict
                else:
                    result[ocel_type][nt][attr] = cats

    return result


def _initialize_category_importances(
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]],
) -> Dict[str, Dict[str, Dict[str, Dict[Any, float]]]]:
    """Create a zero-initialized category importance structure from node_cat_keys."""
    result: Dict[str, Dict[str, Dict[str, Dict[Any, float]]]] = {}
    for nt, type_dict in node_cat_keys.items():
        result.setdefault(nt, {})
        for obj_type, attrs in type_dict.items():
            result[nt].setdefault(obj_type, {})
            for attr, cats in attrs.items():
                if isinstance(cats, list):
                    result[nt][obj_type][attr] = {cat: 0.0 for cat in cats}
                else:
                    result[nt][obj_type][attr] = cats
    return result


def aggregate_dataset_category_importances(
    examples: Iterable[
        Tuple[
            Explanation | HeteroExplanation,
            Optional[HeteroData],
            Dict[str, List[str]],
            Optional[Dict[str, List[str]]],
        ]
    ],
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]],
    metadata: Metadata = None,
    average: bool = True,
) -> Dict[str, Dict[str, Dict[str, Dict[Any, float]]]]:
    """Aggregate category importance scores across multiple explanations.

    Each tuple in ``examples`` should contain:
      - explanation
      - hetero_data (or None for heterogeneous explanations)
      - feat_label_dict
      - node_label_dict (or None if not needed)

    The returned structure preserves the full ``node_cat_keys`` schema and
    aggregates importance scores by category across all examples. If
    ``average`` is ``True``, the result is averaged by example count.
    """
    aggregated = _initialize_category_importances(node_cat_keys)
    count = 0

    for explanation, hetero_data, feat_label_dict, _ in examples:
        importances = get_dataset_category_importances(
            explanation=explanation,
            feat_label_dict=feat_label_dict,
            node_cat_keys=node_cat_keys,
            metadata=metadata,
            hetero_data=hetero_data,
        )

        for nt, obj_types in importances.items():
            for obj_type, attrs in obj_types.items():
                for attr_name, cats in attrs.items():
                    for cat_name, imp in cats.items():
                        aggregated[nt][obj_type][attr_name][cat_name] += imp

        count += 1

    if average and count > 0:
        for obj_types in aggregated.values():
            for attrs in obj_types.values():
                for cats in attrs.values():
                    for cat_name in cats:
                        cats[cat_name] /= float(count)

    return aggregated


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
    include_attributes: List[str] = []

    discretized_attributes = {}
    for attr, spec in attribute_spec_dict.items():
        include_attributes.append(attr)

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
                include_attributes=include_attributes,
                discretized_attributes=discretized_attributes,
            )
        )

    return actions


def build_event_substitution_actions(
    target_nodes: Iterable[Tuple[Any, Any]],
    ocel_nodes: Iterable[Tuple[Any, Any]],
    graph: Graph = None,
    check: Callable[[Dict, Dict], bool] = lambda a, b: True,
    attribute_spec_dict: Dict[str, Any] = None,
) -> List[EventNodeSubstitution]:
    """Construct EventNodeSubstitution actions from target graph nodes.

    - target_nodes: iterable of (node_id, node_data) where node_data has 'attr'
    - ocel_nodes: iterable of (node_id, node_data) from the OCEL graph
    - check: custom filter (node_attr, subst_attr) -> bool
    - attribute_spec_dict: optional dictionary for discretization information
    """
    actions: List[EventNodeSubstitution] = []
    include_attributes: List[str] = []

    discretized_attributes = {}
    for attr, spec in (attribute_spec_dict or {}).items():
        include_attributes.append(attr)

        if (
            isinstance(spec, (list, tuple))
            and len(spec) == 2
            and all(isinstance(x, (int, float)) for x in spec)
        ):
            discretized_attributes[attr] = spec

    ocel_list = list(ocel_nodes)
    target_map = dict(target_nodes)

    def _event_objects(n_id):
        if graph is None:
            return None
        try:
            return {
                obj_id
                for _, obj_id, eattr in graph.out_edges(n_id, data="attr")
                if eattr.get("type") == "E2O"
            }
        except Exception:
            return None

    for node_id, node_data in target_map.items():
        attr = _extract_attr(node_data)
        if attr.get("type", "") != "EVENT":
            continue

        base_object_set = _event_objects(node_id)
        substitution_events: List[Tuple[Any, Any]] = []
        for subst_id, subst_data in ocel_list:
            subst_attr = _extract_attr(subst_data)
            if subst_id == node_id:
                continue
            if subst_attr.get("type", "") != "EVENT":
                continue
            if not check(attr, subst_attr):
                continue

            if base_object_set is not None:
                subst_object_set = _event_objects(subst_id)
                if subst_object_set is None or subst_object_set != base_object_set:
                    continue

            substitution_events.append((subst_id, subst_data))

        actions.append(
            EventNodeSubstitution(
                event_id=node_id,
                event_data=node_data,
                substitution_events=substitution_events,
                include_attributes=include_attributes,
                discretized_attributes=discretized_attributes,
            )
        )

    return actions


def construct_object_base_data(
    object_types: List[str],
    metadata: Any,
    object_type_column: str = "ocel:type",
    random_state: int = None,
) -> Dict[str, Dict[str, Any]]:
    """Construct base data per object type from metadata node keys."""
    rnd = random.Random(random_state)
    base_data: Dict[str, Dict[str, Any]] = {}

    # metadata expected to have node_num_keys and node_cat_keys
    node_num_keys = getattr(metadata, "node_num_keys", {})
    node_cat_keys = getattr(metadata, "node_cat_keys", {})

    for object_type in object_types:
        obj_data: Dict[str, Any] = {
            "type": "OBJECT",
            object_type_column: object_type,
        }

        num_keys_for_type = (
            node_num_keys.get("OBJECT", {}).get(object_type, {})
            or node_num_keys.get("OBJECT", {}).get("OBJECT", {})
            or {}
        )
        for attr, rng in num_keys_for_type.items():
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                continue
            vmin, vmax = float(rng[0]), float(rng[1])
            if vmin == vmax:
                obj_data[attr] = vmin
            else:
                obj_data[attr] = rnd.uniform(vmin, vmax)

        cat_keys_for_type = (
            node_cat_keys.get("OBJECT", {}).get(object_type, {})
            or node_cat_keys.get("OBJECT", {}).get("OBJECT", {})
            or {}
        )
        for attr, values in cat_keys_for_type.items():
            if not values:
                continue
            obj_data[attr] = rnd.choice(values)

        base_data[object_type] = obj_data

    return base_data


def build_event_insertion_actions(
    target_event_nodes: Iterable[Tuple[Any, Any]],
    event_activities: List[str],
    event_activity_column: str = "ocel:activity",
    base_event_data: Dict[str, Any] = None,
    object_ids: List[str] = None,
) -> List["EventNodeInsertion"]:
    """Construct EventNodeInsertion actions for target event nodes.

    - event_activities: list of activity labels to generate one event option each.
    - event_activity_column: name of the activity column to set in each event option.
    - base_event_data: optional base event attributes to merge into each generated option.
    """
    actions: List[EventNodeInsertion] = []
    object_ids = object_ids or []
    base_event_data = deepcopy(base_event_data or {})

    event_data_options = []
    for activity in event_activities:
        event_attr = deepcopy(base_event_data)
        event_attr["type"] = event_attr.get("type", "EVENT")
        event_attr[event_activity_column] = activity
        # Use nested attr wrapper for consistency with node_data format
        event_data_options.append({"attr": event_attr})

    for node_id, node_data in target_event_nodes:
        attr = _extract_attr(node_data)
        if attr.get("type", "") != "EVENT":
            continue
        actions.append(
            EventNodeInsertion(
                event_id=node_id,
                event_data_options=deepcopy(event_data_options),
                object_ids=object_ids,
            )
        )

    return actions


def build_object_insertion_actions(
    target_event_nodes: Iterable[Tuple[Any, Any]],
    object_types: List[str],
    object_type_column: str = "ocel:type",
    base_object_data: Optional[Dict[str, Dict[str, Any]]] = None,
    metadata: Any = None,
    random_state: int = None,
) -> List["ObjectNodeInsertion"]:
    """Construct ObjectNodeInsertion actions for target event nodes.

    - object_types: list of object type values, each becomes one insertion option.
    - object_type_column: name of type column in object attributes.
    - base_object_data: for each object type optional base object attributes to merge into each option.
    - metadata: optional object with node_num_keys/node_cat_keys to generate default values.
    - random_state: seed for randomized numeric/categorical value selection.
    """
    actions: List[ObjectNodeInsertion] = []

    if base_object_data is None and metadata is not None:
        base_object_data = construct_object_base_data(
            object_types=object_types,
            metadata=metadata,
            object_type_column=object_type_column,
            random_state=random_state,
        )

    base_object_data = deepcopy(base_object_data or {})

    object_data_options = []
    for object_type in object_types:
        object_attr = deepcopy(base_object_data.get(object_type, {}))
        object_attr["type"] = object_attr.get("type", "OBJECT")
        object_attr[object_type_column] = object_type
        object_data_options.append({"attr": object_attr})

    for node_id, node_data in target_event_nodes:
        attr = _extract_attr(node_data)
        if attr.get("type", "") != "EVENT":
            continue
        actions.append(
            ObjectNodeInsertion(
                event_id=node_id,
                object_data_options=deepcopy(object_data_options),
            )
        )

    return actions


def build_event_move_actions(
    target_event_nodes: Iterable[Tuple[Any, Any]],
    candidate_event_nodes: Iterable[Tuple[Any, Any]],
    check: Callable[[Dict, Dict], bool] = lambda a, b: True,
) -> List[EventNodeMove]:
    """Construct EventNodeMove actions from target event nodes.

    - target_event_nodes: iterable of (node_id, node_data) for events to move.
    - candidate_event_nodes: iterable of (node_id, node_data) for potential target events.
    - check: callable(target_attr, candidate_attr) -> bool to filter valid target events.
    """
    actions: List[EventNodeMove] = []

    candidate_list = list(candidate_event_nodes)
    target_map = dict(target_event_nodes)

    for node_id, node_data in target_map.items():
        attr = _extract_attr(node_data)
        if attr.get("type", "") != "EVENT":
            continue

        target_events = []
        for candidate_id, candidate_data in candidate_list:
            if candidate_id == node_id:
                continue
            candidate_attr = _extract_attr(candidate_data)
            if candidate_attr.get("type", "") != "EVENT":
                continue
            if not check(attr, candidate_attr):
                continue
            target_events.append(candidate_id)

        actions.append(EventNodeMove(event_id=node_id, target_events=target_events))

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
    category_importances: Optional[
        Dict[str, Dict[str, Dict[str, Dict[Any, float]]]]
    ] = None,
    category_importance_threshold: float = 0.0,
    category_top_k: Optional[int] = None,
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
                # Filter categories by importance if provided
                if category_importances is not None:
                    obj_type_key = (
                        "EVENT"
                        if node_type == "EVENT"
                        else attr.get(object_type_column, "")
                    )
                    imp_dict = (
                        category_importances.get(node_type, {})
                        .get(obj_type_key, {})
                        .get(attr_name, {})
                    )
                    category_values = [
                        v
                        for v in sorted(
                            category_values,
                            key=lambda v: imp_dict.get(v, 0.0),
                            reverse=True,
                        )
                        if imp_dict.get(v, 0.0) > category_importance_threshold
                    ]

                if category_top_k is not None:
                    category_values = category_values[:category_top_k]
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
    "get_dataset_category_importances",
    "aggregate_dataset_category_importances",
    "build_object_substitution_actions",
    "build_event_substitution_actions",
    "build_event_insertion_actions",
    "build_object_insertion_actions",
    "build_event_move_actions",
    "build_node_deletion_actions",
    "build_node_attribute_actions",
    "construct_attribute_spec_dict",
]
