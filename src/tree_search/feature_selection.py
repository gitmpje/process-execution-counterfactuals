from torch_geometric.explain import HeteroExplanation
from torch import Tensor
from typing import Dict, List


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
    feature_per_category: bool = False,
    top_k=None,
):
    """Return per-node-type feature labels ordered by importance.

    For each node type, features are aggregated across nodes (mean) and
    returned as a list of dicts with keys: `feature`, `importance`.
    """
    out = {}
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

        if feature_per_category:
            # group feature importances by base category extracted from label
            # e.g. 'ocel:activity[Register Customer Order]' -> 'ocel:activity'
            category_vals: Dict[str, List[float]] = {}
            for i, v in enumerate(per_feat.tolist()):
                fname = feat_labels[i] if i < len(feat_labels) else f"feat_{i}"
                base = fname.split("[")[0] if "[" in fname else fname
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
