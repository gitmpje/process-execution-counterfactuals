from typing import Dict

from torch import tensor, zeros

from tree_search.feature_selection import (
    get_nodes_by_importance,
    get_feature_labels_by_importance,
)


class DummyExplanation:
    """Minimal stand-in for torch_geometric.explain.HeteroExplanation."""

    def __init__(self, data: Dict[str, dict]):
        # data maps node_type -> dict with potentially 'node_mask'
        self._data = data
        self.node_types = list(data.keys())

    def __getitem__(self, key):
        return self._data.get(key)

    def get_node_store(self, node_type):
        # Return a dict-like object with key 'x' as required by the utility.
        mask = self._data.get(node_type, {})
        mask_tensor = mask.get("node_mask")
        if mask_tensor is None:
            num_feats = 0
        else:
            if mask_tensor.dim() == 1:
                num_feats = mask_tensor.size(0)
            else:
                num_feats = mask_tensor.size(-1)
        return {"x": zeros((1, num_feats))}


# ---------------------------------------------------------------------------
# tests for get_nodes_by_importance
# ---------------------------------------------------------------------------

def test_get_nodes_by_importance_basic():
    mask = tensor([0.1, 0.5, 0.2])
    expl = DummyExplanation({"evt": {"node_mask": mask}})
    labels = {"evt": ["a", "b", "c"]}
    out = get_nodes_by_importance(expl, labels)
    # should be sorted descending by importance
    assert [d["label"] for d in out] == ["b", "c", "a"]
    # top_k limit
    assert len(get_nodes_by_importance(expl, labels, top_k=2)) == 2


def test_get_nodes_by_importance_multi_feature():
    mask = tensor([[0.2, 0.4], [0.1, 0.3]])
    expl = DummyExplanation({"n1": {"node_mask": mask}})
    res = get_nodes_by_importance(expl, {})
    assert len(res) == 2


# ---------------------------------------------------------------------------
# tests for get_feature_labels_by_importance
# ---------------------------------------------------------------------------

def test_get_feature_labels_simple():
    # two nodes, three features
    mask = tensor([[1.0, 0.0, 2.0], [0.5, 1.5, 0.0]])
    expl = DummyExplanation({"type1": {"node_mask": mask}})
    labels = {"type1": ["f1", "f2", "f3"]}
    out = get_feature_labels_by_importance(expl, labels)
    assert "type1" in out
    assert out["type1"][0]["feature"] == "f3"


def test_get_feature_labels_per_category_and_topk():
    mask = tensor([[1.0, 2.0], [3.0, 4.0]])
    expl = DummyExplanation({"t": {"node_mask": mask}})
    feat_labels = {"t": ["foo[bar]", "foo[baz]"]}
    out = get_feature_labels_by_importance(
        expl, feat_labels, feature_per_category=True, top_k=1
    )
    assert out["t"][0]["feature"] == "foo"
    assert len(out["t"]) == 1
