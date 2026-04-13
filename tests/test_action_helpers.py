import pandas as pd
import pytest

from torch import tensor, zeros
from torch_geometric.explain import HeteroExplanation
from typing import Dict

from tree_search.action_helpers import (
    _extract_attr,
    build_event_insertion_actions,
    build_event_substitution_actions,
    build_object_insertion_actions,
    build_object_substitution_actions,
    build_node_deletion_actions,
    _parse_value_spec,
    build_node_attribute_actions,
    construct_attribute_spec_dict,
    get_nodes_by_importance,
    get_feature_labels_by_importance,
)
from tree_search.action import (
    EventNodeDeletion,
    EventNodeSubstitution,
    ObjectNodeDeletion,
    NodeAttributeNumeric,
    NodeAttributeCategorical,
    ObjectNodeSubstitution,
)
from process_execution.process_execution import ProcessExecution


class DummyExplanation(HeteroExplanation):
    """Minimal stand-in for torch_geometric.explain.HeteroExplanation."""

    def __init__(self, data: Dict[str, dict]):
        super().__init__()
        # data maps node_type -> dict with potentially 'node_mask'
        for node_type, node_data in data.items():
            mask_tensor = node_data.get("node_mask")
            if mask_tensor is not None:
                self[node_type].node_mask = mask_tensor
                num_feats = 1 if mask_tensor.dim() == 1 else mask_tensor.size(-1)
                self[node_type].x = zeros((1, num_feats))


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
    node_cat_keys = {"OBJECT": {"t": {"foo": ["bar", "baz"]}}}
    out = get_feature_labels_by_importance(
        expl, feat_labels, node_cat_keys, one_hot_encoding=True, top_k=1
    )
    print(out)
    assert out["t"][0]["feature"] == "foo"
    assert len(out["t"]) == 1


# ---------------------------------------------------------------------------
# _extract_attr
# ---------------------------------------------------------------------------


def test_extract_attr_from_dict():
    assert _extract_attr({"attr": {"x": 1}}) == {"x": 1}
    assert _extract_attr({"x": 2}) == {"x": 2}


# ---------------------------------------------------------------------------
# build_node_deletion_actions
# ---------------------------------------------------------------------------


def test_build_node_deletion_actions_event_object():
    nodes = [
        ("e1", {"attr": {"type": "EVENT"}}),
        ("o1", {"attr": {"type": "OBJECT", "ocel:type": "Foo"}}),
        ("o2", {"attr": {"type": "OBJECT", "ocel:type": "View"}}),
    ]
    # viewpoint filters out o2
    feats = build_node_deletion_actions(
        nodes, object_type_column="ocel:type", viewpoint="View"
    )
    assert any(isinstance(f, EventNodeDeletion) for f in feats)
    assert any(isinstance(f, ObjectNodeDeletion) for f in feats)
    assert all(len(f.deletion_options) == 1 for f in feats)


# ---------------------------------------------------------------------------
# _parse_value_spec
# ---------------------------------------------------------------------------


def test_parse_value_spec_tuple_and_range():
    assert _parse_value_spec((0, 10, 2)) == (0, 10, 2)
    assert _parse_value_spec(range(0, 5, 2)) == (0, 5, 2)
    with pytest.raises(ValueError):
        _parse_value_spec("not a spec")


# ---------------------------------------------------------------------------
# build_node_attribute_actions
# ---------------------------------------------------------------------------


def test_build_node_attribute_actions_numeric_and_cat():
    nodes = [
        ("n1", {"attr": {"type": "EVENT", "a": 5, "b": "red"}}),
        ("n2", {"attr": {"type": "OBJECT", "a": 1}}),
    ]
    spec = {"a": (0, 10, 5), "b": ["red", "blue"]}
    feats = build_node_attribute_actions(nodes, spec, node_type="EVENT")
    assert any(isinstance(f, NodeAttributeNumeric) for f in feats)
    assert any(isinstance(f, NodeAttributeCategorical) for f in feats)
    # verify values preserved
    num = [f for f in feats if isinstance(f, NodeAttributeNumeric)][0]
    assert num.node_id == "n1"
    cat = [f for f in feats if isinstance(f, NodeAttributeCategorical)][0]
    assert cat.category_values == ["red", "blue"]


# ---------------------------------------------------------------------------
# build_object_substitution_actions
# ---------------------------------------------------------------------------


def test_build_object_substitution_simple_graph():
    # target graph with one object and one event connected by E2O
    p = ProcessExecution()
    p.add_node("o1", attr={"type": "OBJECT", "ocel:type": "T"})
    p.add_node("e1", attr={"type": "EVENT"})
    p.add_edge("e1", "o1", attr={"type": "E2O"})

    target_nodes = [("o1", {"attr": {"type": "OBJECT", "ocel:type": "T"}})]
    ocel_nodes = [
        ("o1", {"attr": {"type": "OBJECT", "ocel:type": "T"}}),
        ("o2", {"attr": {"type": "OBJECT", "ocel:type": "T"}}),
    ]

    actions = build_object_substitution_actions(
        target_nodes,
        ocel_nodes,
        p,
        object_type_column="ocel:type",
        check=lambda a, b: True,
        attribute_spec_dict={},
    )
    assert len(actions) == 1
    feat = actions[0]
    assert isinstance(feat, ObjectNodeSubstitution)
    # substitution_objects should include o2 only
    assert any(sub[0] == "o2" for sub in feat.substitution_objects)
    assert feat.event_ids == ["e1"]


def test_build_event_substitution_simple_graph():
    p = ProcessExecution()
    p.add_node("e1", attr={"type": "EVENT", "ocel:activity": "X"})
    p.add_node("e2", attr={"type": "EVENT", "ocel:activity": "X"})
    p.add_node("e3", attr={"type": "EVENT", "ocel:activity": "Y"})

    target_nodes = [("e1", {"attr": {"type": "EVENT", "ocel:activity": "X"}})]
    ocel_nodes = [
        ("e1", {"attr": {"type": "EVENT", "ocel:activity": "X"}}),
        ("e2", {"attr": {"type": "EVENT", "ocel:activity": "X"}}),
        ("e3", {"attr": {"type": "EVENT", "ocel:activity": "Y"}}),
    ]

    actions = build_event_substitution_actions(
        target_nodes,
        ocel_nodes,
        p,
        check=lambda a, b: True,
        attribute_spec_dict={},
    )

    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, EventNodeSubstitution)
    assert action.event_id == "e1"
    assert any(sub_id == "e2" for sub_id, _ in action.substitution_events)


def test_build_event_substitution_with_object_relations():
    p = ProcessExecution()
    p.add_node("e1", attr={"type": "EVENT", "ocel:activity": "X"})
    p.add_node("e2", attr={"type": "EVENT", "ocel:activity": "X"})
    p.add_node("e3", attr={"type": "EVENT", "ocel:activity": "X"})
    p.add_node("o1", attr={"type": "OBJECT", "ocel:type": "A"})
    p.add_node("o2", attr={"type": "OBJECT", "ocel:type": "A"})

    p.add_edge("e1", "o1", attr={"type": "E2O"})
    p.add_edge("e2", "o1", attr={"type": "E2O"})
    p.add_edge("e3", "o2", attr={"type": "E2O"})

    target_nodes = [("e1", {"attr": {"type": "EVENT", "ocel:activity": "X"}})]
    ocel_nodes = [
        ("e1", {"attr": {"type": "EVENT", "ocel:activity": "X"}}),
        ("e2", {"attr": {"type": "EVENT", "ocel:activity": "X"}}),
        ("e3", {"attr": {"type": "EVENT", "ocel:activity": "X"}}),
    ]

    actions = build_event_substitution_actions(
        target_nodes,
        ocel_nodes,
        p,
        check=lambda a, b: True,
        attribute_spec_dict={},
    )

    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, EventNodeSubstitution)
    assert action.event_id == "e1"
    assert any(sub_id == "e2" for sub_id, _ in action.substitution_events)
    assert all(sub_id != "e3" for sub_id, _ in action.substitution_events)


def test_build_event_insertion_actions():
    actions = build_event_insertion_actions(
        [("e1", {"attr": {"type": "EVENT"}})],
        event_activities=["A", "B"],
        event_activity_column="ocel:activity",
        base_event_data={"type": "EVENT"},
        object_ids=["o1"],
    )
    assert len(actions) == 1
    action = actions[0]
    assert action.event_id == "e1"
    assert action.object_ids == ["o1"]
    assert len(action.event_data_options) == 2
    assert action.event_data_options[0]["attr"]["ocel:activity"] == "A"
    assert action.event_data_options[1]["attr"]["ocel:activity"] == "B"


def test_build_object_insertion_actions():
    actions = build_object_insertion_actions(
        [("e1", {"attr": {"type": "EVENT"}})],
        object_types=["A", "B"],
        object_type_column="ocel:type",
        base_object_data={
            "A": {"type": "OBJECT", "ocel:color": "red"},
            "B": {"type": "OBJECT", "ocel:color": "blue"},
        },
    )
    assert len(actions) == 1
    action = actions[0]
    assert action.event_id == "e1"
    assert len(action.object_data_options) == 2
    assert action.object_data_options[0]["attr"]["ocel:type"] == "A"
    assert action.object_data_options[0]["attr"]["ocel:color"] == "red"
    assert action.object_data_options[1]["attr"]["ocel:type"] == "B"
    assert action.object_data_options[1]["attr"]["ocel:color"] == "blue"


def test_build_object_insertion_actions_with_metadata():
    class MetadataObj:
        pass

    metadata = MetadataObj()
    metadata.node_num_keys = {
        "OBJECT": {
            "OBJECT": {"weight": (1.0, 3.0)},
            "A": {"size": (10.0, 20.0)},
        }
    }
    metadata.node_cat_keys = {
        "OBJECT": {
            "OBJECT": {"shape": ["circle", "square"]},
            "A": {"material": ["metal", "plastic"]},
        }
    }

    actions = build_object_insertion_actions(
        [("e1", {"attr": {"type": "EVENT"}})],
        object_types=["A"],
        object_type_column="ocel:type",
        metadata=metadata,
        random_state=42,
    )

    action = actions[0]
    assert len(action.object_data_options) == 1
    option = action.object_data_options[0]["attr"]
    assert option["ocel:type"] == "A"
    assert option["type"] == "OBJECT"
    assert "size" in option
    assert option["material"] in ["metal", "plastic"]


# ---------------------------------------------------------------------------
# construct_attribute_spec_dict
# ---------------------------------------------------------------------------


def make_fake_ocel():
    class FakeOCEL:
        object_type_column = "type"
        event_activity = "act"

        def __init__(self):
            self.objects = pd.DataFrame([{"type": "A", "foo": 1}])
            self.events = pd.DataFrame([{"act": "X", "foo": 5}])

    return FakeOCEL()


def test_construct_attribute_spec_dict():
    ocel = make_fake_ocel()
    node_cat_keys = {"EVENT": {"X": {"foo": [1, 2, 3]}}}
    node_num_keys = {"EVENT": {"X": ["foo"]}}
    spec = construct_attribute_spec_dict(
        ["foo"], ocel, node_cat_keys, node_num_keys, num_bins=2
    )
    assert "foo" in spec
    # numeric range tuple should be present
    assert isinstance(spec["foo"], tuple)
    assert len(spec["foo"]) == 3
