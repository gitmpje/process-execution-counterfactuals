import pandas as pd
import pytest

from tree_search.feature_helpers import (
    _extract_attr,
    build_object_substitution_features,
    build_node_deletion_features,
    _parse_value_spec,
    build_node_attribute_features,
    construct_attribute_spec_dict,
)
from tree_search.feature import (
    EventNodeDeletion,
    ObjectNodeDeletion,
    NodeAttributeNumeric,
    NodeAttributeCategorical,
    ObjectNodeSubstitution,
)
from process_execution.process_execution import ProcessExecution


# ---------------------------------------------------------------------------
# _extract_attr
# ---------------------------------------------------------------------------

def test_extract_attr_from_dict():
    assert _extract_attr({"attr": {"x": 1}}) == {"x": 1}
    assert _extract_attr({"x": 2}) == {"x": 2}


# ---------------------------------------------------------------------------
# build_node_deletion_features
# ---------------------------------------------------------------------------

def test_build_node_deletion_features_event_object():
    nodes = [
        ("e1", {"attr": {"type": "EVENT"}}),
        ("o1", {"attr": {"type": "OBJECT", "ocel:type": "Foo"}}),
        ("o2", {"attr": {"type": "OBJECT", "ocel:type": "View"}}),
    ]
    # viewpoint filters out o2
    feats = build_node_deletion_features(
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
# build_node_attribute_features
# ---------------------------------------------------------------------------

def test_build_node_attribute_features_numeric_and_cat():
    nodes = [
        ("n1", {"attr": {"type": "EVENT", "a": 5, "b": "red"}}),
        ("n2", {"attr": {"type": "OBJECT", "a": 1}}),
    ]
    spec = {"a": (0, 10, 5), "b": ["red", "blue"]}
    feats = build_node_attribute_features(nodes, spec, node_type="EVENT")
    assert any(isinstance(f, NodeAttributeNumeric) for f in feats)
    assert any(isinstance(f, NodeAttributeCategorical) for f in feats)
    # verify values preserved
    num = [f for f in feats if isinstance(f, NodeAttributeNumeric)][0]
    assert num.node_id == "n1"
    cat = [f for f in feats if isinstance(f, NodeAttributeCategorical)][0]
    assert cat.category_values == ["red", "blue"]


# ---------------------------------------------------------------------------
# build_object_substitution_features
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

    features = build_object_substitution_features(
        target_nodes,
        ocel_nodes,
        p,
        object_type_column="ocel:type",
        check=lambda a, b: True,
        attribute_spec_dict={},
    )
    assert len(features) == 1
    feat = features[0]
    assert isinstance(feat, ObjectNodeSubstitution)
    # substitution_objects should include o2 only
    assert any(sub[0] == "o2" for sub in feat.substitution_objects)
    assert feat.event_ids == ["e1"]


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
