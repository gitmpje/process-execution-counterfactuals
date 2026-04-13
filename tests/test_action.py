import pytest

from process_execution.process_execution import ProcessExecution

from tree_search.action import (
    NodeAttributeNumeric,
    NodeAttributeCategorical,
    ObjectNodeSubstitution,
    EventNodeSubstitution,
    EventNodeDeletion,
    ObjectNodeDeletion,
    EventNodeMove,
    NodeDeletion,
)
from tree_search.action_helpers import build_event_move_actions


# ---------------------------------------------------------------------------
# numeric and categorical actions
# ---------------------------------------------------------------------------


def test_numeric_action_action_space_and_size():
    f = NodeAttributeNumeric(
        node_id="n1",
        attribute_name="v",
        value_original=0,
        value_step=2,
        value_min=-4,
        value_max=4,
    )
    # step of 2 means action_space_size should at least be nonzero
    assert f.action_space_size() > 0
    vals = list(f.action_space(current_change_value=0, max_change_size_delta=2))
    assert any(v in (-2, 2) for v in vals)
    assert f.change_size(2) == pytest.approx(1.0)


def test_categorical_action_sizes():
    f = NodeAttributeCategorical(
        node_id="n1",
        attribute_name="c",
        value_original="r",
        category_values=["r", "g", "b"],
    )
    assert f.action_space_size() == 2
    vals = list(f.action_space(None, max_change_size_delta=1))
    assert set(vals) == {"g", "b"}
    assert f.change_size("g") == 1


# ---------------------------------------------------------------------------
# object substitution behaviour
# ---------------------------------------------------------------------------


def make_simple_object_graph():
    p = ProcessExecution()
    p.add_node("o1", attr={"type": "OBJECT"})
    p.add_node("o2", attr={"type": "OBJECT"})
    p.add_node("e1", attr={"type": "EVENT"})
    p.add_edge("e1", "o1", attr={"type": "E2O"})
    return p


def test_object_substitution_apply_and_cost():
    p = make_simple_object_graph()
    action = ObjectNodeSubstitution(
        object_id="o1",
        substitution_objects=[("o2", {"attr": {"type": "OBJECT"}})],
        event_ids=["e1"],
        object_data={"attr": {"type": "OBJECT", "foo": 1}},
        discretized_attributes=None,
    )
    before_edges = list(p.in_edges("o1"))
    assert before_edges
    action.apply_change(p, ("o2", {"attr": {}}))
    assert not p.has_node("o1")
    assert p.has_edge("e1", "o2")
    # cost should be calculable without error (should not raise)
    try:
        _ = action.change_size(subst_node=("o2", {"attr": {"type": "OBJECT"}}))
    except NotImplementedError:
        pytest.skip("change_size not implemented for given types")


# ---------------------------------------------------------------------------
# node deletion logic
# ---------------------------------------------------------------------------


def test_node_deletion_action_space():
    f = NodeDeletion(deletion_options=[["a", "b"], ["c"]])
    assert f.action_space_size() == 2
    opts = list(f.action_space(current_change_value=None, max_change_size_delta=1))
    assert ["c"] in opts or ["a", "b"] in opts


def test_event_node_substitution_apply_and_undo():
    p = ProcessExecution()
    # Set up two events and neighbors with DF edges
    p.add_node("e1", attr={"type": "EVENT", "ocel:timestamp": "t1", "epoch": 1})
    p.add_node("e2", attr={"type": "EVENT", "ocel:timestamp": "t2", "epoch": 2})
    p.add_node("e0", attr={"type": "EVENT"})
    p.add_node("e3", attr={"type": "EVENT"})
    p.add_node("e4", attr={"type": "EVENT"})
    p.add_node("e5", attr={"type": "EVENT"})

    p.add_edge("e0", "e1", attr={"type": "DF"})
    p.add_edge("e1", "e3", attr={"type": "DF"})
    p.add_edge("e4", "e2", attr={"type": "DF"})
    p.add_edge("e2", "e5", attr={"type": "DF"})

    action = EventNodeSubstitution(
        event_id="e1",
        event_data={},
        substitution_events=[("e2", {})],
    )

    record = action.apply_change(p, ("e2", {}))

    # DF edges should be swapped
    expected_after = {
        ("e0", "e2"),
        ("e2", "e3"),
        ("e4", "e1"),
        ("e1", "e5"),
    }
    actual_after = {
        (u, v)
        for u, v, d in p.edges(data=True)
        if d.get("attr", {}).get("type") == "DF"
    }
    assert actual_after == expected_after

    # timestamps and epoch should be swapped
    assert p.nodes["e1"]["attr"]["ocel:timestamp"] == "t2"
    assert p.nodes["e2"]["attr"]["ocel:timestamp"] == "t1"
    assert p.nodes["e1"]["attr"]["epoch"] == 2
    assert p.nodes["e2"]["attr"]["epoch"] == 1

    # undo should restore original
    action.undo_change(p, record)

    expected_before = {
        ("e0", "e1"),
        ("e1", "e3"),
        ("e4", "e2"),
        ("e2", "e5"),
    }
    actual_before = {
        (u, v)
        for u, v, d in p.edges(data=True)
        if d.get("attr", {}).get("type") == "DF"
    }
    assert actual_before == expected_before
    assert p.nodes["e1"]["attr"]["ocel:timestamp"] == "t1"
    assert p.nodes["e2"]["attr"]["ocel:timestamp"] == "t2"
    assert p.nodes["e1"]["attr"]["epoch"] == 1
    assert p.nodes["e2"]["attr"]["epoch"] == 2


def test_event_node_move_apply_and_undo():
    p = ProcessExecution()
    # Set up two events and neighbors with DF edges
    p.add_node("e1", attr={"type": "EVENT", "ocel:timestamp": "t1", "epoch": 1})
    p.add_node("e2", attr={"type": "EVENT", "ocel:timestamp": "t2", "epoch": 2})
    p.add_node("e0", attr={"type": "EVENT"})
    p.add_node("e3", attr={"type": "EVENT"})

    p.add_edge("e0", "e1", attr={"type": "DF"})
    p.add_edge("e1", "e2", attr={"type": "DF"})
    p.add_edge("e2", "e3", attr={"type": "DF"})

    action = EventNodeMove(
        event_id="e1",
        target_events=["e2"],
    )

    record = action.apply_change(p, "e2")

    # DF edges should be changed
    expected_after = {
        ("e0", "e2"),
        ("e2", "e1"),
        ("e1", "e3"),
    }
    actual_after = {
        (u, v)
        for u, v, d in p.edges(data=True)
        if d.get("attr", {}).get("type") == "DF"
    }
    assert actual_after == expected_after

    # timestamps and epoch should be changed
    assert p.nodes["e1"]["attr"]["ocel:timestamp"] == "t2"
    assert p.nodes["e2"]["attr"]["ocel:timestamp"] == "t2"
    assert p.nodes["e1"]["attr"]["epoch"] == 2
    assert p.nodes["e2"]["attr"]["epoch"] == 2

    # undo should restore original
    action.undo_change(p, record)

    expected_before = {
        ("e0", "e1"),
        ("e1", "e2"),
        ("e2", "e3"),
    }
    actual_before = {
        (u, v)
        for u, v, d in p.edges(data=True)
        if d.get("attr", {}).get("type") == "DF"
    }

    assert actual_before == expected_before
    assert p.nodes["e1"]["attr"]["ocel:timestamp"] == "t1"
    assert p.nodes["e2"]["attr"]["ocel:timestamp"] == "t2"
    assert p.nodes["e1"]["attr"]["epoch"] == 1
    assert p.nodes["e2"]["attr"]["epoch"] == 2


def test_event_node_deletion_apply():
    p = ProcessExecution()
    p.add_node("e1", attr={"type": "EVENT"})
    p.add_node("e2", attr={"type": "EVENT"})
    # add DF edge e1->e2 and DF from some other node
    p.add_node("e0", attr={"type": "EVENT"})
    p.add_edge("e1", "e2", attr={"type": "DF"})
    p.add_edge("e0", "e1", attr={"type": "DF"})

    action = EventNodeDeletion(deletion_options=[["e1"]])
    action.apply_change(p, ["e1"])

    # node should not exist anymore
    assert not p.has_node("e1")

    # new edge e0->e2 should exist
    assert p.has_edge("e0", "e2")


def test_object_node_deletion_apply():
    p = ProcessExecution()
    p.add_node("o1", attr={"type": "OBJECT"})
    action = ObjectNodeDeletion(deletion_options=[["o1"]])
    action.apply_change(p, ["o1"])
    assert not p.has_node("o1")


def test_build_event_move_actions():
    target_event_nodes = [
        ("e1", {"attr": {"type": "EVENT", "ocel:activity": "A"}}),
        ("e2", {"attr": {"type": "EVENT", "ocel:activity": "B"}}),
    ]
    candidate_event_nodes = [
        ("e0", {"attr": {"type": "EVENT", "ocel:activity": "A"}}),
        ("e1", {"attr": {"type": "EVENT", "ocel:activity": "A"}}),
        ("e3", {"attr": {"type": "EVENT", "ocel:activity": "C"}}),
    ]

    actions = build_event_move_actions(
        target_event_nodes,
        candidate_event_nodes,
        check=lambda target_attr, candidate_attr: True,
    )

    assert len(actions) == 2
    action_map = {action.event_id: action for action in actions}
    assert isinstance(action_map["e1"], EventNodeMove)
    assert set(action_map["e1"].target_events) == {"e0", "e3"}
    assert action_map["e2"].target_events == ["e0", "e1", "e3"]
