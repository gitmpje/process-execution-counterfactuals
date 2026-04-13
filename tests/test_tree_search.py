"""Unit tests for :mod:`tree_search.tree_search` and related helpers.

The tests are intentionally simple - the data and expected results are defined
inline using fixtures so that adding additional scenarios is straight‑forward.
"""

from copy import deepcopy

import pytest

from tree_search.tree_search import (
    has_value,
    TreeSearchCounterFactual,
    TreeSearchCounterFactualParallel,
)
from tree_search.action_set import ActionSet
from tree_search.action import (
    NodeAttributeNumeric,
    EventNodeDeletion,
    EventNodeSubstitution,
    EventNodeInsertion,
    ObjectNodeInsertion,
)
from process_execution.process_execution import ProcessExecution


class DummyAction:
    """Minimal action stub used only for testing ``maximum_number_of_actions``."""

    def __init__(self, size: int):
        self._size = size

    def action_space_size(self):
        return self._size


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def generate_simple_process(initial_x: int = 0) -> ProcessExecution:
    p = ProcessExecution()
    p.add_node("n1", attr={"x": initial_x})
    return p


# top-level outcome functions so that they are picklable by multiprocessing


def outcome_ge_1(p: ProcessExecution) -> bool:  # noqa: D401
    """True when the attribute ``x`` is >= 1."""
    return p.nodes()["n1"]["attr"]["x"] >= 1


def outcome_ge_minus1(p: ProcessExecution) -> bool:  # noqa: D401
    """True when the attribute ``x`` is >= -1."""
    return p.nodes()["n1"]["attr"]["x"] >= -1


def simple_outcome_threshold(threshold: int):
    """Factory used for sequential tests (not used in parallel runs)."""
    return outcome_ge_1 if threshold == 1 else outcome_ge_minus1


# ----------------------------------------------------------------------------
# tests for simple utility functions
# ----------------------------------------------------------------------------
def test_has_value_true():
    """Generator with a value should return ``True``."""
    assert has_value((i for i in [1, 2, 3]))


def test_has_value_false():
    assert not has_value((i for i in []))


def test_maximum_number_of_actions():
    t = TreeSearchCounterFactual(
        process_outcome=lambda x: False, counterfactual_label=False
    )
    feats = [DummyAction(3), DummyAction(4)]
    # product of sizes (3 * 4)
    assert t.maximum_number_of_actions(feats) == 12


# ----------------------------------------------------------------------------
# tests for ``ActionSet`` behaviour
# ----------------------------------------------------------------------------
def test_action_equality_repr_and_copy(numeric_action, categorical_action):
    a1 = ActionSet()
    a2 = deepcopy(a1)
    assert a1 == a2
    a1.set_change_value(numeric_action, 1)
    assert a1 != a2
    r = repr(a1)
    assert "node_attributes_modification" in r


def test_action_size_objective_and_apply(simple_process_execution, numeric_action):
    a = ActionSet()
    # no changes initially
    assert a.action_size() == 0

    # set a change value and apply it to the process
    a.set_change_value(numeric_action, 1)
    assert a.get_change_value(numeric_action) == 1
    assert a.action_size() == numeric_action.change_size(1)

    p = simple_process_execution
    p_after, rec = a.apply_changes(p)
    assert p_after.nodes()["n1"]["attr"]["x"] == 1

    # undo should restore original value
    a.undo_changes(p_after, rec)
    assert p_after.nodes()["n1"]["attr"]["x"] == 0


def test_action_set_conflict_prevents_substitution_on_deleted_node():
    a = ActionSet()
    delete_action = EventNodeDeletion(deletion_options=[["n1"]])
    a.set_change_value(delete_action, ["n1"])

    subst_action = EventNodeSubstitution(
        event_id="n1",
        event_data={"type": "EVENT"},
        substitution_events=[("n2", {"type": "EVENT"})],
    )

    assert not a.is_change_allowed(subst_action, ("n2", {"type": "EVENT"}))


def test_action_set_conflict_prevents_event_insertion_on_deleted_event():
    a = ActionSet()
    delete_action = EventNodeDeletion(deletion_options=[["e1"]])
    a.set_change_value(delete_action, ["e1"])

    insertion = EventNodeInsertion(
        event_id="e1",
        event_data_options=[{"type": "EVENT", "ocel:activity": "new"}],
        object_ids=[],
    )

    assert not a.is_change_allowed(insertion, {"type": "EVENT", "ocel:activity": "new"})


def test_action_set_conflict_prevents_object_insertion_on_deleted_event():
    a = ActionSet()
    delete_action = EventNodeDeletion(deletion_options=[["e2"]])
    a.set_change_value(delete_action, ["e2"])

    insertion = ObjectNodeInsertion(
        event_id="e2",
        object_data_options=[{"type": "OBJECT", "ocel:type": "X"}],
    )

    assert not a.is_change_allowed(insertion, {"type": "OBJECT", "ocel:type": "X"})


def test_event_insertion_apply_undo():
    p = generate_simple_process(0)
    p.add_node("e1", attr={"type": "EVENT"})
    p.add_node("o1", attr={"type": "OBJECT"})

    event_data = {"type": "EVENT", "ocel:activity": "new"}
    insertion = EventNodeInsertion(
        event_id="e1",
        event_data_options=[event_data],
        object_ids=["o1"],
    )

    a = ActionSet()
    a.set_change_value(insertion, event_data)

    p_after, rec = a.apply_changes(p)
    # event inserted as a new node
    inserted_nodes = [n for n in p_after.nodes() if n.startswith("insert_event_")]
    assert len(inserted_nodes) == 1
    inserted = inserted_nodes[0]
    assert p_after.has_edge("e1", inserted)
    assert p_after.has_edge(inserted, "o1")

    a.undo_changes(p_after, rec)
    assert not p_after.has_node(inserted)


def test_event_insertion_multiple_data_options():
    insertion = EventNodeInsertion(
        event_id="e1",
        event_data_options=[
            {"attr": {"type": "EVENT", "ocel:activity": "x"}},
            {"attr": {"type": "EVENT", "ocel:activity": "y"}},
        ],
        object_ids=[],
    )
    assert insertion.action_space_size() == 2
    assert {"type": "EVENT", "ocel:activity": "x"} in list(insertion.action_space())
    assert {"type": "EVENT", "ocel:activity": "y"} in list(insertion.action_space())


def test_object_insertion_apply_undo():
    p = generate_simple_process(0)
    p.add_node("e1", attr={"type": "EVENT"})

    object_data = {"type": "OBJECT", "ocel:type": "new"}
    insertion = ObjectNodeInsertion(
        event_id="e1",
        object_data_options=[object_data],
    )

    a = ActionSet()
    a.set_change_value(insertion, object_data)

    p_after, rec = a.apply_changes(p)
    inserted_nodes = [n for n in p_after.nodes() if n.startswith("insert_object_")]
    assert len(inserted_nodes) == 1
    inserted = inserted_nodes[0]
    assert p_after.has_edge("e1", inserted)

    a.undo_changes(p_after, rec)
    assert not p_after.has_node(inserted)


def test_object_insertion_multiple_data_options():
    p = generate_simple_process(0)
    p.add_node("e1", attr={"type": "EVENT"})

    insertion = ObjectNodeInsertion(
        event_id="e1",
        object_data_options=[
            {"attr": {"type": "OBJECT", "ocel:type": "A"}},
            {"attr": {"type": "OBJECT", "ocel:type": "B"}},
        ],
    )

    assert insertion.action_space_size() == 2
    options = list(insertion.action_space())
    assert {"type": "OBJECT", "ocel:type": "A"} in options
    assert {"type": "OBJECT", "ocel:type": "B"} in options

    a = ActionSet()
    a.set_change_value(insertion, {"type": "OBJECT", "ocel:type": "B"})

    p_after, rec = a.apply_changes(p)
    inserted_nodes = [n for n in p_after.nodes() if n.startswith("insert_object_")]
    assert len(inserted_nodes) == 1
    inserted = inserted_nodes[0]
    assert p_after.nodes()[inserted]["attr"]["ocel:type"] == "B"

    a.undo_changes(p_after, rec)
    assert not p_after.has_node(inserted)


def test_action_apply_multiple_action_types(
    simple_process_execution, numeric_action, categorical_action
):
    a = ActionSet()
    a.set_change_value(numeric_action, -1)
    a.set_change_value(categorical_action, "blue")
    # categorical changes are now counted in ``action_size``
    assert a.action_size() == numeric_action.change_size(
        -1
    ) + categorical_action.change_size("blue")
    p = simple_process_execution
    p_after, rec = a.apply_changes(p)
    assert p_after.nodes()["n1"]["attr"]["x"] == -1
    assert p_after.nodes()["n1"]["attr"]["color"] == "blue"
    # undo restores both attributes
    a.undo_changes(p_after, rec)
    assert p_after.nodes()["n1"]["attr"]["x"] == 0
    assert (
        p_after.nodes()["n1"]["attr"]["color"] == "red"
    )  # original color from fixture


# ----------------------------------------------------------------------------
# tests for action enumerations (numeric & categorical)
# ----------------------------------------------------------------------------
def test_numeric_action_action_space(numeric_action):
    # starting from zero, we should be able to move +-1 within +/-2 bound
    vals = list(
        numeric_action.action_space(current_change_value=0, max_change_size_delta=1)
    )
    assert 1 in vals or -1 in vals
    # values should not exceed defined bounds
    assert all(
        numeric_action.value_min - numeric_action.value_original
        <= v
        <= numeric_action.value_max - numeric_action.value_original
        for v in vals
    )


def test_categorical_action_action_space(categorical_action):
    vals = list(
        categorical_action.action_space(
            current_change_value=None, max_change_size_delta=1
        )
    )
    assert set(vals) <= set(categorical_action.category_values)
    assert categorical_action.value_original not in vals


# ----------------------------------------------------------------------------
# higher‑level tree search tests
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "threshold,expected_change",
    [
        (1, 1),  # need to increase x to meet threshold
        (-1, -1),  # we expect at least one action achieving the negative change
    ],
)
def test_search_layer_finds_counterfactual(threshold, expected_change):
    """Search should return at least one action that achieves the desired threshold."""
    action = NodeAttributeNumeric(
        node_id="n1",
        attribute_name="x",
        value_original=0,
        value_step=1,
        value_min=-2,
        value_max=2,
    )
    p = generate_simple_process(0)
    tree = TreeSearchCounterFactual(
        process_outcome=simple_outcome_threshold(threshold),
        counterfactual_label=True,
        step_change_size=1,
        max_change_size=2,
        log_level=50,  # suppress logging during tests
    )
    selected_action_sets = tree.search_layer([(ActionSet(), [action])], p)
    assert selected_action_sets, "no actions returned"
    got_values = {
        action_set.get_change_value(action) for action_set in selected_action_sets
    }
    assert expected_change in got_values
    # all returned actions should satisfy the outcome after being applied
    for a in selected_action_sets:
        p_copy = deepcopy(p)
        p_after, _ = a.apply_changes(p_copy)
        assert tree.process_outcome(p_after)


def outcome_ge_3(p: ProcessExecution) -> bool:
    """Helper used only in this test: ``x >= 3``."""
    return p.nodes()["n1"]["attr"]["x"] >= 3


def test_search_layer_no_solution():
    # threshold unreachable within max_change_size (requires x>=3)
    feat = NodeAttributeNumeric(
        node_id="n1",
        attribute_name="x",
        value_original=0,
        value_step=1,
        value_min=-2,
        value_max=2,
    )
    p = generate_simple_process(0)
    tree = TreeSearchCounterFactual(
        process_outcome=outcome_ge_3,
        counterfactual_label=True,
        step_change_size=1,
        max_change_size=2,
        log_level=50,
    )
    actions = tree.search_layer([(ActionSet(), [feat])], p)
    assert actions == []


def test_parallel_search_matches_sequential():
    feat = NodeAttributeNumeric(
        node_id="n1",
        attribute_name="x",
        value_original=0,
        value_step=1,
        value_min=-2,
        value_max=2,
    )
    p = generate_simple_process(0)
    seq = TreeSearchCounterFactual(
        process_outcome=simple_outcome_threshold(1),
        counterfactual_label=True,
        step_change_size=1,
        max_change_size=2,
        log_level=50,
    )
    par = TreeSearchCounterFactualParallel(
        num_workers=1,
        process_outcome=simple_outcome_threshold(1),
        counterfactual_label=True,
        step_change_size=1,
        max_change_size=2,
        log_level=50,
    )
    seq_actions = seq.search_layer([(ActionSet(), [feat])], deepcopy(p))
    # use a picklable outcome for the parallel instance
    par_actions = par.search_layer([(ActionSet(), [feat])], deepcopy(p))
    assert seq_actions == par_actions


def test_logger_configuration():
    """Logger should have two handlers (stream and file) after instantiation."""
    tree = TreeSearchCounterFactual(
        process_outcome=lambda x: True, counterfactual_label=True
    )
    handlers = [h.__class__.__name__ for h in tree.logger.handlers]
    assert "StreamHandler" in handlers
    assert "RotatingFileHandler" in handlers
