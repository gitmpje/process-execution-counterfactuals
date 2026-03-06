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
from tree_search.action import Action
from tree_search.feature import NodeAttributeNumeric
from process_execution.process_execution import ProcessExecution


class DummyFeature:
    """Minimal feature stub used only for testing ``maximum_number_of_actions``."""

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
    feats = [DummyFeature(3), DummyFeature(4)]
    # product of sizes (3 * 4)
    assert t.maximum_number_of_actions(feats) == 12


# ----------------------------------------------------------------------------
# tests for ``Action`` behaviour
# ----------------------------------------------------------------------------
def test_action_equality_repr_and_copy(numeric_feature, categorical_feature):
    a1 = Action()
    a2 = deepcopy(a1)
    assert a1 == a2
    a1.set_change_value(numeric_feature, 1)
    assert a1 != a2
    r = repr(a1)
    assert "node_attributes_modification" in r


def test_action_size_objective_and_apply(simple_process_execution, numeric_feature):
    a = Action()
    # no changes initially
    assert a.action_size() == 0
    assert a.objective_value() == 0

    # set a change value and apply it to the process
    a.set_change_value(numeric_feature, 1)
    assert a.get_change_value(numeric_feature) == 1
    assert a.action_size() == numeric_feature.change_size(1)
    # objective counts nonzero modifications
    assert a.objective_value() == 1

    p = simple_process_execution
    a.apply_changes(p)
    assert p.nodes()["n1"]["attr"]["x"] == 1


def test_action_apply_multiple_feature_types(
    simple_process_execution, numeric_feature, categorical_feature
):
    a = Action()
    a.set_change_value(numeric_feature, -1)
    a.set_change_value(categorical_feature, "blue")
    # categorical changes are now counted in ``action_size``
    assert a.action_size() == numeric_feature.change_size(
        -1
    ) + categorical_feature.change_size("blue")
    p = simple_process_execution
    a.apply_changes(p)
    assert p.nodes()["n1"]["attr"]["x"] == -1
    assert p.nodes()["n1"]["attr"]["color"] == "blue"


# ----------------------------------------------------------------------------
# tests for feature enumerations (numeric & categorical)
# ----------------------------------------------------------------------------
def test_numeric_feature_action_space(numeric_feature):
    # starting from zero, we should be able to move +-1 within +/-2 bound
    vals = list(
        numeric_feature.action_space(current_change_value=0, max_change_size_delta=1)
    )
    assert 1 in vals or -1 in vals
    # values should not exceed defined bounds
    assert all(
        numeric_feature.value_min - numeric_feature.value_original
        <= v
        <= numeric_feature.value_max - numeric_feature.value_original
        for v in vals
    )


def test_categorical_feature_action_space(categorical_feature):
    vals = list(
        categorical_feature.action_space(
            current_change_value=None, max_change_size_delta=1
        )
    )
    assert set(vals) <= set(categorical_feature.category_values)
    assert categorical_feature.value_original not in vals


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
        process_outcome=simple_outcome_threshold(threshold),
        counterfactual_label=True,
        step_change_size=1,
        max_change_size=2,
        log_level=50,  # suppress logging during tests
    )
    actions = tree.search_layer([(Action(), [feat])], p)
    assert actions, "no actions returned"
    got_values = {a.get_change_value(feat) for a in actions}
    assert expected_change in got_values
    # all returned actions should satisfy the outcome after being applied
    for a in actions:
        p_copy = deepcopy(p)
        p_after = a.apply_changes(p_copy)
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
    actions = tree.search_layer([(Action(), [feat])], p)
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
    seq_actions = seq.search_layer([(Action(), [feat])], deepcopy(p))
    # use a picklable outcome for the parallel instance
    par_actions = par.search_layer([(Action(), [feat])], deepcopy(p))
    assert seq_actions == par_actions


def test_logger_configuration():
    """Logger should have two handlers (stream and file) after instantiation."""
    tree = TreeSearchCounterFactual(
        process_outcome=lambda x: True, counterfactual_label=True
    )
    handlers = [h.__class__.__name__ for h in tree.logger.handlers]
    assert "StreamHandler" in handlers
    assert "RotatingFileHandler" in handlers
