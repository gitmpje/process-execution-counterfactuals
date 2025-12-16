import logging

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

from process_execution import ProcessExecution

logger = logging.getLogger(__name__)


class Feature:
    """
    Base class for features that can be modified in a process execution.
    """

    def __init__(
        self,
    ): ...


class NodeAttributeNumeric(Feature):
    """
    Feature representing a node attribute that can be modified.

    Attributes:
        node_id (str): The ID of the node whose attribute is being modified.
        attribute_name (str): The name of the attribute to be modified.
        value_range (Iterable): The range of possible values for the attribute.
        value_default (int|float|complex): The default value of the attribute.
    """

    def __init__(
        self,
        *args,
        node_id: str,
        attribute_name: str,
        value_range: Iterable[int | float | complex],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.node_id = node_id
        self.attribute_name = attribute_name
        self.value_range = value_range

    def __str__(self) -> str:
        return f"{self.node_id} - {self.attribute_name} {self.value_range}"

    def action_space(self, current_change: int | float | complex) -> Iterable:
        """
        Generate possible values for the node attribute feature.
        """
        for v in self.value_range:
            if v >= current_change:
                yield v

    def apply_change(self, p: ProcessExecution, delta_value: Any) -> ProcessExecution:
        p.nodes()[self.node_id]["attr"][self.attribute_name] += delta_value
        return p


class ObjectSubstitutions(Feature):
    """ """

    def __init__(
        self,
        *args,
        substitution_options: Iterable[List[Tuple[Tuple[str, dict], Tuple[str, dict]]]],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.substitution_options = substitution_options

    def __str__(self) -> str:
        return f"{len(self.substitution_options)} substitution options"

    def action_space(
        self, current_change
    ) -> Iterable[List[Tuple[Tuple[str, dict], Tuple[str, dict]]]]:
        """ """
        for v in self.substitution_options:
            if len(v) >= len(current_change):
                yield v

    def apply_change(
        self,
        p: ProcessExecution,
        substitutions: List[Tuple[Tuple[str, dict], Tuple[str, dict]]],
    ) -> ProcessExecution:
        for substitution in substitutions:
            if not substitution:
                return p

            # Construct edges to substitution node
            node_id = substitution[0][0]
            subst_node_id = substitution[1][0]
            subst_node_attr = substitution[1][1]
            edges = []
            for u, v, d in p.in_edges(node_id, data=True):
                edges.append((u, subst_node_id, d))
            for u, v, d in p.out_edges(node_id, data=True):
                edges.append((subst_node_id, v, d))

            # Remove node
            p.remove_node(node_id)

            # Add node with attributes and edges
            p.add_node(subst_node_id, **subst_node_attr)
            p.add_edges_from(edges)

        return p


class Action:
    """
    Class representing a set of changes (actions) to be applied to a process execution.

    Attributes:
        event_deletion (List[str]): List of event IDs to remove.
        event_insertion (List[str]): List of event IDs to add.
        object_substitution (List[Tuple[Tuple[str, dict], Tuple[str, dict]]): List of object nodes to substitute
        relation_deletion (List[Tuple[str, str]]): List of relation tuples to remove.
        relation_insertion (List[Tuple[str, str]]): List of relation tuples to add.
        node_attributes_modification (Dict[NodeAttributeNumeric, Any]): Dictionary mapping node attribute features to their new values.
    """

    def __init__(
        self,
        event_deletion: List[str] = None,
        event_insertion: List[str] = None,
        object_substitution: Dict[
            ObjectSubstitutions, List[Tuple[Tuple[str, dict], Tuple[str, dict]]]
        ] = None,
        relation_deletion: List[Tuple[str, str]] = None,
        relation_insertion: List[Tuple[str, str]] = None,
        node_attributes_modification: Dict[NodeAttributeNumeric, Any] = None,
    ):
        self.event_deletion = event_deletion if event_deletion else []
        self.event_insertion = event_insertion if event_insertion else []
        self.object_substitution = object_substitution if object_substitution else {}
        self.relation_deletion = relation_deletion if relation_deletion else []
        self.relation_insertion = relation_insertion if relation_insertion else []
        self.node_attributes_modification = (
            node_attributes_modification if node_attributes_modification else {}
        )

    def __str__(self):
        return f"""Action<{id(self)}>
    event_deletion: {len(self.event_deletion)}
    event_insertion: {len(self.event_insertion)}
    object_substitution: {len(self.object_substitution)}
    relation_deletion: {len(self.relation_deletion)}
    relation_insertion: {len(self.relation_insertion)}
    node_attributes_modification: {len(self.node_attributes_modification)}"""

    def get_change_value(self, feature: Feature) -> Any:
        """
        Get the current change value for a given feature.
        Args:
            feature (Feature): The feature for which to get the change value.
        Returns:
            Any: The current change value for the feature.
        """
        if isinstance(feature, NodeAttributeNumeric):
            return self.node_attributes_modification.get(feature, 0)
        if isinstance(feature, ObjectSubstitutions):
            return self.object_substitution.get(feature, [])

    def set_change_value(self, feature: Feature, value: Any):
        """
        Set the change value for a given feature.
        Args:
            feature (Feature): The feature for which to set the change value.
            value (Any): The new value to set for the feature.
        """
        if isinstance(feature, NodeAttributeNumeric):
            self.node_attributes_modification[feature] = value
        if isinstance(feature, ObjectSubstitutions):
            self.object_substitution[feature] = value
        else:
            NotImplementedError(f"Feature of type {type(feature)} is not supported")

    def action_size(self) -> int:
        """
        Calculate the total number of changes in the action.
        Returns:
            int: The total number of changes.
        """
        return (
            len(self.event_deletion)
            + len(self.event_insertion)
            + len(self.object_substitution)
            + len(self.relation_deletion)
            + len(self.relation_insertion)
            + len(self.node_attributes_modification)
        )

    def objective_value(self) -> int:
        """
        Calculate the objective value of the action, defined as the total number of non-default changes.
        Returns:
            int: The objective value of the action.
        """
        substitutions = [
            subst for v in self.object_substitution.values() for subst in v if subst
        ]

        return (
            len(self.event_deletion)
            + len(self.event_insertion)
            + len([subst for subst in substitutions if subst[0] != subst[1]])
            + len(self.relation_deletion)
            + len(self.relation_insertion)
            + len([k for k, v in self.node_attributes_modification.items() if v != 0])
        )

    def apply_changes(
        self,
        p: ProcessExecution,
    ) -> ProcessExecution:
        """
        Apply the changes defined in the action to a given process execution.
        Args:
            p (ProcessExecution): The process execution to which the changes will be applied.
        Returns:
            ProcessExecution: The modified process execution after applying the changes.
        """
        p.remove_nodes_from(self.event_deletion)
        p.add_nodes_from(self.event_insertion)
        p.remove_edges_from(self.relation_deletion)
        p.add_edges_from(self.relation_insertion)

        for substitution, value in self.object_substitution.items():
            substitution.apply_change(p, value)

        for feature, value in self.node_attributes_modification.items():
            feature.apply_change(p, value)

        return p


class BranchAndBoundCounterFactual:
    def __init__(
        self,
        process_outcome: callable,
        max_changes: int,
        counterfactual_label: bool,
    ):
        self.max_changes = max_changes
        self.counterfactual_label = counterfactual_label
        self.process_outcome = process_outcome

    def select_feature(self, available_features: List[Feature]) -> Feature | None:
        """
        Select the next feature to consider from the available features.
        Args:
            available_features (List[Feature]): List of available features to select from.
        Returns:
            Feature: The selected feature, or None if no features are available.
        """
        try:
            return available_features.pop(0)
        except IndexError:
            return None

    def enumerate(
        self,
        action: Action,
        available_features: List[Feature],
        fixed_features: List[Feature],
        process_execution: ProcessExecution,
        selected_actions: List[Action],
    ):
        """
        Recursively enumerate possible actions to find counterfactuals.
        Args:
            action (Action): The current action being evaluated.
            available_features (List[Feature]): List of features that can still be modified.
            fixed_features (List[Feature]): List of features that have already been fixed.
            process_execution (ProcessExecution): The original process execution.
            selected_actions (List[Action]): List to store found counterfactual actions.
        """
        logger.debug("-" * 20)
        logger.debug(action)

        if (action.action_size() > self.max_changes) or any(
            action.objective_value() >= selected_action.objective_value()
            for selected_action in selected_actions
        ):
            return

        process_execution_c = action.apply_changes(deepcopy(process_execution))
        outcome_c = self.process_outcome(process_execution_c)
        logger.debug("Counterfactual outcome: %s", outcome_c)
        if outcome_c == self.counterfactual_label:
            selected_actions.append(deepcopy(action))
            return

        selected_feature = self.select_feature(available_features)
        if not selected_feature:
            return

        fixed_features.append(selected_feature)

        for value in selected_feature.action_space(
            action.get_change_value(selected_feature)
        ):
            action_prime = deepcopy(action)
            logger.debug("%s: %s", selected_feature, value)

            action_prime.set_change_value(selected_feature, value)
            self.enumerate(
                action_prime,
                available_features,
                fixed_features,
                process_execution,
                selected_actions,
            )
        logger.debug("-" * 50)
