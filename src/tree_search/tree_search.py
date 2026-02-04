import logging

from copy import deepcopy
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Tuple

from process_execution.process_execution import ProcessExecution

from tree_search.feature import (
    EventNodeDeletion,
    Feature,
    NodeAttributeNumeric,
    ObjectNodeSubstitution,
)


class Action:
    """
    Class representing a set of changes (actions) to be applied to a process execution.
    Attributes:
        event_deletion (List[str]): List of event IDs to remove.
        object_substitution (Dict[ObjectSubstitutions, List[Tuple[Tuple[str, dict], Tuple[str, dict]]]): Dictionary mapping object substitutions options to the selected substitutions.
        node_attributes_modification (Dict[NodeAttributeNumeric, Any]): Dictionary mapping node attribute features to their change value.
    """

    def __init__(
        self,
        event_deletion: Dict[EventNodeDeletion, List[str]] = None,
        object_substitution: Dict[
            ObjectNodeSubstitution, Tuple[str, dict] | None
        ] = None,
        node_attributes_modification: Dict[NodeAttributeNumeric, Any] = None,
    ):
        self.event_deletion = event_deletion if event_deletion else {}
        self.object_substitution = object_substitution if object_substitution else {}
        self.node_attributes_modification = (
            node_attributes_modification if node_attributes_modification else {}
        )

    def __repr__(self):
        return f"""Action<{id(self)}>
    event_deletion: {self.event_deletion}
    object_substitution: {self.object_substitution}
    node_attributes_modification: {self.node_attributes_modification}"""

    def get_change_value(self, feature: Feature) -> Any:
        """
        Get the current change value for a given feature.
        Args:
            feature (Feature): The feature for which to get the change value.
        Returns:
            Any: The current change value for the feature.
        """
        if isinstance(feature, EventNodeDeletion):
            return self.event_deletion.get(feature, [])
        elif isinstance(feature, NodeAttributeNumeric):
            return self.node_attributes_modification.get(feature, 0)
        elif isinstance(feature, ObjectNodeSubstitution):
            return self.object_substitution.get(feature)
        else:
            NotImplementedError(f"Feature of type {type(feature)} is not supported")

    def set_change_value(self, feature: Feature, value: Any):
        """
        Set the change value for a given feature.
        Args:
            feature (Feature): The feature for which to set the change value.
            value (Any): The new value to set for the feature.
        """
        if isinstance(feature, EventNodeDeletion):
            self.event_deletion[feature] = value
        elif isinstance(feature, NodeAttributeNumeric):
            self.node_attributes_modification[feature] = value
        elif isinstance(feature, ObjectNodeSubstitution):
            self.object_substitution[feature] = value
        else:
            NotImplementedError(f"Feature of type {type(feature)} is not supported")

    def action_size(self) -> int:
        """
        Calculate the total number of changes in the action.
        Returns:
            int: The total number of changes.
        """
        substitutions = [
            (k.object_id, v[0]) for k, v in self.object_substitution.items() if v
        ]

        return (
            len(self.event_deletion)
            + len(
                [
                    obj_id
                    for obj_id, subst_obj_id in substitutions
                    if obj_id != subst_obj_id
                ]
            )
            + len([k for k, v in self.node_attributes_modification.items() if v != 0])
        )

    def objective_value(self) -> int:
        """
        Calculate the objective value of the action, defined as the total number changes.
        Returns:
            int: The objective value of the action.
        """
        substitutions = [
            (k.object_id, v[0]) for k, v in self.object_substitution.items() if v
        ]

        return (
            len(self.event_deletion)
            + len(
                [
                    obj_id
                    for obj_id, subst_obj_id in substitutions
                    if obj_id != subst_obj_id
                ]
            )
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

        for substitution, value in self.object_substitution.items():
            if value:
                substitution.apply_change(p, value)

        for feature, value in self.node_attributes_modification.items():
            feature.apply_change(p, value)

        return p


class TreeSearchCounterFactual:
    """
    Class implementing the tree search algorithm to find counterfactual actions.
    Attributes:
        process_outcome (callable): Function to determine the outcome of a process execution.
        counterfactual_label (bool): Desired outcome label for the counterfactual.
        step_change_size (int): Step size in the change size for each search layer.
        max_change_size (int): Maximum change size to consider.
        log_level (int): Logging level for the process (default: logging.INFO).
    """

    def __init__(
        self,
        process_outcome: callable,
        counterfactual_label: bool,
        step_change_size: int = 1,
        max_change_size: int = 10,
        log_level=logging.INFO,
    ):
        self.process_outcome = process_outcome
        self.counterfactual_label = counterfactual_label
        self.step_change_size = step_change_size
        self.max_change_size = max_change_size
        self.log_level = log_level

        self.count = 0

        self.logger = self._configure_logger()

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

    def maximum_number_of_actions(
        self,
        available_features: List[Feature],
    ) -> int:
        """
        Calculate maximum number of actions.
        """
        n_actions = 1
        for feature in available_features:
            n_actions *= max(feature.action_space_size(), 1)

        # TODO: incorporate limit on max_changes
        return n_actions

    def evaluate_action(self, action, process_execution):
        # Check process outcome after applying changes
        process_execution_c = action.apply_changes(deepcopy(process_execution))
        outcome_c = self.process_outcome(process_execution_c)
        self.logger.debug("Process outcome: %s", outcome_c)

        return outcome_c == self.counterfactual_label

    def explore_features(
        self,
        action: Action,
        features: List[Feature],
        process_execution: ProcessExecution,
        change_size=1,
    ):
        change_lower = change_size - self.step_change_size
        change_upper = change_size

        explored_actions = []
        selected_actions = []
        for feature in features:
            self.logger.debug("%s", feature)

            other_features = [f for f in features if f != feature]
            for change_value in feature.action_space(change_lower, change_upper):
                action_prime = deepcopy(action)
                action_prime.set_change_value(feature, change_value)

                eval_result = self.evaluate_action(action_prime, process_execution)
                if eval_result:
                    self.logger.info("Found counterfactual: %s", action_prime)
                    selected_actions.append(deepcopy(action_prime))

                explored_actions.append((action_prime, other_features))

                self.count += 1
                if self.count % 50 == 0:
                    self.logger.info(f"Evaluated {self.count} actions")

        return explored_actions, selected_actions

    def search_layer(
        self,
        actions_features: List[Tuple[Action, List[Feature]]],
        process_execution: ProcessExecution,
        change_size=1,
    ):
        """
        Recursively enumerate possible actions to find counterfactuals.
        Args:
            actions_features (List[Tuple[Action, List[Feature]]]): Actions from preceding search step with list of features that can still be modified.
            process_execution (ProcessExecution): The original process execution.
            change_size: Change size to search actions for.
        """
        self.logger.debug("change_size=%s\n%s", change_size, actions_features)

        # Terminate when max change size is reached
        if change_size >= self.max_change_size:
            self.logger.info(
                f"No valid counterfactual action found with size smaller than {self.max_change_size}"
            )
            return []

        explored_actions = []
        selected_actions = []
        for action, features in actions_features:
            explored_actions_a, selected_actions_a = self.explore_features(action, features, process_execution, change_size)
            explored_actions.extend(explored_actions_a)
            selected_actions.extend(selected_actions_a)

        # Return when valid counterfactual actions are found
        if selected_actions:
            return selected_actions

        # If there are no actions explored, increase change size
        if not explored_actions:
            explored_actions = actions_features

        self.search_layer(
            actions_features,
            process_execution,
            change_size=change_size + self.step_change_size,
        )

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)

        # Configure logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
        )

        # Handler to output warnings to the console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.WARNING)
        logger.addHandler(stream_handler)

        # Handler to output logs to a file
        file_handler = RotatingFileHandler(f"{__name__}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)
        logger.addHandler(file_handler)

        return logger
