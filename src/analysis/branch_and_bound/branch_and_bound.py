import logging
import multiprocessing as mp

from copy import deepcopy
from functools import partial
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from numpy import inf
from typing import Any, Dict, Iterable, List, Tuple

from ..process_execution import ProcessExecution

from .feature import (
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


class BranchAndBoundCounterFactual:
    """
    Class implementing the Branch and Bound algorithm to find counterfactual actions.
    Attributes:
        max_changes (int): Maximum number of allowed changes in the action.
        counterfactual_label (bool): Desired outcome label for the counterfactual.
        process_outcome (callable): Function to determine the outcome of a process execution.
        num_workers (int): Number of parallel worker processes to use (default: 1).
        log_level (int): Logging level for the process (default: logging.INFO).
    """

    def __init__(
        self,
        process_outcome: callable,
        max_changes: int,
        counterfactual_label: bool,
        num_workers: int = 1,
        log_level=logging.INFO,
    ):
        self.max_changes = max_changes
        self.counterfactual_label = counterfactual_label
        self.process_outcome = process_outcome
        self.num_workers = num_workers
        self.log_level = log_level

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
            n_actions *= feature.action_space_size()

        # TODO: incorporate limit on max_changes
        return n_actions

    def find_counterfactuals(
        self,
        process_execution: ProcessExecution,
        available_features: List[Feature],
    ) -> List[Action]:
        """
        Find counterfactual actions using multiprocessing.
        Args:
            process_execution (ProcessExecution): The original process execution.
            available_features (List[Feature]): List of available features.
        Returns:
            List[Action]: List of found counterfactual actions.
        """
        # Configure logging
        log_queue = mp.Queue()

        logger = logging.getLogger("find_counterfactuals")
        logger.setLevel(self.log_level)
        logger.addHandler(QueueHandler(log_queue))

        log_listener = self._configure_log_listener(log_queue)
        log_listener.start()

        action = Action()
        min_objective_value = mp.Value("f", inf)
        returned_actions = []
        with mp.Pool(
            processes=self.num_workers,
            initializer=self._setup_worker,
            initargs=(log_queue, min_objective_value),
        ) as pool:
            try:
                branch_and_bound_worker = partial(
                    self._branch_and_bound_worker,
                    process_execution=process_execution,
                )
                count = 0
                for result in pool.imap(
                    branch_and_bound_worker,
                    self._branch_and_bound_task(action, available_features),
                ):
                    count += 1
                    if count % 100 == 0:
                        logger.info("Processed %d actions", count)
                    if result:
                        logger.info("Found counterfactual action: %s", result)
                        objective_val = result.objective_value()
                        logger.info("Objective value = %s", objective_val)
                        if objective_val < min_objective_value.value:
                            min_objective_value.value = objective_val
                        returned_actions.append(result)
                        pool.terminate()
                        break
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Terminating workers...")
                pool.terminate()
            finally:
                pool.close()
                pool.join()
                log_listener.stop()

        # Collect results
        selected_actions = []
        min_value = inf
        for returned_action in returned_actions:
            objective_value = returned_action.objective_value()
            if objective_value <= min_value:
                min_value = objective_value
                selected_actions.append(returned_action)

        return selected_actions

    def _branch_and_bound_task(
        self,
        action: Action,
        available_features: List[Feature],
    ) -> Iterable[Action]:
        """
        Recursive generator for branch and bound tasks.
        Args:
            action (Action): The current action being explored.
            available_features (List[Feature]): List of available features to consider.
        Yields:
            Action: Generated actions to be evaluated.
        """

        # Prune actions that exceed max changes
        if action.action_size() + 1 > self.max_changes:
            return

        # Select the next feature to explore
        selected_feature = self.select_feature(available_features)
        if not selected_feature:
            return

        # Explore all possible values for the selected feature
        for value in selected_feature.action_space(
            action.get_change_value(selected_feature)
        ):
            action_prime = deepcopy(action)

            # Set the new change value for the selected feature
            action_prime.set_change_value(selected_feature, value)

            yield action_prime
            yield from self._branch_and_bound_task(
                action_prime,
                available_features.copy(),
            )

    def _branch_and_bound_worker(
        self,
        action: Action,
        process_execution: ProcessExecution,
    ) -> Action | None:
        """
        Worker process for parallel branch and bound.
        Args:
            action (Action): The action to evaluate.
            process_execution (ProcessExecution): The original process execution.
        Returns:
            Action | None: The action if it meets the counterfactual condition, else None.
        """
        logger = logging.getLogger(__name__)

        try:
            # Prune actions that exceed max changes or current best objective value
            if action.objective_value() >= min_objective_value.value:
                return

            process_execution_c = action.apply_changes(deepcopy(process_execution))
            outcome_c = self.process_outcome(process_execution_c)
            logger.debug("Action: %s => Outcome: %s", str(action), outcome_c)

            # Check if counterfactual condition is met
            if outcome_c == self.counterfactual_label:
                return action
        except Exception as e:
            logger.error("Error in worker", e)
            raise e

        return

    def _configure_log_listener(self, log_queue: mp.Queue) -> QueueListener:
        """
        Configure a logging listener to handle log records from worker processes.
        Args:
            log_queue (mp.Queue): The queue to receive log records from worker processes.
        Returns:
            QueueListener: Configured logging listener.
        """
        # Configure logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
        )

        # Handler to output warnings to the console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.WARNING)

        # Handler to output logs to a file
        file_handler = RotatingFileHandler(f"{__name__}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)

        return QueueListener(
            log_queue, stream_handler, file_handler, respect_handler_level=True
        )

    # Function to set up a logger with a QueueHandler
    def _setup_worker(self, queue: mp.Queue, min_obj_value):
        """
        Setup logging and shared variable for worker processes.
        Args:
            queue (mp.Queue): The logging queue.
            min_obj_value: Shared minimum objective value.
        Returns:
            logging.Logger: Configured logger for the worker.
        """
        global min_objective_value
        min_objective_value = min_obj_value

        global log_queue
        log_queue = queue

        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)

        #  QueueHandler to send log records to a logging queue
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(queue_handler)

        return logger
