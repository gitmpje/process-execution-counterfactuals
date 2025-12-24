import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import multiprocessing as mp

from copy import deepcopy
from functools import partial
from itertools import combinations, product
from numpy import inf
from typing import Any, Dict, Iterable, List, Tuple


from .process_execution import ProcessExecution


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

    def __repr__(self) -> str:
        return f"{self.node_id} - {self.attribute_name} {self.value_range}"

    def action_space(
        self, current_change: int | float | complex
    ) -> Iterable[int | float | complex]:
        """
        Generate possible values for the node attribute feature.
        Args:
            current_change (int | float | complex): The current change value for the attribute.
        Yields:
            Iterable[int | float | complex]: Possible values for the attribute.
        """
        for v in self.value_range:
            if v >= current_change:
                yield v

    def apply_change(self, p: ProcessExecution, delta_value: Any) -> ProcessExecution:
        """
        Apply the attribute value change to the process execution.
        Args:
            p (ProcessExecution): The process execution to modify.
            delta_value (Any): The value to add to the current attribute value.
        Returns:
            ProcessExecution: The modified process execution.
        """
        p.nodes()[self.node_id]["attr"][self.attribute_name] += delta_value
        return p


class ObjectSubstitutions(Feature):
    """
    Feature representing possible object node substitutions in a process execution.
    Attributes:
        substitution_options (Optional[List[List[Tuple[Tuple[str, dict], Tuple[str, dict]]]]]):
            An iterable of lists of substitution options, where each substitution option is a tuple
            containing the original node (ID and attributes) and the substitute node (ID and attributes).
        allowed_substitutions (Optional[List[Tuple[Tuple[str, dict], Tuple[str, dict]]]]):
            A list of allowed substitutions for the object nodes.
    """

    def __init__(
        self,
        *args,
        substitution_options: List[
            List[Tuple[Tuple[str, dict], Tuple[str, dict]]]
        ] = None,
        allowed_substitutions: List[Tuple[Tuple[str, dict], Tuple[str, dict]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.substitution_options = substitution_options
        self.allowed_substitutions = (
            allowed_substitutions if allowed_substitutions else []
        )

    def __repr__(self) -> str:
        if self.substitution_options:
            return f"{len(self.substitution_options)} substitution options"
        else:
            return f"{len(self.allowed_substitutions)} allowed substitutions"

    def action_space(
        self, current_change
    ) -> Iterable[List[Tuple[Tuple[str, dict], Tuple[str, dict]]]]:
        """
        Generate possible substitution options for the object node feature.
        Args:
            current_change: The current substitution option.
        Yields:
            Iterable[List[Tuple[Tuple[str, dict], Tuple[str, dict]]]]: Possible substitution options.
        """

        if self.substitution_options:
            for option in self.substitution_options:
                if len(option) >= len(current_change):
                    yield option
        else:
            for r in range(len(self.allowed_substitutions) + 1):
                for c in combinations(self.allowed_substitutions, r):
                    for p in product(*c):
                        if len(p) >= len(current_change):
                            yield p

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
        object_substitution (Dict[ObjectSubstitutions, List[Tuple[Tuple[str, dict], Tuple[str, dict]]]): Dictionary mapping object substitutions options to the selected substitutions.
        relation_deletion (List[Tuple[str, str]]): List of relation tuples to remove.
        relation_insertion (List[Tuple[str, str]]): List of relation tuples to add.
        node_attributes_modification (Dict[NodeAttributeNumeric, Any]): Dictionary mapping node attribute features to their change value.
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

    def __repr__(self):
        return f"""Action<{id(self)}>
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
        Calculate the objective value of the action, defined as the total number changes.
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

        # Select the next feature to explore
        selected_feature = self.select_feature(available_features)
        if not selected_feature:
            return

        # Explore all possible values for the selected feature
        for value in selected_feature.action_space(
            action.get_change_value(selected_feature)
        ):
            action_prime = deepcopy(action)

            # Prune actions that exceed max changes
            if action_prime.action_size() > self.max_changes:
                return

            # Set the new change value for the selected feature
            action_prime.set_change_value(selected_feature, value)

            yield from self._branch_and_bound_task(
                action_prime,
                deepcopy(available_features),
            )
            yield action_prime

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
            logger.error("Error in worker: %s", e)

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
