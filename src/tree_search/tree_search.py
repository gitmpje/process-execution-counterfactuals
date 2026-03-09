import logging
import multiprocessing as mp

from copy import copy
from logging.handlers import QueueListener, RotatingFileHandler
from typing import List, Tuple

from process_execution.process_execution import ProcessExecution

from tree_search.action import Action
from tree_search.feature import Feature


log_queue = mp.Queue()


def has_value(gen):
    try:
        next(gen)
        return True
    except StopIteration:
        return False


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
        log_file: str = f"{__name__}.log",
    ):
        self.process_outcome = process_outcome
        self.counterfactual_label = counterfactual_label
        self.step_change_size = step_change_size
        self.max_change_size = max_change_size
        self.log_level = log_level
        self.log_file = log_file

        self.logger = self._configure_logger()

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

    def explore_features(
        self,
        action: Action,
        features: List[Feature],
        process_execution: ProcessExecution,
        change_size=1,
    ):
        def evaluate_action(action, process_execution):
            # Check process outcome after applying changes
            process_execution_c, recorded_changes = action.apply_changes(
                process_execution
            )
            outcome_c = self.process_outcome(process_execution_c)
            action.undo_changes(process_execution_c, recorded_changes)

            return outcome_c == self.counterfactual_label

        max_change_size_delta = change_size - action.action_size()

        next_actions = []
        selected_actions = []
        features_no_change = []
        explored_features = []
        for feature in features:
            explored_features.append(feature)
            current_change_value = action.get_change_value(feature)

            # Only explore features that have not been explored in this layer yet
            # as the order of features does not matter
            next_features = {f for f in features if f not in explored_features}
            explored_feature = False
            for change_value in feature.action_space(
                current_change_value, max_change_size_delta
            ):
                explored_feature = True
                action_prime = copy(action)
                action_prime.set_change_value(feature, change_value)

                eval_result = evaluate_action(action_prime, process_execution)
                if eval_result:
                    selected_actions.append(copy(action_prime))

                # If feature actions space is not empty after selected change value
                if has_value(
                    feature.action_space(
                        change_value, self.max_change_size - action.action_size()
                    )
                ):
                    next_features.add(feature)

                next_actions.append((action_prime, next_features))

            if not explored_feature:
                features_no_change.append(feature)

        # Take action to next layer with 'unexplored' features
        if features_no_change:
            next_actions.append((action, features_no_change))

        return next_actions, selected_actions

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

        # Terminate when max change size is reached
        if change_size > self.max_change_size:
            self.logger.info(
                f"No valid counterfactual action found with size smaller than {self.max_change_size}"
            )
            return []

        self.logger.info(
            "Expanding %s actions for change_size %s",
            len(actions_features),
            change_size,
        )

        next_actions_features = []
        selected_actions = []
        for action, features in actions_features:
            explored, selected = self.explore_features(
                action, features, process_execution, change_size
            )

            # Collect distinct next actions
            self.logger.debug("Explored %s actions", len(explored))
            for next_action in explored:
                if next_action not in next_actions_features:
                    next_actions_features.append(next_action)

            # Collect distinct selected actions
            self.logger.info("Found counterfactual: %s", selected)
            if selected:
                selected_actions.append(selected)
                break

        if self.log_level == logging.DEBUG:
            with open(f"{self.log_file}-next_actions_features-{change_size}", "w") as f:
                f.write(
                    "\n".join(
                        [f"{item[0]}\n\t{item[1]}" for item in next_actions_features]
                    )
                )

        # Return when valid counterfactual actions are found
        if selected_actions:
            return selected_actions

        # Start next search layer
        return self.search_layer(
            next_actions_features,
            process_execution,
            change_size=change_size + self.step_change_size,
        )

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)

        # Clear existing handlers to avoid duplicate logs if multiple instances are created
        logger.handlers = []

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
        file_handler = RotatingFileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)
        logger.addHandler(file_handler)

        return logger


class TreeSearchCounterFactualParallel(TreeSearchCounterFactual):
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
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_workers = num_workers

        # Listener for collecting logs from parallel processes
        log_listener = self._configure_log_listener()
        log_listener.start()

    def search_layer(
        self,
        actions_features: List[Tuple[Action, List[Feature]]],
        process_execution: ProcessExecution,
        change_size=1,
    ):
        """
        Recursively enumerate possible actions to find counterfactuals.
        Explore features on a layer in parallel.
        Args:
            actions_features (List[Tuple[Action, List[Feature]]]): Actions from preceding search step with list of features that can still be modified.
            process_execution (ProcessExecution): The original process execution.
            change_size: Change size to search actions for.
        """

        # Terminate when max change size is reached
        if change_size > self.max_change_size:
            self.logger.info(
                f"No valid counterfactual action found with size smaller than {self.max_change_size}"
            )
            return []

        self.logger.info(
            "Expanding %s actions for change_size %s",
            len(actions_features),
            change_size,
        )

        next_actions_features = []
        selected_actions = []
        evaluate_args = [
            (action, features, process_execution, change_size)
            for action, features in actions_features
        ]

        with mp.Pool(self.num_workers) as pool:
            for explored, selected in pool.starmap(
                self.explore_features, evaluate_args
            ):
                # Collect distinct next actions
                for next_action in explored:
                    if next_action not in next_actions_features:
                        next_actions_features.append(next_action)

                # Collect distinct selected actions
                for selected_action in selected:
                    if selected_action not in selected_actions:
                        selected_actions.append(selected_action)

        if self.log_level == logging.DEBUG:
            with open(f"{self.log_file}-explored_actions-{change_size}", "w") as f:
                f.write("\n".join([f"{item[0]}" for item in next_actions_features]))

        # Return when valid counterfactual actions are found
        if selected_actions:
            return selected_actions

        # Start next search layer
        return self.search_layer(
            next_actions_features,
            process_execution,
            change_size=change_size + self.step_change_size,
        )

    def _configure_log_listener(self) -> QueueListener:
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
