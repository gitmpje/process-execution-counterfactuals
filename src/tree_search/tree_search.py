import logging
import multiprocessing as mp

from copy import copy
from logging.handlers import QueueListener, RotatingFileHandler
from typing import Dict, List, Tuple

from process_execution.process_execution import ProcessExecution

from tree_search.action_set import ActionSet
from tree_search.action import Action


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

        self.count_explored = 0

        self.logger = self._configure_logger()

    def maximum_number_of_actions(
        self,
        available_actions: List[Action],
    ) -> int:
        """
        Calculate maximum number of actions.
        """
        n_actions = 1
        for action in available_actions:
            n_actions *= max(action.action_space_size(), 1)

        # TODO: incorporate limit on max_changes
        return n_actions

    def explore_actions(
        self,
        action_set: ActionSet,
        actions: List[Action],
        process_execution: ProcessExecution,
        change_size=1,
    ) -> Tuple[List, List]:
        def evaluate_action_set(
            action_set: ActionSet, process_execution: ProcessExecution
        ):
            # Check process outcome after applying changes
            process_execution_c, recorded_changes = action_set.apply_changes(
                process_execution
            )
            outcome_c = self.process_outcome(process_execution_c)
            action_set.undo_changes(process_execution_c, recorded_changes)

            return outcome_c == self.counterfactual_label

        max_change_size_delta = change_size - action_set.action_size()

        explore_next = []
        selected_action_sets = []
        actions_no_change = []
        explored_actions = []
        for action in actions:
            explored_actions.append(action)
            current_change_value = action_set.get_change_value(action)

            # Only explore actions that have not been explored in this layer yet
            # as the order of actions does not matter
            next_actions = {a for a in actions if a not in explored_actions}
            explored_action_set = False
            for change_value in action.action_space(
                current_change_value, max_change_size_delta
            ):
                explored_action_set = True
                action_set_prime = copy(action_set)
                action_set_prime.set_change_value(action, change_value)

                # Skip if this candidate conflicts with existing modifications
                if not action_set_prime.is_change_allowed(action, change_value, process_execution):
                    self.logger.debug("Not allowed: %s", action_set_prime)
                    continue

                eval_result = evaluate_action_set(action_set_prime, process_execution)
                if eval_result:
                    selected_action_sets.append(copy(action_set_prime))

                # If action actions space is not empty after selected change value
                if has_value(
                    action.action_space(
                        change_value, self.max_change_size - action_set.action_size()
                    )
                ):
                    next_actions.add(action)

                explore_next.append((action_set_prime, next_actions))

            if not explored_action_set:
                actions_no_change.append(action)

        # Take action to next layer with 'unexplored' actions
        if actions_no_change:
            explore_next.append((action_set, actions_no_change))

        return explore_next, selected_action_sets

    def search_layer(
        self,
        actions_to_explore: List[Tuple[ActionSet, List[Action]]],
        process_execution: ProcessExecution,
        change_size: int = 1,
    ) -> List | None:
        """
        Recursively enumerate possible actions to find counterfactuals.
        Args:
            actions_to_explore (List[Tuple[ActionSet, List[Action]]]): ActionSets from preceding search step with list of actions that can still be modified.
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
            len(actions_to_explore),
            change_size,
        )

        next_actions_to_explore = []
        selected_action_sets = []
        for action_set, actions in actions_to_explore:
            explored, selected = self.explore_actions(
                action_set, actions, process_execution, change_size
            )
            self.count_explored += len(explored)

            # Collect distinct next actions
            self.logger.debug("Explored %s actions", len(explored))
            for explore_next in explored:
                if explore_next not in next_actions_to_explore:
                    next_actions_to_explore.append(explore_next)

            # Collect distinct selected actions
            if selected:
                self.logger.info("Found counterfactual: %s", selected)
                selected_action_sets.extend(selected)
                break

        if self.log_level == logging.DEBUG:
            with open(
                f"{self.log_file}-next_actions_to_explore-{change_size}", "w"
            ) as f:
                f.write(
                    "\n".join(
                        [f"{item[0]}\n\t{item[1]}" for item in next_actions_to_explore]
                    )
                )

        # Return when valid counterfactual actions are found
        if selected_action_sets:
            return selected_action_sets

        # Start next search layer
        return self.search_layer(
            next_actions_to_explore,
            process_execution,
            change_size=change_size + self.step_change_size,
        )

    def search_depth_first(
        self,
        actions_grouped: Dict[str, List[Action]],
        process_execution: ProcessExecution,
    ) -> List | None:
        for i, actions_group in actions_grouped.items():
            self.logger.info("Searching in group %s", i)
            selected = self.search_layer(
                actions_to_explore=[(ActionSet(), actions_group)],
                process_execution=process_execution,
            )
            if selected:
                return selected

        return []

    def _configure_logger(self) -> logging.Logger:
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
        actions_actions: List[Tuple[ActionSet, List[Action]]],
        process_execution: ProcessExecution,
        change_size=1,
    ) -> List | None:
        """
        Recursively enumerate possible actions to find counterfactuals.
        Explore actions on a layer in parallel.
        Args:
            actions_actions (List[Tuple[ActionSet, List[Action]]]): ActionSets from preceding search step with list of actions that can still be modified.
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
            len(actions_actions),
            change_size,
        )

        next_actions_actions = []
        selected_action_sets = []
        evaluate_args = [
            (action, actions, process_execution, change_size)
            for action, actions in actions_actions
        ]

        with mp.Pool(self.num_workers) as pool:
            for explored, selected in pool.starmap(self.explore_actions, evaluate_args):
                # Collect distinct next actions
                for next_action in explored:
                    if next_action not in next_actions_actions:
                        next_actions_actions.append(next_action)

                # Collect distinct selected actions
                for selected_action in selected:
                    if selected_action not in selected_action_sets:
                        selected_action_sets.append(selected_action)

        if self.log_level == logging.DEBUG:
            with open(f"{self.log_file}-explored_actions-{change_size}", "w") as f:
                f.write("\n".join([f"{item[0]}" for item in next_actions_actions]))

        # Return when valid counterfactual actions are found
        if selected_action_sets:
            return selected_action_sets

        # Start next search layer
        return self.search_layer(
            next_actions_actions,
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
