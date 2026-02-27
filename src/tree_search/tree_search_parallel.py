import logging
import multiprocessing as mp

from logging.handlers import QueueListener, RotatingFileHandler
from typing import List, Tuple

from process_execution.process_execution import ProcessExecution

from tree_search.tree_search import Action, TreeSearchCounterFactual
from tree_search.feature import Feature

log_queue = mp.Queue()


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
        super().__init__(log_file=f"{__name__}.log", **kwargs)
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

        self.logger.info(
            "Explored %s actions for change_size %s",
            len(next_actions_features),
            change_size,
        )

        if self.log_level == logging.DEBUG:
            with open(f"{self.log_file}-explored_actions-{change_size}", "w") as f:
                f.write("\n".join([f"{item[0]}" for item in next_actions_features]))

        # Return when valid counterfactual actions are found
        if selected_actions:
            return selected_actions

        # Start next search layer
        self.search_layer(
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
