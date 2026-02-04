import logging
import multiprocessing as mp

from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
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
        super().__init__(**kwargs)
        self.num_workers = num_workers

        # Listener for collecting logs from parallel processes
        log_listener = self._configure_log_listener()
        log_listener.start()

    def explore_features_worker(
        self,
        action: Action,
        features: List[Feature],
        process_execution: ProcessExecution,
        change_size=1,
    ):
        # Configure logging in each worker to send logs to the queue
        queue_handler = QueueHandler(log_queue)
        logger = logging.getLogger()
        logger.setLevel(self.log_level)
        logger.handlers = []  # Remove inherited handlers
        logger.addHandler(queue_handler)

        return super().explore_features(
            action, features, process_execution, change_size
        )

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
        self.logger.debug("change_size=%s\n%s", change_size, actions_features)

        # Terminate when max change size is reached
        if change_size >= self.max_change_size:
            self.logger.info(
                f"No valid counterfactual action found with size smaller than {self.max_change_size}"
            )
            return []

        explored_actions = []
        selected_actions = []
        evaluate_args = [
            (action, features, process_execution, change_size)
            for action, features in actions_features
        ]
        with mp.Pool(self.num_workers) as pool:
            for explored_actions_a, selected_actions_a in pool.starmap(
                self.explore_features_worker, evaluate_args
            ):
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
