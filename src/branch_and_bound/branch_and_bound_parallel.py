import logging
import multiprocessing as mp

from copy import deepcopy
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from numpy import inf
from queue import Empty
from typing import Iterable, List

from process_execution.process_execution import ProcessExecution

from branch_and_bound import Action, BranchAndBoundCounterFactual
from branch_and_bound.feature import Feature


class BranchAndBoundCounterFactualParallel(BranchAndBoundCounterFactual):
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
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_workers = num_workers

    def find_counterfactuals(
        self,
        available_features: List[Feature],
        process_execution: ProcessExecution,
    ) -> List[Action]:
        """
        Find counterfactual actions using multiprocessing.
        Args:
            available_features (List[Feature]): List of available features.
            process_execution (ProcessExecution): The original process execution.
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

        task_queue = mp.Queue()
        result_queue = mp.Queue()

        action = Action()
        fixed_features = []
        task_queue.put(
            (action, available_features[:], fixed_features[:], process_execution)
        )

        processes = []
        min_objective_value = mp.Value("f", inf)

        try:
            workers = [
                mp.Process(
                    target=self._branch_and_bound_worker,
                    args=(task_queue, result_queue, min_objective_value, log_queue),
                )
                for _ in range(self.num_workers)
            ]
            for worker in workers:
                worker.start()

            for worker in workers:
                worker.join()

            # Collect results
            selected_actions = []
            min_objective_value = inf
            while not result_queue.empty():
                selected_action = result_queue.get()
                objective_value = selected_action.objective_value()
                if objective_value <= min_objective_value:
                    min_objective_value = objective_value
                    selected_actions.append((selected_action, objective_value))

            return [a for a, v in selected_actions if v == min_objective_value]

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Terminating all processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.0)  # Wait up to 1 second for process to terminate
                    if p.is_alive():
                        p.kill()  # Force kill if still alive
            raise
        finally:
            task_queue.close()
            result_queue.close()
            log_queue.close()

    def _action_queue_clean_up(
        self,
        action: Action,
        task_queue: mp.Queue,
        available_features,
        fixed_features,
        process_execution,
    ) -> Iterable[Action]:
        if action.objective_value() >= min_objective_value.value:
            pass
        else:
            task_queue.put(
                action, available_features, fixed_features, process_execution
            )

    def _branch_and_bound_worker(
        self,
        task_queue: mp.Queue,
        selected_actions_queue: mp.Queue,
        min_objective_value,
        log_queue: mp.Queue,
    ) -> Action | None:
        """Worker process for parallel branch and bound.
        Args:
           task_queue (mp.Queue): Queue containing tasks to process.
           selected_actions_queue (mp.Queue): Queue to store selected counterfactual actions.
           min_objective_value (mp.Value): Shared value for the minimum objective value found.
           log_queue (mp.Queue): Queue for logging.
        """
        self._setup_worker(log_queue)
        logger = logging.getLogger(__name__)

        while True:
            logger.info("Queue size: %s", task_queue.qsize())
            try:
                action, available_features, fixed_features, process_execution = (
                    task_queue.get(timeout=1)
                )
            except Empty:
                break

            # Prune branches that exceed max changes or current best objective value
            if action.action_size() > self.max_changes:
                continue

            if action.objective_value() >= min_objective_value.value:
                continue

            process_execution_c = action.apply_changes(deepcopy(process_execution))
            outcome_c = self.process_outcome(process_execution_c)
            logger.debug("""%s\nCounterfactual outcome: %s""", action, outcome_c)

            # Check if counterfactual condition is met
            if outcome_c == self.counterfactual_label:
                with min_objective_value.get_lock():
                    min_objective_value.value = action.objective_value()
                selected_actions_queue.put(deepcopy(action))
                continue

            # Select next feature to explore
            selected_feature = self.select_feature(available_features)
            if not selected_feature:
                continue
            fixed_features.append(selected_feature)
            print(selected_feature)

            # Explore all possible values for the selected feature
            for value in selected_feature.action_space(
                action.get_change_value(selected_feature)
            ):
                action_prime = deepcopy(action)
                action_prime.set_change_value(selected_feature, value)

                task_queue.put(
                    (
                        action_prime,
                        available_features[:],
                        fixed_features[:],
                        process_execution,
                    )
                )

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
    def _setup_worker(self, queue: mp.Queue):
        """
        Setup logging and shared variable for worker processes.
        Args:
            queue (mp.Queue): The logging queue.
        Returns:
            logging.Logger: Configured logger for the worker.
        """
        global log_queue
        log_queue = queue

        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)

        #  QueueHandler to send log records to a logging queue
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(queue_handler)

        return logger
