from itertools import combinations
from math import comb
from networkx import NetworkXError
from pm4py.objects.ocel.constants import DEFAULT_EVENT_ACTIVITY, DEFAULT_OBJECT_TYPE

from typing import Any, Dict, Iterable, List, Optional, Tuple

from process_execution.process_execution import ProcessExecution
from process_execution.comparison import (
    attribute_diff_numeric,
    node_subst_cost,
)

TYPE_ATTRIBUTES = [
    "type",  # EVENT/OBJECT
    DEFAULT_EVENT_ACTIVITY,
    DEFAULT_OBJECT_TYPE,
]


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
        node_id (str): Identifier of the node whose attribute is being modified.
        attribute_name (str): Nme of the attribute to be modified.
        value_original (int | float | complex): Original value of the node attribute.
        value_range (Iterable): Range of possible values for the attribute.
    """

    def __init__(
        self,
        *args,
        node_id: str,
        attribute_name: str,
        value_original: int | float | complex,
        value_step: int | float,
        value_max: int | float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.node_id = node_id
        self.attribute_name = attribute_name
        self.value_original = value_original
        self.value_step = value_step
        self.value_max = value_max

    def __repr__(self) -> str:
        return f"{self.node_id} - {self.attribute_name} (action space size {self.action_space_size()})"

    def action_space_size(self):
        return (self.value_max - self.value_original) / self.value_step

    def action_space(
        self,
        current_change_value: int | float | complex = 0,
        max_change_size_delta: int | float | complex = 1,
    ) -> Iterable[int | float | complex]:
        """
        Generate possible values for the node attribute feature.
        Args:
            current_change_value (Optional[List[str]]): Current change value.
            max_change_size_delta (int | float | complex): Upper bound on the delta of the change size.
        Yields:
            Iterable[int | float | complex]: Possible values for the attribute.
        """

        change_value = current_change_value if current_change_value else 0
        current_change_size = self.change_size(change_value)

        while change_value + self.value_step < self.value_max:
            change_value += self.value_step
            if (
                self.change_size(change_value) - current_change_size
                <= max_change_size_delta
            ):
                yield change_value

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

    def change_size(self, change_value=0):
        return attribute_diff_numeric(
            self.value_original,
            self.value_original + change_value,
            interval_size=self.value_step,
        )


class ObjectNodeSubstitution(Feature):
    """
    Feature representing possible object node substitutions in a process execution.
    Attributes:
        object_id (str): identifier of the object to substitute.
        substitution_objects (List[Tuple[str, dict]]):
            An iterable of lists of substitution objects, where each substitution object is a tuple
            containing the substitute node (ID and attributes).
        event_id (Optional[str]): identifier of the event to substitute the object for.
            If not defined, the object is substituted for all events in the graph.
    """

    def __init__(
        self,
        *args,
        object_id: str,
        substitution_objects: List[Tuple[str, dict]],
        event_ids: List[str],
        object_data: dict = None,
        discretized_event_attributes: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.object_id = object_id
        self.substitution_objects = substitution_objects
        self.event_ids = event_ids
        self.object_data = object_data if object_data else {}
        self.discretized_event_attributes = discretized_event_attributes

    def __repr__(self) -> str:
        return f"Event(s) {self.event_ids} - object {self.object_id} with {len(self.substitution_objects)} substitution options"

    def action_space_size(self):
        return len(self.substitution_objects)

    def action_space(
        self,
        current_change_value: Tuple[str, dict] = None,
        max_change_size_delta: int | float | complex = 1,
    ) -> Iterable[Tuple[str, dict]]:
        """
        Generate possible substitution options for the object node feature.
        Args:
            current_change_value (Optional[List[str]]): Current change value.
            max_change_size_delta (int | float | complex): Upper bound on the delta of the change size.
        Yields:
            Iterable[Tuple[str, dict]]: substitution object.
        """

        for subst_node in self.substitution_objects:
            if self.change_size(subst_node=subst_node) <= max_change_size_delta:
                yield subst_node

    def apply_change(
        self,
        p: ProcessExecution,
        substitution_object: Tuple[str, dict],
    ) -> ProcessExecution:
        subst_object_id = substitution_object[0]
        subst_object_attr = substitution_object[1]

        # Add object node if not exists
        if not p.nodes.get(subst_object_id):
            p.add_node(subst_object_id, **subst_object_attr)

        # Replace event-object relationships
        remove_edges = []
        add_edges = []
        for u, v, d in p.in_edges(self.object_id, data=True):
            if u in self.event_ids:
                remove_edges.append((u, v))
                add_edges.append((u, subst_object_id, d))
        p.remove_edges_from(remove_edges)
        p.add_edges_from(add_edges)

        # Remove isolated object nodes (no incoming E2O edges)
        if not any(
            (u, v)
            for u, v, attr in p.in_edges(self.object_id, data="attr")
            if attr["type"] == "E2O"
        ):
            try:
                p.remove_node(self.object_id)
            except NetworkXError:
                print(f"{self.object_id} does not exist in the graph.")

        return p

    def change_size(self, subst_node: Tuple[str, dict] = None):
        subst_node_data = subst_node[1] if subst_node else {}
        return node_subst_cost(
            self.object_data,
            subst_node_data,
            exclude_attributes=TYPE_ATTRIBUTES,
            aggregation_type="sum",
            discretized_event_attributes=self.discretized_event_attributes,
        )


class EventNodeDeletion(Feature):
    """
    Feature representing possible event node deletions from a process execution.
    Attributes:
        deletion_options (Optional[Iterable[List[str]]]):
            An iterable of deletion options, where each deletion option is a list of nodes identifiers.
        allowed_deletions (Optional[List[str]]):
            A list of node identifiers that can be deleted.
    """

    def __init__(
        self,
        *args,
        deletion_options: Iterable[List[str]] = None,
        allowed_deletions: List[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.deletion_options = deletion_options = (
            deletion_options if deletion_options else []
        )
        self.allowed_deletions = allowed_deletions if allowed_deletions else []

    def __repr__(self) -> str:
        if self.deletion_options:
            return f"{len(self.deletion_options)} deletion options (action space size {self.action_space_size()})"
        else:
            return f"{len(self.allowed_deletions)} allowed deletions (action space size {self.action_space_size()})"

    def action_space_size(self):
        if self.deletion_options:
            return len(self.deletion_options)
        else:
            size = 0
            for r in range(len(self.allowed_deletions) + 1):
                size += comb(len(self.allowed_deletions), r)
            return size

    def action_space(
        self,
        current_change_value: Optional[List[str]] = None,
        max_change_size_delta: int | float | complex = 1,
    ) -> Iterable[List[str]]:
        """
        Generate possible deletion options for the object node feature.
        Args:
            current_change_value (Optional[List[str]]): Current change value.
            max_change_size_delta (int | float | complex): Upper bound on the delta of the change size.
        Yields:
            Iterable[List[str]]: Possible deletion options.
        """
        change_lower = (
            self.change_size(current_change_value) if current_change_value else 0
        )
        change_upper = change_lower + max_change_size_delta

        if self.deletion_options:
            for option in self.deletion_options:
                if (
                    self.change_size(option) > change_lower
                    and self.change_size(option) <= change_upper
                ):
                    yield option
        else:
            for r in range(len(self.allowed_deletions) + 1):
                for c in combinations(self.allowed_deletions, r):
                    if self.change_size(c) > change_upper:
                        break

                    # Only return set of nodes to delete that extends the current set
                    overlap = True
                    # if current_change_value:
                    #     print(c[:len(current_change_value)])
                    #     overlap = c[:len(current_change_value)] == current_change_value

                    if overlap and self.change_size(c) > change_lower:
                        yield c

    def change_size(self, del_nodes=None):
        if del_nodes:
            return len(del_nodes)
        else:
            return 0

    def apply_change(
        self,
        p: ProcessExecution,
        deletions: List[str],
    ) -> ProcessExecution:
        for deletion_node_id in deletions:
            # Add DF edges to 'skip' deletion node
            edges = []
            target_df_events = [
                v
                for _, v, d in p.out_edges(deletion_node_id, data=True)
                if d["attr"]["type"] == "DF"
            ]
            for u, _, d in p.in_edges(deletion_node_id, data=True):
                if d["attr"]["type"] != "DF":
                    continue
                edges.extend(
                    [(u, target_event, d) for target_event in target_df_events]
                )

            # Remove node
            p.remove_node(deletion_node_id)

            # Add edges
            p.add_edges_from(edges)

        return p
