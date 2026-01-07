from itertools import combinations
from math import comb

from typing import Any, Iterable, List, Tuple

from ..process_execution import ProcessExecution


def is_empty(iterator):
    try:
        next(iterator)
    except StopIteration:
        return True
    else:
        return False


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
        return f"{self.node_id} - {self.attribute_name} {self.value_range} (action space size {self.action_space_size()})"

    def action_space_size(self):
        return len(self.value_range)

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


class ObjectNodeSubstitution(Feature):
    """
    Feature representing possible object node substitutions in a process execution.
    Attributes:
        substitution_objects (List[Tuple[str, dict]]):
            An iterable of lists of substitution objects, where each substitution object is a tuple
            containing the substitute node (ID and attributes).
    """

    def __init__(
        self,
        *args,
        event_id: str,
        object_id: str,
        substitution_objects: List[Tuple[str, dict]],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_id = event_id
        self.object_id = object_id
        self.substitution_objects = substitution_objects

    def __repr__(self) -> str:
        return f"Event {self.event_id} - object {self.object_id} with {len(self.substitution_objects)} substitution options (action space size {self.action_space_size()})"

    def action_space_size(self):
        return len(self.substitution_objects)

    def action_space(self, current_change=None) -> Iterable[Tuple[str, dict]]:
        """
        Generate possible substitution options for the object node feature.
        Args:
            current_change: The current substitution option.
        Yields:
            Iterable[Tuple[str, dict]]: substitution object.
        """

        for obj in self.substitution_objects:
            yield obj

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
            if u == self.event_id:
                remove_edges.append((u, v))
                add_edges.append((u, subst_object_id, d))
        p.remove_edges_from(remove_edges)
        p.add_edges_from(add_edges)

        # Remove isolated object node
        if not p.in_edges(self.object_id):
            p.remove_node(self.object_id)

        return p


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

    def action_space(self, current_change) -> Iterable[List[str]]:
        """
        Generate possible deletion options for the object node feature.
        Args:
            current_change: The current deletion option.
        Yields:
            Iterable[List[str]]: Possible deletion options.
        """

        if self.deletion_options:
            for option in self.deletion_options:
                if len(option) >= len(current_change):
                    yield option
        else:
            for r in range(len(self.allowed_deletions) + 1):
                for c in combinations(self.allowed_deletions, r):
                    if len(c) >= len(current_change):
                        yield c

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
                for u, v, d in p.out_edges(deletion_node_id, data=True)
                if d["attr"]["type"] == "DF"
            ]
            for u, v, d in p.in_edges(deletion_node_id, data=True):
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
