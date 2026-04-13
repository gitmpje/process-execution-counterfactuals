from itertools import combinations
from math import ceil, comb
from copy import deepcopy
from networkx import NetworkXError
from pm4py.objects.ocel.constants import DEFAULT_EVENT_ACTIVITY, DEFAULT_OBJECT_TYPE

from typing import Any, Dict, Iterable, List, Optional, Tuple

from process_execution.process_execution import ProcessExecution
from process_execution.comparison import (
    attribute_diff,
    attribute_diff_numeric,
    node_subst_cost,
)

TYPE_ATTRIBUTES = [
    "type",  # EVENT/OBJECT
    DEFAULT_EVENT_ACTIVITY,
    DEFAULT_OBJECT_TYPE,
]


class Action:
    """
    Base class for actions that can be modified in a process execution.
    """

    def __init__(
        self,
    ):
        """Initialize the Action base class."""

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return False

        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        # Subclasses should provide a reasonable __repr__ implementation
        # that reflects the important attributes.
        return hash(repr(self))

    def apply_change(self, p: ProcessExecution) -> ProcessExecution:
        """
        Apply the change represented by this action to the process execution `p`.
        """
        raise NotImplementedError("apply_change must be implemented by subclasses")

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        """
        Reverse a previously applied change using the information stored in
        `record`. Subclasses must override when they modify the graph in a
        nontrivial way.
        """
        raise NotImplementedError("undo_change must be implemented by subclasses")


class NodeAttributeNumeric(Action):
    """
    Action representing a node attribute that can be modified.
    Attributes:
        node_id (str): Identifier of the node whose attribute is being modified.
        attribute_name (str): Name of the attribute to be modified.
        value_original (int | float): Original value of the node attribute.
    """

    def __init__(
        self,
        *args,
        node_id: str,
        attribute_name: str,
        value_original: int | float,
        value_step: int | float,
        value_min: int | float,
        value_max: int | float,
        **kwargs,
    ):
        """
        Initialize NodeAttributeNumeric action.

        Args:
            node_id: Identifier of the node.
            attribute_name: Name of the attribute.
            value_original: Original value.
            value_step: Step size for changes.
            value_min: Minimum value.
            value_max: Maximum value.
        """
        super().__init__(*args, **kwargs)
        self.node_id = node_id
        self.attribute_name = attribute_name
        self.value_original = value_original
        self.value_step = value_step
        self.value_min = value_min
        self.value_max = value_max

    def __repr__(self) -> str:
        return f"{self.node_id} - {self.attribute_name} (action space size {self.action_space_size()})"

    def action_space_size(self) -> int:
        return ceil((self.value_max - self.value_original) / self.value_step) + ceil(
            (self.value_original - self.value_min) / self.value_step
        )

    def action_space(
        self,
        current_change_value: Optional[int | float] = None,
        max_change_size_delta: Optional[int | float] = 1,
    ) -> Iterable[int | float]:
        """
        Generate possible values for the node attribute action.
        Args:
            current_change_value (Optional[List[str]]): Current change value.
            max_change_size_delta (int | float): Upper bound on the delta of the change size.
        Yields:
            Iterable[int | float]: Possible values for the attribute.
        """
        current_change_value = current_change_value if current_change_value else 0
        change_lower = self.change_size(current_change_value)
        change_upper = change_lower + max_change_size_delta

        # Decrease
        change_value = current_change_value
        if current_change_value <= 0:
            while True:
                change_value = (
                    change_value - self.value_step
                    if self.value_original + change_value - self.value_step
                    >= self.value_min
                    else self.value_min - self.value_original
                )

                change_size = self.change_size(change_value)
                if (change_size > change_lower) and (change_size <= change_upper):
                    yield change_value

                if self.value_original + change_value == self.value_min:
                    break

        # Increase
        change_value = current_change_value
        if current_change_value >= 0:
            while True:
                change_value = (
                    change_value + self.value_step
                    if self.value_original + change_value + self.value_step
                    <= self.value_max
                    else self.value_max - self.value_original
                )

                change_size = self.change_size(change_value)
                if (change_size > change_lower) and (change_size <= change_upper):
                    yield change_value

                if self.value_original + change_value == self.value_max:
                    break

    def apply_change(self, p: ProcessExecution, delta_value: Any) -> Any:
        """
        Apply the attribute value change to the process execution and return the
        original value so that the operation can be undone later.
        Args:
            p (ProcessExecution): The process execution to modify.
            delta_value (Any): The value to add to the current attribute value.
        Returns:
            Any: the previous attribute value (undo record).
        """
        node = p.nodes()[self.node_id]
        old = node["attr"][self.attribute_name]
        node["attr"][self.attribute_name] = old + delta_value
        return old

    def change_size(self, change_value=0) -> float:
        return attribute_diff_numeric(
            self.value_original,
            self.value_original + change_value,
            interval_size=self.value_step,
        )

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        # record holds the original value
        if p.has_node(self.node_id):
            p.nodes()[self.node_id]["attr"][self.attribute_name] = record
        return p


class NodeAttributeCategorical(Action):
    """
    Action representing a node attribute that can be modified.
    Attributes:
        node_id (str): Identifier of the node whose attribute is being modified.
        attribute_name (str): Name of the attribute to be modified.
        value_original (int | float): Original value of the node attribute.
        value_range (Iterable): Range of possible values for the attribute.
    """

    def __init__(
        self,
        *args,
        node_id: str,
        attribute_name: str,
        value_original: str,
        category_values: List[str],
        **kwargs,
    ):
        """
        Initialize NodeAttributeCategorical action.

        Args:
            node_id: Identifier of the node.
            attribute_name: Name of the attribute.
            value_original: Original value.
            category_values: List of possible values.
        """
        super().__init__(*args, **kwargs)
        self.node_id = node_id
        self.attribute_name = attribute_name
        self.value_original = value_original
        self.category_values = category_values

    def __repr__(self) -> str:
        return f"{self.node_id} - {self.attribute_name} (action space size {self.action_space_size()})"

    def action_space_size(self) -> int:
        return len([v for v in self.category_values if v != self.value_original])

    def action_space(
        self,
        current_change_value: Optional[str] = None,
        max_change_size_delta: Optional[int | float] = 1,
    ) -> Iterable[str]:
        """
        Generate possible values for the node attribute action.
        Args:
            current_change_value (Optional[str]): Current change value.
            max_change_size_delta (int | float): Upper bound on the delta of the change size.
        Yields:
            Iterable[str]: Possible values for the attribute.
        """
        change_lower = (
            self.change_size(current_change_value) if current_change_value else 0
        )
        change_upper = change_lower + max_change_size_delta

        for change_value in self.category_values:
            change_size = self.change_size(change_value)
            if (change_size > change_lower) and (change_size <= change_upper):
                yield change_value

    def apply_change(self, p: ProcessExecution, value: Any) -> Any:
        """
        Apply the attribute value change to the process execution and return the
        original value so that undoing is possible.
        Args:
            p (ProcessExecution): The process execution to modify.
            value (Any): The value to set the attribute to.
        Returns:
            Any: the previous attribute value.
        """
        node = p.nodes()[self.node_id]
        old = node["attr"][self.attribute_name]
        node["attr"][self.attribute_name] = value
        return old

    def change_size(self, change_value=0) -> int:
        return attribute_diff(
            self.value_original,
            change_value,
        )

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        if p.has_node(self.node_id):
            p.nodes()[self.node_id]["attr"][self.attribute_name] = record
        return p


class ObjectNodeSubstitution(Action):
    """
    Action representing possible object node substitutions in a process execution.
    Attributes:
        object_id (str): identifier of the object to substitute.
        substitution_objects (List[Tuple[str, dict]]):
            An iterable of lists of substitution objects, where each substitution object is a tuple
            containing the substitute node (ID and attributes).
        event_id (Optional[str]): identifier of the event to substitute the object for.
            If not defined, the object is substituted for all events in the graph.
        include_attributes (List[str]): list of attributes to include in the node comparison.
    """

    def __init__(
        self,
        *args,
        object_id: str,
        substitution_objects: List[Tuple[str, dict]],
        event_ids: List[str],
        object_data: dict = None,
        include_attributes: List[str] = None,
        discretized_attributes: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Initialize ObjectNodeSubstitution action.

        Args:
            object_id: Identifier of the object to substitute.
            substitution_objects: List of substitution objects.
            event_ids: List of event IDs.
            object_data: Object data.
            include_attributes: Attributes to include.
            discretized_attributes: Discretized attributes.
        """
        super().__init__(*args, **kwargs)
        self.object_id = object_id
        self.substitution_objects = substitution_objects
        self.event_ids = event_ids
        self.object_data = object_data or {}
        self.include_attributes = include_attributes
        self.discretized_attributes = discretized_attributes

    def __repr__(self) -> str:
        return f"Event(s) {self.event_ids} - object {self.object_id} with {len(self.substitution_objects)} substitution options"

    def action_space_size(self) -> int:
        return len(self.substitution_objects)

    def action_space(
        self,
        current_change_value: Tuple[str, dict] = None,
        max_change_size_delta: int | float = 1,
    ) -> Iterable[Tuple[str, dict]]:
        """
        Generate possible substitution options for the object node action.
        Args:
            current_change_value (Optional[List[str]]): Current change value.
            max_change_size_delta (int | float): Upper bound on the delta of the change size.
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
    ) -> dict:
        """
        Apply the object substitution and return a small snapshot of the graph
        state that is necessary to undo the operation.
        """
        subst_object_id = substitution_object[0]
        subst_object_attr = substitution_object[1]

        # snapshot nodes/edges involving the two object IDs before modification
        record: dict = {"nodes": {}, "edges": [], "nodes_added": []}
        for nid in (self.object_id, subst_object_id):
            if p.has_node(nid):
                record["nodes"][nid] = p.nodes[nid].copy()

        record["edges"] = [
            (u, v, k, d.copy())
            for u, v, k, d in p.edges(keys=True, data=True)
            if u in {self.object_id, subst_object_id}
            or v in {self.object_id, subst_object_id}
        ]

        # Add object node if not exists
        if not p.nodes.get(subst_object_id):
            p.add_node(subst_object_id, **subst_object_attr)
            record["nodes_added"].append(nid)

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

        return record

    def change_size(self, subst_node: Tuple[str, dict] = None):
        subst_node_data = subst_node[1] if subst_node else {}
        return node_subst_cost(
            self.object_data,
            subst_node_data,
            include_attributes=self.include_attributes,
            aggregation_type="sum",
            discretized_attributes=self.discretized_attributes,
        )

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        # restore nodes/edges snapshot saved before substitution
        # first, remove edges involving the two object ids
        ids = {self.object_id}
        # if substitution object was added, we need its id too
        # record may not have had it explicitly but edges snapshot contains both ids
        ids |= set(record.get("nodes", {}).keys())
        for u, v, k, _ in list(p.edges(keys=True, data=True)):
            if u in ids or v in ids:
                try:
                    p.remove_edge(u, v, key=k)
                except Exception:
                    pass
        # restore nodes attrs
        for nid, attrs in record.get("nodes", {}).items():
            if not p.has_node(nid):
                p.add_node(nid, **attrs)
            else:
                p.nodes()[nid].update(attrs)
        # restore edges
        for u, v, k, d in record.get("edges", []):
            if not p.has_edge(u, v, key=k):
                p.add_edge(u, v, key=k, **d)
        # remove substitution node if it was created by the change
        for nid in record.get("nodes_added", []):
            if p.has_node(nid):
                try:
                    p.remove_node(nid)
                except Exception:
                    pass
        return p


class EventNodeSubstitution(Action):
    """
    Action representing possible event node substitutions in a process execution.
    Attributes:
        event_id (str): identifier of the event to substitute.
        event_data (dict): data of the event to substitute.
        substitution_events (List[str]):
            An iterable of lists of substitution events (node IDs).
        include_attributes (List[str]): list of attributes to include in the node comparison.
    """

    def __init__(
        self,
        *args,
        event_id: str,
        event_data: dict,
        substitution_events: List[Tuple[str, dict]],
        include_attributes: List[str] = None,
        discretized_attributes: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_id = event_id
        self.event_data = event_data
        self.substitution_events = substitution_events
        self.include_attributes = include_attributes
        self.discretized_attributes = discretized_attributes

    def __repr__(self) -> str:
        return f"Event {self.event_id} with {len(self.substitution_events)} substitution options"

    def action_space_size(self):
        return len(self.substitution_events)

    def action_space(
        self,
        current_change_value: str = None,
        max_change_size_delta: int | float = 1,
    ) -> Iterable[Tuple[str, dict]]:
        """
        Generate possible substitution options for the event node action.
        Args:
            current_change_value (Optional[List[str]]): Current change value.
            max_change_size_delta (int | float): Upper bound on the delta of the change size.
        Yields:
            Iterable[Tuple[str, dict]]: substitution event.
        """

        for subst_node in self.substitution_events:
            if self.change_size(subst_node=subst_node) <= max_change_size_delta:
                yield subst_node

    def apply_change(
        self,
        p: ProcessExecution,
        substitution_event: Tuple[str, dict],
    ) -> dict:
        """
        Apply the event substitution and return a small snapshot of the graph
        state that is necessary to undo the operation.

        Swaps all incoming/outgoing DF edges between self.event_id and
        subst_event_id and swaps ocel:timestamp and epoch attributes (if present).
        """
        subst_event_id = substitution_event[0]

        # snapshot state so undo can restore exactly
        record: dict = {
            "nodes": {},
            "edges": [],
            "had_subst": False,
        }

        for nid in (self.event_id, subst_event_id):
            if p.has_node(nid):
                record["nodes"][nid] = deepcopy(p.nodes[nid])
            else:
                # Skip if one of the two nodes does not exist in the graph
                return record

        record["had_subst"] = p.has_node(subst_event_id)

        record["edges"] = [
            (u, v, k, deepcopy(d))
            for u, v, k, d in p.edges(keys=True, data=True)
            if u in {self.event_id, subst_event_id}
            or v in {self.event_id, subst_event_id}
        ]

        if not p.has_node(subst_event_id):
            raise Exception(f"Event node {subst_event_id} does not exist")

        # Collect DF edges for the two nodes
        def df_edges(node):
            incoming = [
                (u, v, k, d)
                for u, v, k, d in p.in_edges(node, keys=True, data=True)
                if d.get("attr", {}).get("type") == "DF"
            ]
            outgoing = [
                (u, v, k, d)
                for u, v, k, d in p.out_edges(node, keys=True, data=True)
                if d.get("attr", {}).get("type") == "DF"
            ]
            return incoming, outgoing

        event_in, event_out = df_edges(self.event_id)
        subst_in, subst_out = df_edges(subst_event_id)

        # remove original DF edges
        for u, v, k, _ in event_in + event_out + subst_in + subst_out:
            if p.has_edge(u, v, key=k):
                p.remove_edge(u, v, key=k)

        # add swapped relationships
        for u, _, k, d in event_in:
            p.add_edge(u, subst_event_id, key=k, **{"attr": d.get("attr", {}).copy()})
        for _, v, k, d in event_out:
            p.add_edge(subst_event_id, v, key=k, **{"attr": d.get("attr", {}).copy()})

        for u, _, k, d in subst_in:
            p.add_edge(u, self.event_id, key=k, **{"attr": d.get("attr", {}).copy()})
        for _, v, k, d in subst_out:
            p.add_edge(self.event_id, v, key=k, **{"attr": d.get("attr", {}).copy()})

        # swap timestamps/epoch if present in node attr dicts
        for key in ["ocel:timestamp", "epoch"]:
            v1 = p.nodes[self.event_id].get("attr", {}).get(key)
            v2 = p.nodes[subst_event_id].get("attr", {}).get(key)
            if v1 is not None or v2 is not None:
                p.nodes[self.event_id]["attr"][key] = v2
                p.nodes[subst_event_id]["attr"][key] = v1

        return record

    def change_size(self, subst_node: Tuple[str, dict] = None):
        subst_node_data = subst_node[1] if subst_node else {}
        return node_subst_cost(
            self.event_data,
            subst_node_data,
            include_attributes=self.include_attributes,
            aggregation_type="sum",
            discretized_attributes=self.discretized_attributes,
        )

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        # Restore nodes as they were at snapshot time.
        for nid, attrs in record.get("nodes", {}).items():
            if not p.has_node(nid):
                p.add_node(nid, **attrs)
            else:
                p.nodes()[nid].clear()
                p.nodes()[nid].update(attrs)

        # Remove all current edges incident to the involved nodes.
        involved = set(record.get("nodes", {}).keys())
        for u, v, k, _ in list(p.edges(keys=True, data=True)):
            if u in involved or v in involved:
                try:
                    p.remove_edge(u, v, key=k)
                except Exception:
                    pass

        # Restore snapshot edges.
        for u, v, k, d in record.get("edges", []):
            if not p.has_edge(u, v, key=k):
                p.add_edge(u, v, key=k, **d)

        return p


class EventNodeMove(Action):
    """
    Action representing possible event node moving in a process execution.
    Attributes:
        event_id (str): identifier of the event to substitute.
        target_events (List[str]):
            An iterable of lists of target events (node IDs) after which to move the event.
    """

    def __init__(
        self,
        *args,
        event_id: str,
        target_events: List[str],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_id = event_id
        self.target_events = target_events

    def __repr__(self) -> str:
        return f"Event {self.event_id} with {len(self.target_events)} target options"

    def action_space_size(self):
        return len(self.target_events)

    def action_space(
        self,
        current_change_value: Optional[str] = None,
        max_change_size_delta: int | float = 1,
    ) -> Iterable[Tuple[str, dict]]:
        """
        Generate possible target options for the event node action.
        Args:
            current_change_value (Optional[str]): Current change value.
            max_change_size_delta (int | float): Upper bound on the delta of the change size.
        Yields:
            Iterable[str]: target event after which to move the event.
        """

        for target_event_id in self.target_events:
            if (
                self.change_size(target_event_id=target_event_id)
                <= max_change_size_delta
            ):
                yield target_event_id

    def apply_change(
        self,
        p: ProcessExecution,
        target_event_id: str,
    ) -> dict:
        """
        Apply the event move and return a small snapshot of the graph
        state that is necessary to undo the operation.

        Moves self.event_id to follow target_event_id by updating DF relations.
        """

        # snapshot state so undo can restore exactly
        record: dict = {
            "nodes": {},
            "original_edges": [],
            "added_edges": [],
        }

        record["nodes"][self.event_id] = deepcopy(p.nodes[self.event_id])

        record["original_edges"] = [
            (u, v, k, deepcopy(d))
            for u, v, k, d in p.edges(keys=True, data=True)
            if u in {self.event_id, target_event_id}
            or v in {self.event_id, target_event_id}
        ]

        if not p.has_node(target_event_id):
            raise Exception(f"Event node {target_event_id} does not exist")

        # Collect DF edges for the two nodes
        def df_edges(node):
            incoming = [
                (u, v, k, d)
                for u, v, k, d in p.in_edges(node, keys=True, data=True)
                if d.get("attr", {}).get("type") == "DF"
            ]
            outgoing = [
                (u, v, k, d)
                for u, v, k, d in p.out_edges(node, keys=True, data=True)
                if d.get("attr", {}).get("type") == "DF"
            ]
            return incoming, outgoing

        self_in, self_out = df_edges(self.event_id)
        _, target_out = df_edges(target_event_id)

        # remove original DF edges for the moved node and target node
        for u, v, k, _ in self_in + self_out + target_out:
            if p.has_edge(u, v, key=k):
                p.remove_edge(u, v, key=k)

        # reconnect self.event_id's original neighbors around its old position
        for u, _, k, d in self_in:
            for _, v, _, _ in self_out:
                p.add_edge(u, v, key=k, **{"attr": d.get("attr", {}).copy()})

                record["added_edges"].append((u, v, k))

        # insert self.event_id after target_event_id
        p.add_edge(target_event_id, self.event_id, attr={"type": "DF"})
        record["added_edges"].append((target_event_id, self.event_id, 0))

        # connect self.event_id to target's old successors
        for _, v, _, d in target_out:
            p.add_edge(self.event_id, v, attr=d.get("attr", {}).copy())
            record["added_edges"].append((self.event_id, v, 0))

        # swap timestamps/epoch if present in node attr dicts
        for key in ["ocel:timestamp", "epoch"]:
            v1 = p.nodes[self.event_id].get("attr", {}).get(key)
            v2 = p.nodes[target_event_id].get("attr", {}).get(key)
            if v1 is not None or v2 is not None:
                p.nodes[self.event_id]["attr"][key] = v2

        return record

    def change_size(self, target_event_id: str = None):
        if target_event_id:
            return 1

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        # Restore nodes as they were at snapshot time.
        for nid, attrs in record.get("nodes", {}).items():
            if not p.has_node(nid):
                p.add_node(nid, **attrs)
            else:
                p.nodes()[nid].clear()
                p.nodes()[nid].update(attrs)

        # Remove added edges.
        for u, v, k in record.get("added_edges", []):
            if p.has_edge(u, v, key=k):
                p.remove_edge(u, v, key=k)

        # Restore snapshot edges.
        for u, v, k, d in record.get("original_edges", []):
            if not p.has_edge(u, v, key=k):
                p.add_edge(u, v, key=k, **d)

        return p


class NodeDeletion(Action):
    """
    Action representing possible node deletions from a process execution.
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
        self.deletion_options = deletion_options or []
        self.allowed_deletions = allowed_deletions or []

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
        max_change_size_delta: int | float = 1,
    ) -> Iterable[List[str]]:
        """
        Generate possible deletion options.
        Args:
            current_change_value (Optional[List[str]]): Current change value.
            max_change_size_delta (int | float): Upper bound on the delta of the change size.
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

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        # re‑add deleted nodes and edges, remove any edges that were inserted
        for node_id, attrs in record.get("deleted_nodes", []):
            if not p.has_node(node_id):
                p.add_node(node_id, **attrs)
            else:
                p.nodes()[node_id].update(attrs)
        for u, v, d in record.get("deleted_edges", []):
            if not p.has_edge(u, v):
                p.add_edge(u, v, **d)
        for u, v, _ in record.get("added_edges", []):
            if p.has_edge(u, v):
                try:
                    p.remove_edge(u, v)
                except Exception:
                    pass
        return p

    def apply_change(
        self,
        p: ProcessExecution,
        deletions: List[str],
    ) -> dict:
        """
        Remove the requested nodes, recording their attributes and incident
        edges so they can be restored later.
        Returns a dict with ``deleted_nodes`` and ``deleted_edges`` plus any
        ``added_edges`` that were created during event‑node deletion.
        """
        record = {"deleted_nodes": [], "deleted_edges": [], "added_edges": []}
        for deletion_node_id in deletions:
            # capture node data
            if p.has_node(deletion_node_id):
                record["deleted_nodes"].append(
                    (deletion_node_id, p.nodes()[deletion_node_id].copy())
                )
                # capture any incident edges (both in and out)
                record["deleted_edges"].extend(
                    [
                        (u, v, d.copy())
                        for u, v, d in p.in_edges(nbunch=[deletion_node_id], data=True)
                    ]
                )
                record["deleted_edges"].extend(
                    [
                        (u, v, d.copy())
                        for u, v, d in p.out_edges(nbunch=[deletion_node_id], data=True)
                    ]
                )
                p.remove_node(deletion_node_id)
        return record


class EventNodeInsertion(Action):
    """Action for inserting a new event after an existing event node."""

    def __init__(
        self,
        *args,
        event_id: str,
        event_data_options: List[Dict[str, Any]],
        object_ids: List[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_id = event_id
        self.event_data_options = event_data_options
        self.object_ids = object_ids or []

    def __repr__(self) -> str:
        return f"Insert event after {self.event_id} related to objects {self.object_ids} with {len(self.event_data_options)} data options"

    def action_space_size(self):
        return len(self.event_data_options)

    def action_space(self, current_change_value=None, max_change_size_delta=1):
        if max_change_size_delta >= 1:
            for data in self.event_data_options:
                if isinstance(data, dict) and "attr" in data:
                    yield data["attr"]
                else:
                    yield data

    def apply_change(self, p: ProcessExecution, event_data: Any = None) -> dict:
        import uuid

        selected_event = (
            event_data if isinstance(event_data, dict) else self.event_data_options[0]
        )

        # Normalize to raw node attributes; support wrapped format {'attr': {...}}
        if isinstance(selected_event, dict) and "attr" in selected_event:
            node_attr = deepcopy(selected_event["attr"])
        else:
            node_attr = deepcopy(selected_event)

        new_event_id = f"insert_event_{uuid.uuid4().hex}"

        record = {
            "new_event_id": new_event_id,
            "event_id": self.event_id,
            "added_node": None,
            "old_df_edges": [],
            "added_edges": [],
            "added_e2o": [],
            "event_data": selected_event,
        }

        # Create the new event node
        p.add_node(new_event_id, attr=node_attr)
        record["added_node"] = new_event_id

        # Relink DF edges: event_id -> new_event -> old successors
        old_df_out = [
            (u, v, k, d.copy())
            for u, v, k, d in p.out_edges(self.event_id, keys=True, data=True)
            if d.get("attr", {}).get("type") == "DF"
        ]

        # Remove existing outgoing DF edges from parent event
        for u, v, k, _ in old_df_out:
            if p.has_edge(u, v, key=k):
                p.remove_edge(u, v, key=k)

        # Add edge from parent to new event
        p.add_edge(self.event_id, new_event_id, attr={"type": "DF"})
        record["added_edges"].append(
            (self.event_id, new_event_id, {"attr": {"type": "DF"}})
        )

        # Add edge from new event to old successors
        for _, v, _, d in old_df_out:
            p.add_edge(new_event_id, v, attr=d.get("attr", {}).copy())
            record["added_edges"].append((new_event_id, v, d.get("attr", {}).copy()))
            record["old_df_edges"].append((self.event_id, v, d.get("attr", {}).copy()))

        # Add E2O relationships to the specified objects
        for obj_id in self.object_ids:
            if p.has_node(obj_id):
                p.add_edge(new_event_id, obj_id, attr={"type": "E2O"})
                record["added_e2o"].append((new_event_id, obj_id))

        return record

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        new_event_id = record.get("new_event_id")

        # Remove edges incident to new event
        for u, v, k, _ in list(p.edges(keys=True, data=True)):
            if u == new_event_id or v == new_event_id:
                try:
                    p.remove_edge(u, v, key=k)
                except Exception:
                    pass

        # Restore old DF edges from parent to old successors
        for u, v, attr in record.get("old_df_edges", []):
            if not p.has_edge(u, v):
                p.add_edge(u, v, attr=attr)

        # Remove the new event node
        if new_event_id and p.has_node(new_event_id):
            try:
                p.remove_node(new_event_id)
            except Exception:
                pass

        return p

    def change_size(self, event_data: Any = None):
        if event_data:
            return 1
        else:
            return 0


class ObjectNodeInsertion(Action):
    """Action for inserting a new object node for a given event."""

    def __init__(
        self,
        *args,
        event_id: str,
        object_data_options: List[Dict[str, Any]],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_id = event_id
        self.object_data_options = object_data_options

    def __repr__(self) -> str:
        return f"Insert object for event {self.event_id} with {len(self.object_data_options)} data options"

    def action_space_size(self):
        return len(self.object_data_options)

    def action_space(self, current_change_value=None, max_change_size_delta=1):
        if max_change_size_delta >= 1:
            for option in self.object_data_options:
                if isinstance(option, dict) and "attr" in option:
                    yield option["attr"]
                else:
                    yield option

    def apply_change(self, p: ProcessExecution, object_data: Any = None) -> dict:
        import uuid

        selected_object = (
            object_data
            if isinstance(object_data, dict)
            else self.object_data_options[0]
        )

        # normalize nested attr payload
        if isinstance(selected_object, dict) and "attr" in selected_object:
            node_attr = deepcopy(selected_object["attr"])
        else:
            node_attr = deepcopy(selected_object)

        new_object_id = f"insert_object_{uuid.uuid4().hex}"

        record = {
            "new_object_id": new_object_id,
            "event_id": self.event_id,
            "added_edge": None,
            "object_data": selected_object,
        }

        p.add_node(new_object_id, attr=node_attr)
        p.add_edge(self.event_id, new_object_id, attr={"type": "E2O"})

        record["added_edge"] = (self.event_id, new_object_id)

        return record

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        new_object_id = record.get("new_object_id")
        if new_object_id and p.has_node(new_object_id):
            try:
                p.remove_node(new_object_id)
            except Exception:
                pass
        return p

    def change_size(self, object_data: Any = None):
        if object_data:
            return 1
        else:
            return 0


class EventNodeDeletion(NodeDeletion):
    """
    Action representing possible event node deletions from a process execution.

    Records the nodes/edges removed and any skip‑edges that are added so that
    the operation can be undone.
    """

    def apply_change(
        self,
        p: ProcessExecution,
        deletions: List[str],
    ) -> dict:
        base_rec = super().apply_change(p, deletions)

        # identify added "skip" edges so they can be removed on undo
        added = []
        for deletion_node_id in deletions:
            target_df_events = [
                v
                for u, v, d in base_rec["deleted_edges"]
                if u == deletion_node_id and d["attr"]["type"] == "DF"
            ]
            for u, v, d in base_rec["deleted_edges"]:
                if v != deletion_node_id or d["attr"]["type"] != "DF":
                    continue

                added.extend(
                    [(u, target_event, d) for target_event in target_df_events]
                )

        base_rec["added_edges"] = added
        p.add_edges_from(added)

        return base_rec

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        # reuse NodeDeletion undo and then remove any added edges as recorded
        p = super().undo_change(p, record)
        for u, v, _ in record.get("added_edges", []):
            if p.has_edge(u, v):
                try:
                    p.remove_edge(u, v)
                except Exception:
                    pass
        return p


class ObjectNodeDeletion(NodeDeletion):
    def apply_change(
        self,
        p: ProcessExecution,
        deletions: List[str],
    ) -> dict:
        # base deletion collects removed nodes/edges; nothing special to add here
        return super().apply_change(p, deletions)

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        # same logic as base class
        return super().undo_change(p, record)
