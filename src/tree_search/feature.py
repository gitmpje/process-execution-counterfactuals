from itertools import combinations
from math import ceil, comb
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


class Feature:
    """
    Base class for features that can be modified in a process execution.
    """

    def __init__(
        self,
    ):
        ...

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False

        return self.__dict__ == other.__dict__

    def __hash__(self):
        # Subclasses should provide a reasonable __repr__ implementation
        # that reflects the important attributes.
        return hash(repr(self))

    def apply_change(self, p: ProcessExecution) -> ProcessExecution:
        """
        Apply the change represented by this feature to the process execution `p`.
        """
        raise NotImplementedError("apply_change must be implemented by subclasses")

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        """
        Reverse a previously applied change using the information stored in
        `record`. Subclasses must override when they modify the graph in a
        nontrivial way.
        """
        raise NotImplementedError("undo_change must be implemented by subclasses")


class NodeAttributeNumeric(Feature):
    """
    Feature representing a node attribute that can be modified.
    Attributes:
        node_id (str): Identifier of the node whose attribute is being modified.
        attribute_name (str): Nme of the attribute to be modified.
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
        super().__init__(*args, **kwargs)
        self.node_id = node_id
        self.attribute_name = attribute_name
        self.value_original = value_original
        self.value_step = value_step
        self.value_min = value_min
        self.value_max = value_max

    def __repr__(self) -> str:
        return f"{self.node_id} - {self.attribute_name} (action space size {self.action_space_size()})"

    def action_space_size(self):
        return ceil((self.value_max - self.value_original) / self.value_step) + ceil(
            (self.value_original - self.value_min) / self.value_step
        )

    def action_space(
        self,
        current_change_value: Optional[int | float] = None,
        max_change_size_delta: Optional[int | float] = 1,
    ) -> Iterable[int | float]:
        """
        Generate possible values for the node attribute feature.
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

    def change_size(self, change_value=0):
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


class NodeAttributeCategorical(Feature):
    """
    Feature representing a node attribute that can be modified.
    Attributes:
        node_id (str): Identifier of the node whose attribute is being modified.
        attribute_name (str): Nme of the attribute to be modified.
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
        super().__init__(*args, **kwargs)
        self.node_id = node_id
        self.attribute_name = attribute_name
        self.value_original = value_original
        self.category_values = category_values

    def __repr__(self) -> str:
        return f"{self.node_id} - {self.attribute_name} (action space size {self.action_space_size()})"

    def action_space_size(self):
        return len([v for v in self.category_values if v != self.value_original])

    def action_space(
        self,
        current_change_value: Optional[str] = None,
        max_change_size_delta: Optional[int | float] = 1,
    ) -> Iterable[str]:
        """
        Generate possible values for the node attribute feature.
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

    def change_size(self, change_value=0):
        return attribute_diff(
            self.value_original,
            change_value,
        )

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        if p.has_node(self.node_id):
            p.nodes()[self.node_id]["attr"][self.attribute_name] = record
        return p


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
        discretized_attributes: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.object_id = object_id
        self.substitution_objects = substitution_objects
        self.event_ids = event_ids
        self.object_data = object_data if object_data else {}
        self.discretized_attributes = discretized_attributes

    def __repr__(self) -> str:
        return f"Event(s) {self.event_ids} - object {self.object_id} with {len(self.substitution_objects)} substitution options"

    def action_space_size(self):
        return len(self.substitution_objects)

    def action_space(
        self,
        current_change_value: Tuple[str, dict] = None,
        max_change_size_delta: int | float = 1,
    ) -> Iterable[Tuple[str, dict]]:
        """
        Generate possible substitution options for the object node feature.
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
        record: dict = {"nodes": {}, "edges": [], "had_subst": False}
        for nid in (self.object_id, subst_object_id):
            if p.has_node(nid):
                record["nodes"][nid] = p.nodes[nid].copy()
        record["had_subst"] = p.has_node(subst_object_id)
        record["edges"] = [
            (u, v, k, d.copy())
            for u, v, k, d in p.edges(keys=True, data=True)
            if u in {self.object_id, subst_object_id}
            or v in {self.object_id, subst_object_id}
        ]

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

        return record

    def change_size(self, subst_node: Tuple[str, dict] = None):
        subst_node_data = subst_node[1] if subst_node else {}
        return node_subst_cost(
            self.object_data,
            subst_node_data,
            exclude_attributes=TYPE_ATTRIBUTES,
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
        subst_id = list(set(record.get("nodes", {}).keys()) - {self.object_id})
        if subst_id and not record.get("had_subst"):
            nid = subst_id[0]
            if p.has_node(nid):
                try:
                    p.remove_node(nid)
                except Exception:
                    pass
        return p


class NodeDeletion(Feature):
    """
    Feature representing possible node deletions from a process execution.
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
        for u, v, k, d in record.get("deleted_edges", []):
            if not p.has_edge(u, v, key=k):
                p.add_edge(u, v, key=k, **d)
        for u, v, k, _ in record.get("added_edges", []):
            if p.has_edge(u, v, key=k):
                try:
                    p.remove_edge(u, v, key=k)
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
                    [(u, v, k, d.copy()) for u, v, k, d in p.edges(
                        nbunch=[deletion_node_id], keys=True, data=True
                    )]
                )
                p.remove_node(deletion_node_id)
        return record

class EventNodeDeletion(NodeDeletion):
    """
    Feature representing possible event node deletions from a process execution.

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
                for _, v, d in p.out_edges(deletion_node_id, data=True)
                if d["attr"]["type"] == "DF"
            ]
            for u, _, d in p.in_edges(deletion_node_id, data=True):
                if d["attr"]["type"] != "DF":
                    continue
                added.extend(
                    [(u, target_event, d) for target_event in target_df_events]
                )
        base_rec["added_edges"] = added
        return base_rec

    def undo_change(self, p: ProcessExecution, record: Any) -> ProcessExecution:
        # reuse NodeDeletion undo and then remove any added edges as recorded
        p = super().undo_change(p, record)
        for u, v, k, _ in record.get("added_edges", []):
            if p.has_edge(u, v, key=k):
                try:
                    p.remove_edge(u, v, key=k)
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
