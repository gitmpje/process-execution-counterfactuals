import json

from textwrap import indent
from typing import Any, Dict, List, Tuple

from process_execution.process_execution import ProcessExecution
from tree_search.action import (
    Action,
    EventNodeDeletion,
    EventNodeMove,
    EventNodeSubstitution,
    EventNodeInsertion,
    ObjectNodeInsertion,
    NodeAttributeCategorical,
    NodeAttributeNumeric,
    ObjectNodeDeletion,
    ObjectNodeSubstitution,
)

INDENTATION = 2


def make_json_safe(obj) -> Any:
    """Recursively convert Python objects into JSON‑serializable structures."""
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Everything else → string fallback
    return str(obj)


class ActionSet:
    """
    Class representing a set of changes (actions) to be applied to a process execution.
    Attributes:
        node_deletion (List[str]): List of node IDs to remove.
        object_substitution (Dict[ObjectSubstitutions, List[Tuple[Tuple[str, dict], Tuple[str, dict]]]): Dictionary mapping object substitutions options to the selected substitutions.
        node_attributes_modification (Dict[NodeAttributeNumeric, Any]): Dictionary mapping node attribute actions to their change value.
    """

    def __init__(
        self,
        node_attributes_modification: Dict[NodeAttributeNumeric, Any] = None,
        event_insertion: Dict[EventNodeInsertion, bool] = None,
        event_move: Dict[EventNodeMove, str | None] = None,
        event_substitution: Dict[EventNodeSubstitution, Tuple[str, dict] | None] = None,
        object_insertion: Dict[ObjectNodeInsertion, bool] = None,
        object_substitution: Dict[
            ObjectNodeSubstitution, Tuple[str, dict] | None
        ] = None,
        node_deletion: Dict[EventNodeDeletion, List[str]] = None,
    ):
        """
        Initialize an ActionSet with various types of modifications.

        Args:
            node_attributes_modification: Dictionary of node attribute modifications.
            event_insertion: Dictionary of event insertions.
            event_move: Dictionary of event moves.
            event_substitution: Dictionary of event substitutions.
            object_insertion: Dictionary of object insertions.
            object_substitution: Dictionary of object substitutions.
            node_deletion: Dictionary of node deletions.
        """
        self.node_attributes_modification = node_attributes_modification or {}
        self.event_move = event_move or {}
        self.event_substitution = event_substitution or {}
        self.object_substitution = object_substitution or {}
        self.event_insertion = event_insertion or {}
        self.object_insertion = object_insertion or {}
        self.node_deletion = node_deletion or {}

    def __repr__(self) -> str:
        # Pre-process substitution dictionaries
        event_substitution = {
            k: make_json_safe(v[0]) for k, v in self.event_substitution.items()
        }
        object_substitution = {
            k: make_json_safe(v[0]) for k, v in self.object_substitution.items()
        }

        event_insertion = {
            k: make_json_safe(v) for k, v in self.event_insertion.items()
        }
        object_insertion = {
            k: make_json_safe(v) for k, v in self.object_insertion.items()
        }

        full_dict = {
            "node_attributes_modification": make_json_safe(
                self.node_attributes_modification
            ),
            "event_move": make_json_safe(self.event_move),
            "event_substitution": make_json_safe(event_substitution),
            "object_substitution": make_json_safe(object_substitution),
            "event_insertion": make_json_safe(event_insertion),
            "object_insertion": make_json_safe(object_insertion),
            "node_deletion": make_json_safe(self.node_deletion),
        }

        formatted = json.dumps(full_dict, indent=INDENTATION)
        return f"ActionSet<{id(self)}>\n" + indent(formatted, " " * INDENTATION)

    def __copy__(self) -> "ActionSet":
        """
        Make a shallow copy of this object.
        """
        new_obj = type(self)(
            node_attributes_modification={
                k: v for k, v in self.node_attributes_modification.items()
            },
            event_move={k: v for k, v in self.event_move.items()},
            event_substitution={k: v for k, v in self.event_substitution.items()},
            object_substitution={k: v for k, v in self.object_substitution.items()},
            node_deletion={k: v for k, v in self.node_deletion.items()},
            event_insertion={k: v for k, v in self.event_insertion.items()},
            object_insertion={k: v for k, v in self.object_insertion.items()},
        )
        return new_obj

    def __eq__(self, other) -> bool:
        """
        Equality check for ActionSet objects.
        """
        if not isinstance(other, ActionSet):
            return NotImplemented
        return (
            self.node_attributes_modification == other.node_attributes_modification
            and self.event_move == other.event_move
            and self.event_substitution == other.event_substitution
            and self.object_substitution == other.object_substitution
            and self.event_insertion == other.event_insertion
            and self.object_insertion == other.object_insertion
            and self.node_deletion == other.node_deletion
        )

    def undo_changes(self, p: ProcessExecution, record: dict) -> ProcessExecution:
        """
        Revert the changes previously applied by :meth:`apply_changes` using the
        provided record. The process execution `p` is modified in place and
        returned for convenience.
        """
        # Node attribute modifications
        for action, undo_info in record.get("node_attributes", {}).items():
            action.undo_change(p, undo_info)

        # Event moves
        for action, undo_info in record.get("event_move", {}).items():
            action.undo_change(p, undo_info)

        # Event substitutions
        for action, undo_info in record.get("event_substitution", {}).items():
            action.undo_change(p, undo_info)

        # Object substitutions
        for action, undo_info in record.get("object_substitution", {}).items():
            action.undo_change(p, undo_info)

        # Event insertions
        for action, undo_info in record.get("event_insertion", {}).items():
            action.undo_change(p, undo_info)

        # Object insertions
        for action, undo_info in record.get("object_insertion", {}).items():
            action.undo_change(p, undo_info)

        # Deletions (nodes and edges)
        for action, undo_info in record.get("node_deletion", {}).items():
            action.undo_change(p, undo_info)

        return p

    def __ne__(self, other) -> bool:
        """
        Inequality check for ActionSet objects.
        """
        if not isinstance(other, ActionSet):
            return NotImplemented
        return (
            self.node_attributes_modification != other.node_attributes_modification
            or self.event_move != other.event_move
            or self.event_substitution != other.event_substitution
            or self.object_substitution != other.object_substitution
            or self.event_insertion != other.event_insertion
            or self.object_insertion != other.object_insertion
            or self.node_deletion != other.node_deletion
        )

    def _affected_nodes(self, exclude_action: Action = None) -> set:
        """Return the set of node ids already affected by this action set."""
        nodes = set()

        # Node attribute changes
        nodes |= {
            action.node_id
            for action in self.node_attributes_modification.keys()
            if action is not exclude_action
        }

        # Event substitutions
        for action, target_event_id in self.event_move.items():
            if action is exclude_action:
                continue
            nodes.add(action.event_id)
            if target_event_id:
                nodes.add(target_event_id)

        # Event substitutions
        for action, subst in self.event_substitution.items():
            if action is exclude_action:
                continue
            nodes.add(action.event_id)
            if subst:
                nodes.add(subst[0])

        # Object substitutions
        for action, subst in self.object_substitution.items():
            if action is exclude_action:
                continue
            nodes.add(action.object_id)
            if subst:
                nodes.add(subst[0])

        # Event insertion
        for action, insertion in self.event_insertion.items():
            if action is exclude_action:
                continue
            nodes.add(action.event_id)
            if insertion:
                nodes.update(insertion)

        # Object insertion
        for action, insertion in self.object_insertion.items():
            if action is exclude_action:
                continue
            nodes.add(action.event_id)
            if insertion:
                nodes.update(insertion)

        # Node deletions
        for action, deletions in self.node_deletion.items():
            if action is exclude_action:
                continue
            if deletions:
                nodes.update(deletions)

        return nodes

    def is_node_available(self, node_id: Any, exclude_action: Action = None) -> bool:
        """Return True if node is not already involved in a conflicting action."""
        return node_id not in self._affected_nodes(exclude_action=exclude_action)

    def is_change_allowed(
        self, action: Action, change_value: Any, p: ProcessExecution
    ) -> bool:
        """Return False if the node(s) in this change are already modified or do not exist."""
        # Build node set for this specific action change (only pre-existing nodes)
        candidate_nodes = set()

        if isinstance(action, (NodeAttributeNumeric, NodeAttributeCategorical)):
            candidate_nodes.add(action.node_id)

        elif isinstance(action, EventNodeMove):
            candidate_nodes.add(action.event_id)
            if change_value:
                candidate_nodes.add(change_value)

        elif isinstance(action, EventNodeSubstitution):
            candidate_nodes.add(action.event_id)

        elif isinstance(action, ObjectNodeSubstitution):
            candidate_nodes.add(action.object_id)

        elif isinstance(action, (EventNodeDeletion, ObjectNodeDeletion)):
            if change_value:
                candidate_nodes.update(change_value)

        elif isinstance(action, (EventNodeInsertion, ObjectNodeInsertion)):
            # Insertion is anchored on a source event; this event must exist
            candidate_nodes.add(action.event_id)

        else:
            # Unknown action type, can't determine safety; be conservative
            return False

        # Check if all required nodes exist in the graph
        existing_nodes = set(p.nodes)
        if not candidate_nodes.issubset(existing_nodes):
            return False

        # Check for conflicts with other actions
        affected = self._affected_nodes(exclude_action=action)

        if isinstance(action, (NodeAttributeNumeric, NodeAttributeCategorical)):
            # Attribute changes can always co-exist with other changes on the same node
            return True

        # Determine whether any node is already affected by other actions
        return candidate_nodes.isdisjoint(affected)

    def get_change_value(self, action: Action) -> Any | None:
        """
        Get the current change value for a given action.
        Args:
            action (Action): The action for which to get the change value.
        Returns:
            Any: The current change value for the action.
        """
        if isinstance(action, (NodeAttributeNumeric, NodeAttributeCategorical)):
            return self.node_attributes_modification.get(action)
        elif isinstance(action, EventNodeMove):
            return self.event_move.get(action)
        elif isinstance(action, EventNodeSubstitution):
            return self.event_substitution.get(action)
        elif isinstance(action, ObjectNodeSubstitution):
            return self.object_substitution.get(action)
        elif isinstance(action, EventNodeInsertion):
            return self.event_insertion.get(action)
        elif isinstance(action, ObjectNodeInsertion):
            return self.object_insertion.get(action)
        elif isinstance(action, (EventNodeDeletion, ObjectNodeDeletion)):
            return self.node_deletion.get(action)
        else:
            raise NotImplementedError(f"Action of type {type(action)} is not supported")

    def set_change_value(self, action: Action, value: Any) -> None:
        """
        Set the change value for a given action.
        Args:
            action (Action): The action for which to set the change value.
            value (Any): The new value to set for the action.
        """
        if isinstance(action, (NodeAttributeNumeric, NodeAttributeCategorical)):
            self.node_attributes_modification[action] = value
        elif isinstance(action, EventNodeMove):
            self.event_move[action] = value
        elif isinstance(action, EventNodeSubstitution):
            self.event_substitution[action] = value
        elif isinstance(action, ObjectNodeSubstitution):
            self.object_substitution[action] = value
        elif isinstance(action, EventNodeInsertion):
            self.event_insertion[action] = value
        elif isinstance(action, ObjectNodeInsertion):
            self.object_insertion[action] = value
        elif isinstance(action, (EventNodeDeletion, ObjectNodeDeletion)):
            self.node_deletion[action] = value
        else:
            raise NotImplementedError(f"Action of type {type(action)} is not supported")

    def action_size(self) -> int:
        """
        Calculate the total number of changes in the action.
        Returns:
            int: The total number of changes.
        """
        node_attributes_modification_size = sum(
            action.change_size(change_value=change_value)
            for action, change_value in self.node_attributes_modification.items()
        )
        event_move_size = sum(
            action.change_size(target_event_id=target_event_id)
            for action, target_event_id in self.event_move.items()
        )
        event_substitution_size = sum(
            action.change_size(subst_node=subst_event)
            for action, subst_event in self.event_substitution.items()
        )
        object_substitution_size = sum(
            action.change_size(subst_node=subst_obj)
            for action, subst_obj in self.object_substitution.items()
        )
        event_insertion_size = sum(
            action.change_size(change_value)
            for action, change_value in self.event_insertion.items()
        )
        object_insertion_size = sum(
            action.change_size(change_value)
            for action, change_value in self.object_insertion.items()
        )
        deletion_size = sum(
            action.change_size(del_nodes=del_nodes)
            for action, del_nodes in self.node_deletion.items()
        )
        return (
            node_attributes_modification_size
            + event_move_size
            + event_substitution_size
            + object_substitution_size
            + event_insertion_size
            + object_insertion_size
            + deletion_size
        )

    def apply_changes(
        self,
        p: ProcessExecution,
    ) -> Tuple[ProcessExecution, dict]:
        """
        Apply the changes defined in the action to a given process execution.
        A record of the original state is returned to allow undoing the changes.
        Args:
            p (ProcessExecution): The process execution to which the changes will be applied.
        Returns:
            Tuple[ProcessExecution, dict]: The modified process execution and a record
                containing information required to undo the applied changes.
        """

        # record structure holds per-action undo data
        record: dict = {
            "node_attributes": {},
            "event_move": {},
            "event_substitution": {},
            "object_substitution": {},
            "event_insertion": {},
            "object_insertion": {},
            "node_deletion": {},
        }

        for action, value in self.node_attributes_modification.items():
            record["node_attributes"][action] = action.apply_change(p, value)

        for action, target_event_id in self.event_move.items():
            if target_event_id:
                record["event_move"][action] = action.apply_change(p, target_event_id)

        for action, substitution in self.event_substitution.items():
            if substitution:
                record["event_substitution"][action] = action.apply_change(
                    p, substitution
                )

        for action, substitution in self.object_substitution.items():
            if substitution:
                record["object_substitution"][action] = action.apply_change(
                    p, substitution
                )

        for action, deletion in self.node_deletion.items():
            if deletion:
                record["node_deletion"][action] = action.apply_change(p, deletion)

        for action, insertion in self.event_insertion.items():
            if insertion:
                record["event_insertion"][action] = action.apply_change(p, insertion)

        for action, insertion in self.object_insertion.items():
            if insertion:
                record["object_insertion"][action] = action.apply_change(p, insertion)

        return p, record
