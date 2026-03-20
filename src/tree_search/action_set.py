import json

from typing import Any, Dict, List, Tuple

from process_execution.process_execution import ProcessExecution
from tree_search.action import (
    EventNodeDeletion,
    Action,
    NodeAttributeNumeric,
    NodeAttributeCategorical,
    ObjectNodeDeletion,
    ObjectNodeSubstitution,
)

INDENTATION = 2


def make_json_safe(obj):
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
        node_deletion: Dict[EventNodeDeletion, List[str]] = None,
        object_substitution: Dict[
            ObjectNodeSubstitution, Tuple[str, dict] | None
        ] = None,
        node_attributes_modification: Dict[NodeAttributeNumeric, Any] = None,
    ):
        self.node_deletion = node_deletion if node_deletion else {}
        self.object_substitution = object_substitution if object_substitution else {}
        self.node_attributes_modification = (
            node_attributes_modification if node_attributes_modification else {}
        )

    def __repr__(self):
        # Pre-process substitution dict: original code extracts first element of tuple/list
        object_substitution = {
            k: make_json_safe(v[0]) for k, v in self.object_substitution.items()
        }

        return (
            f"ActionSet<{id(self)}>\n"
            f"  node_deletion: "
            f"{json.dumps(make_json_safe(self.node_deletion), indent=INDENTATION)}\n"
            f"  object_substitution: "
            f"{json.dumps(object_substitution, indent=INDENTATION)}\n"
            f"  node_attributes_modification: "
            f"{json.dumps(make_json_safe(self.node_attributes_modification), indent=INDENTATION)}"
        )

    def __copy__(self):
        """
        Make a shallow copy of this object.
        """
        new_obj = type(self)(
            node_deletion={k: v for k, v in self.node_deletion.items()},
            object_substitution={k: v for k, v in self.object_substitution.items()},
            node_attributes_modification={
                k: v for k, v in self.node_attributes_modification.items()
            },
        )
        return new_obj

    def __eq__(self, other):
        """
        Equality check for ActionSet objects.
        Two ActionSets are equal if they are the same type and their node_deletion,
        object_substitution, and node_attributes_modification dictionaries are equal.
        """
        if not isinstance(other, ActionSet):
            return NotImplemented
        return (
            self.node_deletion == other.node_deletion
            and self.object_substitution == other.object_substitution
            and self.node_attributes_modification == other.node_attributes_modification
        )

    def undo_changes(self, p: ProcessExecution, record: dict) -> ProcessExecution:
        """
        Revert the changes previously applied by :meth:`apply_changes` using the
        provided record. The process execution `p` is modified in place and
        returned for convenience.
        """
        # restore node attribute modifications
        for action, undo_info in record.get("node_attributes", {}).items():
            action.undo_change(p, undo_info)

        # restore deletions (nodes and edges)
        for action, undo_info in record.get("node_deletion", {}).items():
            action.undo_change(p, undo_info)

        # restore object substitutions
        for action, undo_info in record.get("object_substitution", {}).items():
            action.undo_change(p, undo_info)

        return p

    def __ne__(self, other):
        """
        Inequality check for ActionSet objects.
        Two ActionSets are not equal if their node_deletion,
        object_substitution, or node_attributes_modification dictionaries are different.
        """
        if not isinstance(other, ActionSet):
            return NotImplemented
        return (
            self.node_deletion != other.node_deletion
            or self.object_substitution != other.object_substitution
            or self.node_attributes_modification != other.node_attributes_modification
        )

    def get_change_value(self, action: Action) -> Any | None:
        """
        Get the current change value for a given action.
        Args:
            action (Action): The action for which to get the change value.
        Returns:
            Any: The current change value for the action.
        """
        if isinstance(action, (EventNodeDeletion, ObjectNodeDeletion)):
            return self.node_deletion.get(action)
        elif isinstance(action, (NodeAttributeNumeric, NodeAttributeCategorical)):
            return self.node_attributes_modification.get(action)
        elif isinstance(action, ObjectNodeSubstitution):
            return self.object_substitution.get(action)
        else:
            raise NotImplementedError(f"Action of type {type(action)} is not supported")

    def set_change_value(self, action: Action, value: Any):
        """
        Set the change value for a given action.
        Args:
            action (Action): The action for which to set the change value.
            value (Any): The new value to set for the action.
        """
        if isinstance(action, (EventNodeDeletion, ObjectNodeDeletion)):
            self.node_deletion[action] = value
        elif isinstance(action, (NodeAttributeNumeric, NodeAttributeCategorical)):
            self.node_attributes_modification[action] = value
        elif isinstance(action, ObjectNodeSubstitution):
            self.object_substitution[action] = value
        else:
            raise NotImplementedError(f"Action of type {type(action)} is not supported")

    def action_size(self) -> int:
        """
        Calculate the total number of changes in the action.
        Returns:
            int: The total number of changes.
        """
        deletion_size = sum(
            action.change_size(del_nodes=del_nodes)
            for action, del_nodes in self.node_deletion.items()
        )

        substitution_size = sum(
            action.change_size(subst_node=subst_obj)
            for action, subst_obj in self.object_substitution.items()
        )

        node_attributes_modification_size = sum(
            action.change_size(change_value=change_value)
            for action, change_value in self.node_attributes_modification.items()
        )
        return deletion_size + substitution_size + node_attributes_modification_size

    def objective_value(self) -> int:
        """
        Calculate the objective value of the action, defined as the total number changes.
        Returns:
            int: The objective value of the action.
        """
        substitutions = [
            (k.object_id, v[0]) for k, v in self.object_substitution.items() if v
        ]

        return (
            len(self.node_deletion)
            + len(
                [
                    obj_id
                    for obj_id, subst_obj_id in substitutions
                    if obj_id != subst_obj_id
                ]
            )
            + len([k for k, v in self.node_attributes_modification.items() if v != 0])
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
            "node_deletion": {},
            "object_substitution": {},
        }

        for action, value in self.node_attributes_modification.items():
            record["node_attributes"][action] = action.apply_change(p, value)

        for action, deletion in self.node_deletion.items():
            if deletion:
                record["node_deletion"][action] = action.apply_change(p, deletion)

        for action, substitution in self.object_substitution.items():
            if substitution:
                record["object_substitution"][action] = action.apply_change(
                    p, substitution
                )
        return p, record
