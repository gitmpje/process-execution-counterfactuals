from typing import Any, Dict, List, Tuple

from process_execution.process_execution import ProcessExecution

from tree_search.feature import (
    EventNodeDeletion,
    Feature,
    NodeAttributeNumeric,
    NodeAttributeCategorical,
    ObjectNodeDeletion,
    ObjectNodeSubstitution,
)

class Action:
    """
    Class representing a set of changes (actions) to be applied to a process execution.
    Attributes:
        node_deletion (List[str]): List of node IDs to remove.
        object_substitution (Dict[ObjectSubstitutions, List[Tuple[Tuple[str, dict], Tuple[str, dict]]]): Dictionary mapping object substitutions options to the selected substitutions.
        node_attributes_modification (Dict[NodeAttributeNumeric, Any]): Dictionary mapping node attribute features to their change value.
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
        object_substitution = {k: v[0] for k, v in self.object_substitution.items()}

        return f"""Action<{id(self)}>
    node_deletion: {self.node_deletion}
    object_substitution: {object_substitution}
    node_attributes_modification: {self.node_attributes_modification}"""

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
        Equality check for Action objects.
        Two Actions are equal if they are the same type and their node_deletion,
        object_substitution, and node_attributes_modification dictionaries are equal.
        """
        if not isinstance(other, Action):
            return NotImplemented
        return (
            self.node_deletion == other.node_deletion
            and self.object_substitution == other.object_substitution
            and self.node_attributes_modification == other.node_attributes_modification
        )

    def __ne__(self, other):
        """
        Inequality check for Action objects.
        Two Actions are not equal if their node_deletion,
        object_substitution, or node_attributes_modification dictionaries are different.
        """
        if not isinstance(other, Action):
            return NotImplemented
        return (
            self.node_deletion != other.node_deletion
            or self.object_substitution != other.object_substitution
            or self.node_attributes_modification != other.node_attributes_modification
        )

    def get_change_value(self, feature: Feature) -> Any | None:
        """
        Get the current change value for a given feature.
        Args:
            feature (Feature): The feature for which to get the change value.
        Returns:
            Any: The current change value for the feature.
        """
        if isinstance(feature, (EventNodeDeletion, ObjectNodeDeletion)):
            return self.node_deletion.get(feature)
        elif isinstance(feature, (NodeAttributeNumeric, NodeAttributeCategorical)):
            return self.node_attributes_modification.get(feature)
        elif isinstance(feature, ObjectNodeSubstitution):
            return self.object_substitution.get(feature)
        else:
            raise NotImplementedError(f"Feature of type {type(feature)} is not supported")

    def set_change_value(self, feature: Feature, value: Any):
        """
        Set the change value for a given feature.
        Args:
            feature (Feature): The feature for which to set the change value.
            value (Any): The new value to set for the feature.
        """
        if isinstance(feature, (EventNodeDeletion, ObjectNodeDeletion)):
            self.node_deletion[feature] = value
        elif isinstance(feature, (NodeAttributeNumeric, NodeAttributeCategorical)):
            self.node_attributes_modification[feature] = value
        elif isinstance(feature, ObjectNodeSubstitution):
            self.object_substitution[feature] = value
        else:
            raise NotImplementedError(f"Feature of type {type(feature)} is not supported")

    def action_size(self) -> int:
        """
        Calculate the total number of changes in the action.
        Returns:
            int: The total number of changes.
        """
        deletion_size = sum(
            feature.change_size(del_nodes=del_nodes)
            for feature, del_nodes in self.node_deletion.items()
        )

        substitution_size = sum(
            feature.change_size(subst_node=subst_obj)
            for feature, subst_obj in self.object_substitution.items()
        )

        node_attributes_modification_size = sum(
            feature.change_size(change_value=change_value)
            for feature, change_value in self.node_attributes_modification.items()
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
    ) -> ProcessExecution:
        """
        Apply the changes defined in the action to a given process execution.
        Args:
            p (ProcessExecution): The process execution to which the changes will be applied.
        Returns:
            ProcessExecution: The modified process execution after applying the changes.
        """

        for feature, value in self.node_attributes_modification.items():
            feature.apply_change(p, value)

        for feature, deletion in self.node_deletion.items():
            if deletion:
                feature.apply_change(p, deletion)

        for feature, substitution in self.object_substitution.items():
            if substitution:
                feature.apply_change(p, substitution)
        return p
