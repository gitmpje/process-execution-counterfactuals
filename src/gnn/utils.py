from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Metadata:
    viewpoint: Any
    node_num_keys: Any
    activities: Any
    node_types: Any
    edge_types: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint": self.viewpoint,
            "node_num_keys": self.node_num_keys,
            "activities": self.activities,
            "node_types": self.node_types,
            "edge_types": self.edge_types,
        }

    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> "Metadata":
        """
        Create a Metadata instance from a dictionary.
        Accepts keys "node_types_set" and "edge_types_set" as in your original logic.
        """
        return cls(
            viewpoint=metadata_dict["viewpoint"],
            node_num_keys=metadata_dict["node_num_keys"],
            activities=metadata_dict["activities"],
            node_types=metadata_dict.get("node_types", metadata_dict.get("node_types_set")),
            edge_types=metadata_dict.get("edge_types", metadata_dict.get("edge_types_set")),
        )
