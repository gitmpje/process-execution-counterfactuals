from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Metadata:
    viewpoint: str
    node_num_keys: Dict[str, Dict[str, List[str]]]
    activities: List[str]
    node_types: List[str]
    edge_types: List[str]
    feat_label_dict: Dict[str, List[str]]
    node_label_dict: Dict[str, List[str]]


    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint": self.viewpoint,
            "node_num_keys": self.node_num_keys,
            "activities": self.activities,
            "node_types": self.node_types,
            "edge_types": self.edge_types,
            "feat_label_dict": self.feat_label_dict,
            "node_label_dict": self.node_label_dict,
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
            node_types=metadata_dict.get("node_types"),
            edge_types=metadata_dict.get("edge_types"),
            feat_label_dict=metadata_dict.get("feat_label_dict"),
            node_label_dict=metadata_dict.get("node_label_dict"),
        )
