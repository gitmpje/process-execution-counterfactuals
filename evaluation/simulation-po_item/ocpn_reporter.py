import json
import pandas as pd

from datetime import datetime, timedelta
from typing import Any
from simpn.reporters import OutputReporter
from ocpn_prototypes import OCPNVar


def _make_object_id(object_type: str, oid: Any) -> str:
    """Create a globally unique object ID string."""
    return f"{object_type}-{oid}"


def _serialize_attr(value: Any) -> Any:
    """Serialize attribute value for OCEL JSON (handles lists, dicts)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def _timestamp_to_iso(timestamp: float) -> str:
    """Convert simulation timestamp to ISO 8601 format."""
    base = datetime(1970, 1, 1)
    dt = base + timedelta(seconds=float(timestamp))
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


class OCELReporter(OutputReporter):
    """
    An OCEL reporter that writes occurring events to Excel, OCEL 1.0 JSON (pm4py),
    or OCEL 2.0 JSON format.
    Every event is characterized by:
    - event_id: a unique identifier for the event
    - activity: the transition that was fired
    - timestamp: the time when the transition was fired
    - omap: the mapping of object types to set of object ids involved in the transition
    - vmap: mapping of object types to set of object attribute dictionaries (one per oid)
    """

    def __init__(self, model):
        self.model = model
        self.header = ["event_id", "activity", "timestamp"]
        self.event_id = 20260200
        self.object_types = []
        for var in self.model.places:
            if isinstance(var, OCPNVar):
                if var.object_type not in self.object_types:
                    self.object_types.append(var.object_type)
        self.header.extend(self.object_types)
        self.header.extend(
            [str(object_type) + "_vmap" for object_type in self.object_types]
        )
        print(self.header)
        self.ocel_df = pd.DataFrame(columns=self.header)

    def callback(self, input_binding, timestamp, activity, output_binding):
        self.event_id += 1

        omap = {object_type: [] for object_type in self.object_types}
        vmap = {str(object_type) + "_vmap": [] for object_type in self.object_types}

        for var, token in input_binding:
            if str(var).endswith(".queue"):
                # Add all tokens from queues to omap and vmap if they are not put back
                for (
                    token
                ) in token.value:  # loop over the SimTokens in the input binding queue
                    for sim_element, content in reversed(
                        output_binding
                    ):  # find matching output binding and check if the token is not put back
                        if sim_element == var:
                            print(token.value)
                            print(content)
                            if str(token.value) not in [str(t.value) for t in content]:
                                omap[token.value["object_type"]].append(
                                    token.value[str(token.value["object_type"]) + "_id"]
                                )  # add the object id to the omap
                                vmap[str(token.value["object_type"]) + "_vmap"].append(
                                    token.value
                                )  # add the object attributes to the vmap

            else:
                omap[var.object_type].append(
                    token.value[str(token.value["object_type"]) + "_id"]
                )
                vmap[str(var.object_type) + "_vmap"].append(token.value)

        result = (
            [self.event_id, activity, timestamp]
            + list(omap.values())
            + list(vmap.values())
        )
        print(result)
        print("")
        self.ocel_df.loc[len(self.ocel_df)] = result

    def _build_ocel_structures(self) -> tuple[dict, dict]:
        """
        Build events and objects dicts from ocel_df for OCEL export.
        Returns (events_dict, objects_dict).
        objects_dict values include ocel:type, ocel:ovmap, and first_seen (ISO timestamp).
        """
        events: dict = {}
        objects: dict = {}

        for _, row in self.ocel_df.iterrows():
            eid = row["event_id"]
            activity = row["activity"]
            timestamp = row["timestamp"]
            ts_iso = _timestamp_to_iso(timestamp)
            flat_omap: list[str] = []

            for ot in self.object_types:
                omap_ids = row.get(ot)
                vmap_list = row.get(f"{ot}_vmap")

                if omap_ids is not None and vmap_list is not None:
                    ids = omap_ids if isinstance(omap_ids, list) else [omap_ids]
                    attrs_list = (
                        vmap_list if isinstance(vmap_list, list) else [vmap_list]
                    )

                    for i, oid in enumerate(ids):
                        obj_id = _make_object_id(ot, oid)
                        flat_omap.append(obj_id)

                        if obj_id not in objects:
                            ovmap = {}
                            if i < len(attrs_list):
                                d = attrs_list[i]
                                if isinstance(d, dict):
                                    for k, v in d.items():
                                        if k != "object_type":
                                            ovmap[k] = _serialize_attr(v)
                            objects[obj_id] = {
                                "ocel:type": ot,
                                "ocel:ovmap": ovmap,
                                "first_seen": ts_iso,
                            }

            events[f"e{eid}"] = {
                "ocel:activity": str(activity),
                "ocel:timestamp": ts_iso,
                "ocel:omap": flat_omap,
                "ocel:vmap": {},
            }

        return events, objects

    def save_report(self, path: str = "ocel_report.xlsx") -> None:
        """Save the event log to Excel (backward compatible)."""
        print(self.ocel_df)
        print(f"saved to {path}")
        self.ocel_df.to_excel(path, index=False)

    def save_jsonocel(self, path: str = "ocel_report.jsonocel") -> None:
        """
        Export to OCEL 1.0 JSON format for pm4py.
        Use file extension .jsonocel for pm4py.read.read_ocel().
        """
        events, objects_raw = self._build_ocel_structures()

        objects = {
            oid: {"ocel:type": o["ocel:type"], "ocel:ovmap": o["ocel:ovmap"]}
            for oid, o in objects_raw.items()
        }

        ocel10 = {
            "ocel:global-log": {
                "ocel:version": "1.0",
                "ocel:attribute-names": [],
                "ocel:object-types": list(self.object_types),
                "ocel:ordering": "timestamp",
            },
            "ocel:global-event": {"ocel:activity": "__INVALID__"},
            "ocel:global-object": {"ocel:type": "__INVALID__"},
            "ocel:events": events,
            "ocel:objects": objects,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(ocel10, f, indent=2)
        print(f"OCEL 1.0 (pm4py) saved to {path}")

    def save_ocel20(self, path: str = "ocel_report.json") -> None:
        """
        Export to OCEL 2.0 JSON format.
        Compatible with ocel-standard.org and tools supporting OCEL 2.0.
        """
        events, objects_raw = self._build_ocel_structures()

        event_types: dict[str, dict] = {}
        for e in events.values():
            act = e["ocel:activity"]
            if act not in event_types:
                event_types[act] = {"name": act, "attributes": []}

        object_types: dict[str, dict] = {}
        for o in objects_raw.values():
            ot = o["ocel:type"]
            if ot not in object_types:
                attrs = [{"name": k, "type": "string"} for k in o["ocel:ovmap"].keys()]
                object_types[ot] = {"name": ot, "attributes": attrs}

        ocel20_events = []
        for eid, e in events.items():
            relationships = [
                {"objectId": oid, "qualifier": "involved"} for oid in e["ocel:omap"]
            ]
            ocel20_events.append(
                {
                    "id": eid,
                    "type": e["ocel:activity"],
                    "time": e["ocel:timestamp"],
                    "attributes": [],
                    "relationships": relationships,
                }
            )

        ocel20_objects = []
        for oid, o in objects_raw.items():
            ts = o.get("first_seen", "1970-01-01T00:00:00")
            attrs = [
                {"name": k, "value": str(v), "time": ts}
                for k, v in o["ocel:ovmap"].items()
            ]
            ocel20_objects.append(
                {
                    "id": oid,
                    "type": o["ocel:type"],
                    "attributes": attrs,
                    "relationships": [],
                }
            )

        ocel20 = {
            "eventTypes": list(event_types.values()),
            "objectTypes": list(object_types.values()),
            "events": ocel20_events,
            "objects": ocel20_objects,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(ocel20, f, indent=2)
        print(f"OCEL 2.0 saved to {path}")
