from networkx import MultiDiGraph
from pandas import DataFrame
from pm4py import convert_ocel_to_networkx, convert_log_to_ocel


def convert_event_log_ocel(event_log: DataFrame):
    object_types = ["case:concept:name", "org:resource"]
    event_log["org:resource"] = event_log["org:resource"] + "_org:resource"

    # Remove Payment event
    event_log_filtered = event_log[event_log["concept:name"] != "Payment"]

    ocel = convert_log_to_ocel(
        event_log_filtered,
        object_types=object_types,
        additional_event_attributes=[
            col
            for col in event_log.columns
            if col
            not in object_types
            + [
                "concept:name",
                "lifecycle:transition",
                "time:timestamp",
            ]
        ],
    )

    # Convert timestamp to epoch
    ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

    for col in ocel.events.columns:
        ocel.events[col] = ocel.events[col].fillna(0)

    ocel_nx = MultiDiGraph(convert_ocel_to_networkx(ocel))

    return ocel, ocel_nx
