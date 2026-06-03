from networkx import MultiDiGraph
from pandas import DataFrame
from pm4py import convert_ocel_to_networkx, convert_log_to_ocel


def convert_event_log_ocel(event_log: DataFrame, viewpoint: str):
    ocel = convert_log_to_ocel(event_log)

    case_columns = [col for col in event_log.columns if col.startswith("case:")]
    df_object = event_log[case_columns].drop_duplicates(keep="last")
    for col in df_object.columns:
        # Convert columns to string type
        if ("Age" not in col) and ("date" not in col):
            df_object[col] = df_object[col].astype(str)
    df_object.set_index(viewpoint, drop=True, inplace=True)

    ocel.objects.set_index("ocel:oid", drop=False, inplace=True)
    ocel.objects = ocel.objects.join(df_object)

    df_event = event_log[[col for col in event_log.columns if col not in case_columns]]
    df_event.drop(columns=["concept:name", "time:timestamp"], inplace=True)
    ocel.events = ocel.events.join(df_event)

    # Convert timestamp to epoch
    ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

    for col in ocel.events.columns:
        ocel.events[col] = ocel.events[col].fillna(0)

    ocel_nx = MultiDiGraph(convert_ocel_to_networkx(ocel))

    return ocel, ocel_nx
