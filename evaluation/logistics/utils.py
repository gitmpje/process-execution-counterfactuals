from pm4py import OCEL


def clean_ocel_dataset(ocel: OCEL):

    # Convert timestamp to epoch
    ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

    return ocel
