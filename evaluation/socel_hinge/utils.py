from pm4py import OCEL


def clean_ocel_dataset(ocel: OCEL):
    for table in [ocel.objects, ocel.events]:
        for col in table.columns:
            try:
                table[col] = table[col].replace("?", 0).astype(float)
                # table.fillna({col: 0}, inplace=True)
            except (ValueError, TypeError):
                print("Skipped", col)

    # Convert timestamp to epoch
    ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

    return ocel
