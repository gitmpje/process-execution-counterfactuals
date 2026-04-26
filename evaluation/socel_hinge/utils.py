from pm4py import OCEL


def clean_ocel_dataset(ocel: OCEL):
    for table in [ocel.objects, ocel.events]:
        for col in table.columns:
            try:
                if table[col].dtype == "object" or table[col].dtype.name == "str":
                    table[col] = table[col].str.replace("~", "")
                table[col] = table[col].replace("?", 0).astype(float)
                # table.fillna({col: 0}, inplace=True)
            except (ValueError, TypeError) as e:
                print("Skipped", col, "due to error:", e)

    # Convert timestamp to epoch
    ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

    return ocel
