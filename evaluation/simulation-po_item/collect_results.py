# %%
import pandas as pd
import re
from pathlib import Path

from typing import List

nested_keys = [
    "dist_x",
    "dist_a",
    "proximity_x",
    "proximity_a",
    "evaluation_valid",
]
node_feat_edit_threshold = 0.05


def average_metric(dict_list, key):
    if isinstance(dict_list, list) and key in dict_list[0]:
        return sum(d.get(key) for d in dict_list) / len(dict_list)
    else:
        return None


def parse_edits(edits_str: str) -> List[str]:
    """
    Parse an edits string like '[AddEdge(...), ChangeNodeFeat(...), ...]'
    into a list of individual edit strings.
    """
    # Remove surrounding brackets
    inner = edits_str.strip()[1:-1]

    # Split on commas that are followed by a new edit (capital letter)
    edits = re.split(r",\s*(?=[A-Z])", inner)
    return [e.strip() for e in edits]


def check_edit(node_feat_edit_str: str, threshold: float):
    """
    Example: ChangeNodeFeat(node=0, feat=1, 1.000 → 0.000)
    """
    if "ChangeNodeFeat" not in node_feat_edit_str:
        return True

    pattern = r"([-+]?\d*\.?\d+)\s*→\s*([-+]?\d*\.?\d+)"
    match = re.search(pattern, node_feat_edit_str)

    if not match:
        raise ValueError("Could not parse numeric values from input string.")

    before, after = map(float, match.groups())
    return abs(after - before) > threshold


# %%
run_data = []
result_dir = Path("results")


def find_result_files(scenario_prefixes, result_dir=result_dir):
    files = []
    for scenario_prefix in scenario_prefixes:
        files.extend(sorted(result_dir.glob(f"{scenario_prefix}-*.json")))
    return files


def parse_scenario_method_run(file_name):
    base = Path(file_name).stem
    match = re.fullmatch(r"(scenario_[0-9]+)-(.+?)(?:-run(\d+))?", base)
    if match:
        return match.group(1), match.group(2), match.group(3)
    scenario, method = parse_scenario_method(file_name)
    return scenario, method, None


for path in find_result_files(
    [
        "scenario_01",
        "scenario_02",
        "scenario_03",
        "scenario_04",
        "scenario_05",
        "scenario_06",
        "scenario_07",
        "scenario_08",
    ]
):
    scenario, method, run = parse_scenario_method_run(path.name)
    try:
        df = pd.read_json(path)
    except FileNotFoundError:
        print(path)
        continue

    for key in nested_keys:
        df[key] = df["proximity_metrics"].apply(lambda x: x.get(key))
        df[f"proximity_all-{key}"] = df["proximity_metrics_all"].apply(
            lambda x: average_metric(list(x.values()), key)
        )

    if "CLEAR" in path.name:
        df["evaluation_valid"] = df["evaluation_valid"].astype(int)
    else:
        df["evaluation_valid"] = 1

    try:
        df["n_changes"] = df["edits"].apply(
            lambda x: sum(
                check_edit(edit, threshold=node_feat_edit_threshold)
                for edit in parse_edits(x)
            )
        )
    except KeyError:
        df["n_changes"] = float("nan")

    run_data.append(
        {
            "scenario": scenario,
            "method": method,
            "run": run,
            "df": df,
        }
    )


# %%
# Table with mean (std) for different metrics
# scenario (scenario_prefix) | method (map file after scenario_prefix (e.g. hetero-node)) | validity (evaluation_valid, None=1) | proximity_x | proximity_a | proximity_all-proximity_x | proximity_all-proximity_a |
def parse_scenario_method(file_path):
    base = file_path.rsplit("/", 1)[-1]
    if base.endswith(".json"):
        base = base[:-5]
    if "-" not in base:
        return base, None
    scenario, method = base.split("-", 1)
    return scenario, method


def format_metric(series):
    mean = series.mean()
    if pd.isna(mean):
        return None
    std = series.std(ddof=0)
    return f"{mean:.2f} ({std:.2f})"


def build_summary_dataframe(run_data):
    run_rows = []
    for entry in run_data:
        df = entry["df"]
        run_rows.append(
            {
                "scenario": entry["scenario"],
                "method": entry["method"],
                "run": entry["run"],
                "validity_mean": df["evaluation_valid"].mean(),
                "n_changes_mean": df["n_changes"].mean(),
                "proximity_x_mean": df["proximity_x"].mean(),
                "proximity_a_mean": df["proximity_a"].mean(),
                "proximity_all-proximity_x_mean": df[
                    "proximity_all-proximity_x"
                ].mean(),
                "proximity_all-proximity_a_mean": df[
                    "proximity_all-proximity_a"
                ].mean(),
            }
        )

    run_summary = pd.DataFrame(run_rows)
    if run_summary.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "method",
                "validity",
                "n_changes",
                "proximity_x",
                "proximity_a",
                "proximity_all-proximity_x",
                "proximity_all-proximity_a",
            ]
        )

    summary_rows = []
    for (scenario, method), group in run_summary.groupby(["scenario", "method"]):
        summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "validity": format_metric(group["validity_mean"]),
                "n_changes": format_metric(group["n_changes_mean"]),
                "proximity_x": format_metric(group["proximity_x_mean"]),
                "proximity_a": format_metric(group["proximity_a_mean"]),
                "proximity_all-proximity_x": format_metric(
                    group["proximity_all-proximity_x_mean"]
                ),
                "proximity_all-proximity_a": format_metric(
                    group["proximity_all-proximity_a_mean"]
                ),
            }
        )

    return pd.DataFrame(
        summary_rows,
        columns=[
            "scenario",
            "method",
            "validity",
            "n_changes",
            "proximity_x",
            "proximity_a",
            "proximity_all-proximity_x",
            "proximity_all-proximity_a",
        ],
    )


def parse_formatted_metric(value):
    if not isinstance(value, str):
        return None
    try:
        return float(value.split("(")[0].strip())
    except ValueError:
        return None


def bold_by_scenario(df, metric_columns, metric_columns_min):
    bold_df = df.copy()
    for _, group in df.groupby("scenario"):
        for column in metric_columns:
            means = group[column].map(parse_formatted_metric)
            if means.isna().all():
                continue
            if column in metric_columns_min:
                mean_min_max = means.min()
            else:
                mean_min_max = means.max()
            for idx, mean in means.items():
                if pd.notna(mean) and mean == mean_min_max:
                    value = bold_df.at[idx, column]
                    if isinstance(value, str):
                        bold_df.at[idx, column] = r"\textbf{" + value + r"}"
    return bold_df


def summary_df_to_latex(df):
    metric_columns = [
        "validity",
        "n_changes",
        "proximity_x",
        "proximity_a",
        "proximity_all-proximity_x",
        "proximity_all-proximity_a",
    ]
    metric_columns_min = [
        "n_changes",
    ]
    latex_df = bold_by_scenario(df, metric_columns, metric_columns_min)

    header = "scenario & method & validity & n_changes & proximity_x & proximity_a & proximity_all-proximity_x & proximity_all-proximity_a \\\\"
    lines = ["\\begin{tabular}{cl||rrrrrr}", "\\toprule", header]
    scenario_number = 0
    for _, row in latex_df.iterrows():
        scenario_number_prev = scenario_number
        scenario_number = int(row["scenario"].replace("scenario_", ""))
        if scenario_number != scenario_number_prev:
            lines.append("\\hline")

        cells = [
            str(scenario_number),
            str(row["method"]),
            str(row["validity"]),
            str(row["n_changes"]),
            str(row["proximity_x"]),
            str(row["proximity_a"]),
            str(row["proximity_all-proximity_x"]),
            str(row["proximity_all-proximity_a"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


summary_df = build_summary_dataframe(run_data)

summary_df_filtered = summary_df[
    ~summary_df["method"].isin(["hetero-depth-first=node"])
]
print(summary_df_to_latex(summary_df_filtered))
