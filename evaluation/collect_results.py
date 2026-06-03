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
# Configuration: specify which child folders to process
CHILD_FOLDERS = [
    "sepsis",
    # "road_traffic_fine_management",
    # "simulation-po_item",
    # "socel_hinge",
]
SCENARIO_PREFIXES = {
    "simulation-po_item": [
        "scenario_01",
        "scenario_02",
        "scenario_03",
        "scenario_04",
        "scenario_05",
        "scenario_06",
        "scenario_07",
        "scenario_08",
    ],
}
BASE_DIR = Path(".")

run_data = []


def find_result_files(child_folder, scenario_prefixes=None):
    """Find result JSON files in a child folder.

    Args:
        child_folder: Name of the child folder (e.g., 'simulation-po_item')
        scenario_prefixes: List of scenario prefixes to filter by, or None to load all

    Returns:
        List of Path objects to result files
    """
    result_dir = BASE_DIR / child_folder / "results"
    if not result_dir.exists():
        print(f"Warning: {result_dir} does not exist")
        return []

    files = []
    if scenario_prefixes:
        for scenario_prefix in scenario_prefixes:
            files.extend(sorted(result_dir.glob(f"{scenario_prefix}-*.json")))
    else:
        files.extend(sorted(result_dir.glob("*.json")))
    return files


def parse_scenario_method(file_path):
    base = file_path.rsplit("/", 1)[-1]
    if base.endswith(".json"):
        base = base[:-5]
    if "-" not in base:
        return base, None
    scenario, method = base.split("-", 1)
    return scenario, method


def parse_scenario_method_run(file_name):
    base = Path(file_name).stem
    match = re.fullmatch(r"(scenario_[0-9]+)-(.+?)(?:-run(\d+))?", base)
    if match:
        return match.group(1), match.group(2), match.group(3)
    scenario, method = parse_scenario_method(file_name)
    return scenario, method, None


for child_folder in CHILD_FOLDERS:
    scenario_prefixes = SCENARIO_PREFIXES.get(child_folder)
    for path in find_result_files(child_folder, scenario_prefixes):
        scenario, method, run = parse_scenario_method_run(path.name)
        try:
            df = pd.read_json(path)
        except FileNotFoundError:
            print(path)
            continue

        for key in nested_keys:
            if key in df.columns:
                print(
                    f"Skipping key '{key}' in file {path} because it is already a column"
                )
                continue
            df[key] = df["proximity_metrics"].apply(
                lambda x: x.get(key) if x else float("nan")
            )
            df[f"proximity_all-{key}"] = df["proximity_metrics_all"].apply(
                lambda x: average_metric(list(x.values()), key) if x else float("nan")
            )

        df.fillna({"evaluation_valid": 1.0}, inplace=True)
        df["evaluation_valid"] = df["evaluation_valid"].astype(float)

        try:
            df["n_changes"] = df["edits"].apply(
                lambda x: (
                    sum(
                        check_edit(edit, threshold=node_feat_edit_threshold)
                        for edit in parse_edits(x)
                    )
                    if isinstance(x, str)
                    else float("nan")
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

# run_data = [r for r in run_data if not r["run"]]


# %%
# Table with mean (std) for different metrics
# scenario (scenario_prefix) | method (map file after scenario_prefix (e.g. hetero-node)) | validity (evaluation_valid, None=1) | proximity_x | proximity_a | proximity_all-proximity_x | proximity_all-proximity_a |
def format_metric(series):
    mean = series.mean()
    if pd.isna(mean):
        return None
    std = series.std(ddof=0)
    std_str = f"({std:.2f})" if len(series) > 1 else ""
    return f"{mean:.2f}" + std_str


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
    scenario = ""
    for _, row in latex_df.iterrows():
        scenario_prev = scenario
        scenario = row["scenario"]
        if "scenario_" in scenario:
            scenario = scenario.replace("scenario_", "")
        if scenario != scenario_prev:
            lines.append("\\hline")

        cells = [
            str(scenario),
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


# %% Graph statistics
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from typing import List, Dict, Any


def aggregate_hetero_stats(graphs: List[HeteroData]) -> Dict[str, Any]:
    """
    Aggregates statistics for a list of HeteroData objects.

    Args:
        graphs (List[HeteroData]): List of PyTorch Geometric HeteroData objects.

    Returns:
        Dict[str, Any]: Aggregated statistics.
    """
    if not graphs:
        raise ValueError("The list of graphs is empty.")

    # Initialize counters
    stats = {
        "total_graphs": len(graphs),
        "node_counts": defaultdict(int),
        "edge_counts": defaultdict(int),
        "node_feature_dims": {},
        "edge_feature_dims": {},
    }

    for g in graphs:
        if not isinstance(g, HeteroData):
            raise TypeError("All elements must be HeteroData objects.")

        # Count nodes per type
        for node_type in g.node_types:
            num_nodes = g[node_type].num_nodes
            stats["node_counts"][node_type] += num_nodes

            # Record feature dimension if available
            if "x" in g[node_type]:
                stats["node_feature_dims"][node_type] = g[node_type].x.size(1)

        # Count edges per type
        for edge_type in g.edge_types:
            num_edges = g[edge_type].edge_index.size(1)
            stats["edge_counts"][edge_type] += num_edges

            # Record edge feature dimension if available
            if "edge_attr" in g[edge_type]:
                stats["edge_feature_dims"][edge_type] = g[edge_type].edge_attr.size(1)

    return stats


report_stats = []
for data_file in [
    "sepsis/data/dataset-pe.pt",
    # "road_traffic_fine_management/data/dataset-pe.pt",
    # "socel_hinge/data/dataset-pe-MalePart.pt",
    # "socel_hinge/data/dataset-pe-HingePack.pt",
]:
    dataset = torch.load(data_file, weights_only=False)
    print(f"Calculating statistics for {data_file}")
    numbers = aggregate_hetero_stats(dataset)

    stats = {"data_file": data_file}
    stats["avg_event_nodes"] = numbers["node_counts"]["EVENT"] / numbers["total_graphs"]

    n_objects = sum(
        numbers["node_counts"][nt] for nt in numbers["node_counts"] if nt != "EVENT"
    )
    stats["avg_object_nodes"] = n_objects / numbers["total_graphs"]

    report_stats.append(stats)

# %% Attribute statistics
import json

from gnn.utils import Metadata

for metadata_file in [
    "sepsis/data/metadata-pe.json",
    # "road_traffic_fine_management/data/metadata-pe.json",
    # "socel_hinge/data/metadata-pe-MalePart.json",
    # "socel_hinge/data/metadata-pe-HingePack.json",
]:
    with open(metadata_file, "r") as f:
        metadata_dict = json.load(f)
    metadata = Metadata.from_dict(metadata_dict)

    print(metadata_file)
    print("EA:", sum(len(v) for v in metadata.node_cat_keys["EVENT"].values()))
    print("EO:", sum(len(v) for v in metadata.node_cat_keys["OBJECT"].values()))
