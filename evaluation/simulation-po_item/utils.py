import torch

from networkx import Graph
from torch_geometric.data import Data, HeteroData

from typing import List, Tuple

from evaluation.clear_graph_cfe import (
    GraphCFEDatasetStats,
)
from gnn.utils import Metadata, to_homogeneous_data


def replace_scenario_prefix(item: dict | list | str, scenario_prefix: str):
    if isinstance(item, str):
        return item.replace("$SCENARIO_PREFIX", scenario_prefix)
    if isinstance(item, dict):
        return {k: replace_scenario_prefix(v, scenario_prefix) for k, v in item.items()}
    if isinstance(item, list):
        return [replace_scenario_prefix(v, scenario_prefix) for v in item]
    return item


def to_homogeneous(
    dataset: List[HeteroData],
    metadata: Metadata,
) -> Tuple[List[Data], GraphCFEDatasetStats]:
    homogeneous_dataset = []
    labels = []
    for data in dataset:
        homogeneous_dataset.append(
            to_homogeneous_data(
                data,
                metadata.node_num_keys,
                metadata.node_cat_keys,
                metadata.node_types,
                metadata.one_hot_encoding,
                metadata.unique_node_type_attribute_columns,
            )
        )
        labels.append(torch.tensor([data.y], dtype=torch.long))

    return homogeneous_dataset, labels


def visualize_process_execution(
    process_execution: Graph,
    output_file_name: str,
):
    from networkx import nx_agraph
    from process_execution.process_execution import ProcessExecution
    from process_execution.visualization import (
        apply_node_styles_nx,
        apply_edge_styles_nx,
    )

    process_execution = ProcessExecution(process_execution)
    process_execution.construct_node_label()
    process_execution.construct_edge_label()

    apply_node_styles_nx(process_execution)
    apply_edge_styles_nx(process_execution)

    agraph = nx_agraph.to_agraph(process_execution)
    agraph.draw(output_file_name, prog="dot")
