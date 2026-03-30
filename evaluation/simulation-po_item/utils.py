from networkx import Graph


def _replace_scenario_prefix(item: dict | list | str, scenario_prefix: str):
    if isinstance(item, str):
        return item.replace("$SCENARIO_PREFIX", scenario_prefix)
    if isinstance(item, dict):
        return {
            k: _replace_scenario_prefix(v, scenario_prefix) for k, v in item.items()
        }
    if isinstance(item, list):
        return [_replace_scenario_prefix(v, scenario_prefix) for v in item]
    return item


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
