# %% Import dependencies
import json
import os
import pm4py
import torch
import yaml

from copy import deepcopy
from networkx import Graph
from random import seed

from tree_search.action_helpers import (
    build_node_attribute_actions,
    build_object_substitution_actions,
    build_event_move_actions,
    build_event_insertion_actions,
    build_object_insertion_actions,
    build_node_deletion_actions,
    construct_attribute_spec_dict,
    get_nodes_by_importance,
    get_feature_labels_by_importance,
)
from tree_search.action_set import ActionSet
from tree_search.tree_search import TreeSearchCounterFactual

from evaluation.metrics import compute_proximity, matched_diff_to_edits
from evaluation.utils import get_dense_representation

from gnn.hetero_graph_data import build_hetero_data, to_homogeneous_data
from gnn.utils import Metadata
from gnn.explanation import generate_explanation
from process_execution.process_execution import extract_process_execution

from utils import (
    replace_scenario_prefix,
    to_homogeneous,
    visualize_process_execution,
)

verbose = False
visualize = False

### Configuration ###
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_file) as f:
    cfg = yaml.safe_load(f)

# Replace $SCENARIO_PREFIX tokens in config
SCENARIO_PREFIX = os.environ.get("SCENARIO_PREFIX", "scenario_01")
if SCENARIO_PREFIX is not None:
    cfg = replace_scenario_prefix(cfg, SCENARIO_PREFIX)

# Dataset
dataset_cfg = cfg["dataset"]
path_ocel = dataset_cfg["path_ocel"]
path_labels = dataset_cfg["path_labels"]
path_dataset = dataset_cfg["path_dataset"]
path_metadata = dataset_cfg["path_metadata"]
normalize = dataset_cfg.get("normalize", False)
one_hot_encoding = dataset_cfg.get("one_hot_encoding", False)
add_reverse_edges = dataset_cfg.get("add_reverse_edges", False)

# Process execution
process_execution_cfg = cfg["process_execution"]
viewpoint = process_execution_cfg["viewpoint"]
process_execution_object_types = process_execution_cfg["object_types"]
process_execution_target_activity = process_execution_cfg.get("target_activity")
trace_backward = process_execution_cfg.get("trace_backward", False)

# GNN
gnn_cfg = cfg["gnn"]
path_model = gnn_cfg["path_model"]
homogeneous = gnn_cfg.get("homogeneous", False)
random_seed = gnn_cfg.get("random_seed", 0)

torch.manual_seed(random_seed)
seed(random_seed)

# Counterfactual search
counterfactual_cfg = cfg["counterfactual"]
viewpoint_label = counterfactual_cfg.get("viewpoint_label")
depth_first = counterfactual_cfg.get("depth_first")
num_bins = counterfactual_cfg["num_bins"]
max_change_size = counterfactual_cfg["max_change_size"]
node_importance_threshold = counterfactual_cfg["node_importance_threshold"]
attr_importance_threshold = counterfactual_cfg["attr_importance_threshold"]

# Load metadata
with open(path_metadata, "r") as f:
    metadata_dict = json.load(f)
metadata = Metadata.from_dict(metadata_dict)

# %% Load OCEL
ocel = pm4py.read_ocel2_json(path_ocel)

# Convert timestamp to epoch
ocel.events["epoch"] = ocel.events["ocel:timestamp"].astype(int)

# %% Convert OCEL to Networkx graph
ocel_nx = pm4py.convert_ocel_to_networkx(ocel)

# %% Load model and define process outcome function
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(path_model, weights_only=False)
model = model.to(device)
model.eval()

dataset = torch.load(path_dataset, weights_only=False)
homogeneous_dataset, labels = to_homogeneous(dataset, metadata)


@torch.no_grad()
def process_outcome(p: Graph) -> bool:
    """Predict the outcome for a single `ProcessExecution` using the loaded GNN model.

    Args:
        ocel_graph (Graph): The process execution to classify.
    Returns:
        float: The predicted value.
    """
    try:
        hetero_data, _, _, _, _, _ = build_hetero_data(
            graph=p,
            node_num_keys=metadata.node_num_keys,
            node_cat_keys=metadata.node_cat_keys,
            object_type_col=ocel.object_type_column,
            event_activity_col=ocel.event_activity,
            viewpoint=metadata.viewpoint,
            normalize=metadata.normalized,
            one_hot_encoding=metadata.one_hot_encoding,
            add_reverse_edges=metadata.add_reverse_edges,
        )

        if homogeneous:
            data = to_homogeneous_data(
                hetero_data,
                metadata.node_num_keys,
                metadata.node_cat_keys,
                metadata.node_types,
                metadata.one_hot_encoding,
                metadata.unique_node_type_attribute_columns,
            )
            batch = torch.zeros(
                data.num_nodes if data.num_nodes else 0,
                dtype=torch.long,
                device=device,
            )
            data = data.to(device)
            out = model(data.x, data.edge_index, batch)
        else:
            batch_dict = {
                node_type: torch.zeros(
                    hetero_data[node_type].num_nodes
                    if hetero_data[node_type].num_nodes
                    else 0,
                    dtype=torch.long,
                    device=device,
                )
                for node_type in metadata.node_types
            }
            data = hetero_data.to(device)
            out = model(data.x_dict, data.edge_index_dict, batch_dict)
    except Exception as e:
        print(f"Error occurred while processing graph: {e}")
        print(data)
        raise e

    return bool(out.argmax(dim=-1).cpu().item())


# Determine viewpoint id(s)
viewpoint_ids = []
if counterfactual_cfg.get("viewpoint_id"):
    viewpoint_ids.append(counterfactual_cfg.get("viewpoint_id"))
    visualize = True
else:
    with open(path_labels, "r") as f:
        labels_data = json.load(f)
    for event_id, event_label in labels_data["viewpoint_event_labels"].items():
        if event_label == viewpoint_label:
            viewpoint_ids.append(event_id)

if not viewpoint_ids:
    raise ValueError(
        f"No viewpoint IDs found in {path_labels} with actual label {viewpoint_label}"
    )

# %% Construct actions for counterfactual generation (tree search)
selected_attributes = []
for node_type in ["EVENT", "OBJECT"]:
    for d in metadata.node_cat_keys[node_type].values():
        selected_attributes.extend(d.keys())

    for d in metadata.node_num_keys[node_type].values():
        selected_attributes.extend(d.keys())

attribute_spec_dict = construct_attribute_spec_dict(
    attributes=selected_attributes,
    ocel=ocel,
    node_cat_keys=metadata.node_cat_keys,
    node_num_keys=metadata.node_num_keys,
    num_bins=num_bins,
)

results = []
for viewpoint_id in viewpoint_ids:
    print("Viewpoint id:", viewpoint_id)

    # Extract target process execution
    target_process_execution = deepcopy(
        extract_process_execution(
            ocel_nx,
            viewpoint_id,
            object_types=process_execution_object_types,
            target_activity_type=process_execution_target_activity,
            backward=trace_backward,
        )
    )

    if visualize:
        visualize_process_execution(
            target_process_execution,
            f"data/{SCENARIO_PREFIX}-{viewpoint_id}-target_pe.svg",
        )

    counterfactual_label = not process_outcome(target_process_execution)
    if counterfactual_label == viewpoint_label:
        print(f"Skipping {viewpoint_id}")
        continue

    target_explanation, hetero_data, feat_label_dict, node_label_dict = (
        generate_explanation(
            G=target_process_execution,
            metadata=metadata,
            model=model,
            object_type_col=ocel.object_type_column,
            event_activity_col=ocel.event_activity,
            homogeneous=homogeneous,
            verbose=verbose,
        )
    )

    nodes_ordered = [
        n["label"]
        for n in get_nodes_by_importance(
            explanation=target_explanation,
            node_label_dict=node_label_dict,
            metadata=metadata if homogeneous else None,
            hetero_data=hetero_data if homogeneous else None,
        )
        if n["importance"] >= node_importance_threshold
    ]
    attr_ordered = {
        node_type: [
            f["feature"]
            for f in features
            if f["importance"] >= attr_importance_threshold
        ]
        for node_type, features in get_feature_labels_by_importance(
            explanation=target_explanation,
            feat_label_dict=metadata.feat_label_dict,
            node_cat_keys=metadata.node_cat_keys,
            one_hot_encoding=metadata.one_hot_encoding,
            metadata=metadata if homogeneous else None,
            hetero_data=hetero_data if homogeneous else None,
        ).items()
    }

    # Actions for object node attributes
    object_node_attributes = build_node_attribute_actions(
        target_nodes=target_process_execution.nodes(data=True),
        attribute_spec_dict=attribute_spec_dict,
        nodes_order=nodes_ordered,
        attr_order=attr_ordered,
        node_type="OBJECT",
        object_type_column=ocel.object_type_column,
    )

    # Event move actions
    target_event_nodes = [
        (n, target_process_execution.nodes(data=True)[n])
        for n in nodes_ordered
        if target_process_execution.nodes(data=True)[n].get("attr", {}).get("type", "")
        == "EVENT"
    ]
    event_move_actions = build_event_move_actions(
        target_event_nodes=target_event_nodes,
        candidate_event_nodes=target_event_nodes,
    )

    # Actions for event node attributes
    event_node_attributes = build_node_attribute_actions(
        target_nodes=target_process_execution.nodes(data=True),
        attribute_spec_dict=attribute_spec_dict,
        nodes_order=nodes_ordered,
        attr_order=attr_ordered,
        node_type="EVENT",
    )

    # Events to insert
    event_insertion_object_ids = [
        n
        for n, attr in target_process_execution.nodes(data="attr")
        if attr.get(ocel.object_type_column, "") in ["PO"]
    ]
    event_insertion_actions = build_event_insertion_actions(
        target_event_nodes=target_event_nodes,
        event_activities=list(ocel.events[ocel.event_activity].unique()),
        object_ids=event_insertion_object_ids,
    )

    # Object substitution actions
    target_object_nodes = (
        (n, target_process_execution.nodes(data=True)[n])
        for n in nodes_ordered
        if target_process_execution.nodes(data=True)[n]
        .get("attr", {})
        .get(ocel.object_type_column, "")
        in ["item"]
    )

    def _check_item_name(node_attr, subst_attr):
        return node_attr.get("item_name", "") == subst_attr.get("item_name", "")

    object_substitution_actions = build_object_substitution_actions(
        target_nodes=target_object_nodes,
        ocel_nodes=ocel_nx.nodes(data=True),
        graph=target_process_execution,
        object_type_column=ocel.object_type_column,
        attribute_spec_dict=attribute_spec_dict,
        check=_check_item_name,
    )

    # Objects to insert
    object_insertion_actions = build_object_insertion_actions(
        target_event_nodes=target_event_nodes,
        object_types=list(ocel.objects[ocel.object_type_column].unique()),
        metadata=metadata,
    )

    # Nodes (events and objects) that can be deleted
    node_deletion_actions = build_node_deletion_actions(
        target_process_execution.nodes(data=True),
        nodes_order=nodes_ordered,
        viewpoint=metadata.viewpoint,
        object_type_column=ocel.object_type_column,
    )

    # Group actions if depth_first is defined
    if depth_first == "node":
        actions_grouped = {node: [] for node in nodes_ordered}
        for action in object_node_attributes + event_node_attributes:
            if action.action_space_size() > 0:
                actions_grouped[action.node_id].append(action)

        for action in object_substitution_actions:
            if action.action_space_size() > 0:
                actions_grouped[action.object_id].append(action)

        for action in event_move_actions:
            if action.action_space_size() > 0:
                actions_grouped[action.event_id].append(action)

        for action in event_insertion_actions:
            if action.action_space_size() > 0:
                actions_grouped[action.event_id].append(action)

        for action in object_substitution_actions:
            if action.action_space_size() > 0:
                actions_grouped[action.object_id].append(action)

        for action in object_insertion_actions:
            if action.action_space_size() > 0:
                actions_grouped[action.event_id].append(action)

        for action in node_deletion_actions:
            if action.action_space_size() > 0:
                for option in action.deletion_options:
                    for node in option:
                        actions_grouped[node].append(action)

    elif depth_first == "attribute":
        actions_grouped = {
            attr: [] for attrs in attr_ordered.values() for attr in attrs
        }
        for action in object_node_attributes + event_node_attributes:
            if action.action_space_size() > 0:
                actions_grouped[action.attribute_name].append(action)

        # Include all node actions (substitute/insert/delete) in one group
        actions_grouped["node"] = (
            object_substitution_actions
            + event_move_actions
            + object_insertion_actions
            + event_insertion_actions
            + node_deletion_actions
        )

    else:
        available_actions = (
            object_node_attributes
            + event_node_attributes
            + object_substitution_actions
            + event_move_actions
            + object_insertion_actions
            + event_insertion_actions
            + node_deletion_actions
        )

    # %% Run tree search algorithm to find counter factuals
    tree_search = TreeSearchCounterFactual(
        process_outcome=process_outcome,
        max_change_size=max_change_size,
        counterfactual_label=counterfactual_label,
        log_file="logs/simulation-po_item.log",
    )

    print(f"counterfactual_label={counterfactual_label}")
    if depth_first:
        for key, group in actions_grouped.items():
            if verbose:
                print(f"Selected number of actions for {key}: {len(group)}")

        selected_action_sets = tree_search.search_depth_first(
            actions_grouped=actions_grouped,
            process_execution=target_process_execution,
        )
    else:
        print(f"Selected number of actions: {len(available_actions)}")
        selected_action_sets = tree_search.search_layer(
            actions_to_explore=[(ActionSet(), available_actions)],
            process_execution=target_process_execution,
        )

    # %% Display results
    print(f"Number of selected action sets: {len(selected_action_sets)}")
    if not selected_action_sets:
        print("No counterfactual found for viewpoint", viewpoint_id)
        results.append(
            {
                "depth_first": depth_first,
                "viewpoint_id": viewpoint_id,
                "count_explored": tree_search.count_explored,
                "action_set": None,
                "action_size": float("nan"),
                "edits": None,
                "proximity_metrics": None,
                "proximity_metrics_all": None,
                "evaluation_valid": False,
            }
        )
        continue

    sorted_action_sets = sorted(
        selected_action_sets, key=lambda a: a.action_size(), reverse=True
    )
    for action_set in sorted_action_sets:
        if verbose:
            print(f"Change size {action_set.action_size()}:", action_set)

        features_orig, adj_orig, _ = get_dense_representation(
            target_process_execution,
            metadata,
            ocel.object_type_column,
            ocel.event_activity,
            device,
        )

        _, changes = action_set.apply_changes(target_process_execution)

        if visualize:
            visualize_process_execution(
                target_process_execution,
                f"data/{SCENARIO_PREFIX}-{viewpoint_id}-cf_pe.svg",
            )

        features_cf, adj_cf, _ = get_dense_representation(
            target_process_execution,
            metadata,
            ocel.object_type_column,
            ocel.event_activity,
            device,
        )
        action_set.undo_changes(target_process_execution, changes)

        proximity_metrics = compute_proximity(
            features_orig, adj_orig, features_cf, adj_cf
        )
        edits = matched_diff_to_edits(
            adj_orig,
            adj_cf,
            features_orig,
            features_cf,
            adj_threshold=0.001,
            feat_threshold=0.001,
            graph_matching=True,
        )

        proximity_metrics_all = {}
        for other_id in viewpoint_ids:
            # Extract target process execution
            process_execution = deepcopy(
                extract_process_execution(
                    ocel_nx,
                    other_id,
                    object_types=process_execution_object_types,
                    target_activity_type=process_execution_target_activity,
                    backward=trace_backward,
                )
            )
            features_i, adj_i, _ = get_dense_representation(
                process_execution,
                metadata,
                ocel.object_type_column,
                ocel.event_activity,
                device,
            )
            proximity_metrics_all[other_id] = compute_proximity(
                features_cf, adj_cf, features_i, adj_i
            )

        results.append(
            {
                "depth_first": depth_first,
                "viewpoint_id": viewpoint_id,
                "count_explored": tree_search.count_explored,
                "action_set": str(action_set),
                "action_size": action_set.action_size(),
                "edits": str(edits),
                "proximity_metrics": proximity_metrics,
                "proximity_metrics_all": proximity_metrics_all,
                "evaluation_valid": True,
            }
        )

# Only store results if evaluated for multiple process executions
if len(viewpoint_ids) > 1:
    print(len(results), "results collected")
    run_id = os.getenv("RUN_ID")
    with open(
        f"results/{SCENARIO_PREFIX}{'-hetero' if not homogeneous else ''}{f'-depth-first={depth_first}' if depth_first else '-breadth-first'}{f'-{run_id}' if run_id else ''}.json",
        "w",
    ) as f:
        json.dump(results, f)
