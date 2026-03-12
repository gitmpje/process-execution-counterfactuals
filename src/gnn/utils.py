from dataclasses import dataclass
from networkx import Graph
from pandas import DataFrame
from torch import save as tsave
from torch import long, zeros
from torch_geometric.data import HeteroData
from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
    HeteroExplanation,
)
from typing import Callable, Dict, List, Optional, Tuple, Any


from process_execution.process_execution import extract_process_execution
from gnn.hetero_graph_data import build_hetero_data
from tree_search.action_helpers import (
    get_nodes_by_importance,
    get_feature_labels_by_importance,
)


@dataclass
class Metadata:
    viewpoint: str
    node_num_keys: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[str]]]]
    node_types: List[str]
    edge_types: List[str]
    feat_label_dict: Dict[str, List[str]]
    normalized: bool
    one_hot_encoding: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint": self.viewpoint,
            "node_num_keys": self.node_num_keys,
            "node_cat_keys": self.node_cat_keys,
            "node_types": self.node_types,
            "edge_types": self.edge_types,
            "feat_label_dict": self.feat_label_dict,
            "normalized": self.normalized,
            "one_hot_encoding": self.one_hot_encoding,
        }

    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> "Metadata":
        """
        Create a Metadata instance from a dictionary.
        """
        return cls(
            viewpoint=metadata_dict["viewpoint"],
            node_num_keys=metadata_dict["node_num_keys"],
            node_cat_keys=metadata_dict["node_cat_keys"],
            node_types=metadata_dict.get("node_types"),
            edge_types=metadata_dict.get("edge_types"),
            feat_label_dict=metadata_dict.get("feat_label_dict"),
            normalized=metadata_dict.get("normalized"),
            one_hot_encoding=metadata_dict.get("one_hot_encoding"),
        )


def construct_node_num_keys(
    selected_object_types: List[str],
    selected_event_types: List[str],
    df_objects: DataFrame,
    df_events: DataFrame,
    object_type_column: str = "ocel:type",
    event_activity_column: str = "ocel:activity",
    exclude_attributes: List[str] = None,
) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    """Construct node_num_keys: numeric attributes mapped to (min, max) per type.

    Returns dict structure:
        node_num_keys[node_type_key][type_name][column] = (min_val, max_val)

    Where node_type_key is "OBJECT" or "EVENT", and type_name is either:
        - Generic "OBJECT" or "EVENT" (for all numeric columns)
        - Specific object type (e.g. "ProductionLot")
        - Specific activity/event type (e.g. "place order")
    """
    if exclude_attributes is None:
        exclude_attributes = []

    node_num_keys = {}

    # Process OBJECT types
    object_num_keys = {}

    # Generic OBJECT: all numeric columns across all objects
    num_cols = (
        df_objects.select_dtypes(include=["number"]).dropna(axis=1).columns.tolist()
    )
    num_cols = [c for c in num_cols if c not in exclude_attributes]
    object_num_keys["OBJECT"] = {
        col: (float(df_objects[col].min()), float(df_objects[col].max()))
        for col in num_cols
    }

    # Per-type OBJECT: specific numeric columns for each object type
    for obj_type in selected_object_types:
        df_t = df_objects[df_objects[object_type_column] == obj_type]
        cols_t = df_t.select_dtypes(include=["number"]).dropna(axis=1).columns.tolist()
        cols_t = [c for c in cols_t if c not in exclude_attributes]
        object_num_keys[obj_type] = {
            col: (float(df_t[col].min()), float(df_t[col].max())) for col in cols_t
        }

    node_num_keys["OBJECT"] = object_num_keys

    # Process EVENT types
    event_num_keys = {}

    # Generic EVENT: all numeric columns across all events
    num_cols = (
        df_events.select_dtypes(include=["number"]).dropna(axis=1).columns.tolist()
    )
    num_cols = [c for c in num_cols if c not in exclude_attributes]
    event_num_keys["EVENT"] = {
        col: (float(df_events[col].min()), float(df_events[col].max()))
        for col in num_cols
    }

    # Per-type EVENT: specific numeric columns for each activity/event type
    for event_type in selected_event_types:
        df_t = df_events[df_events[event_activity_column] == event_type]
        if not df_t.empty:
            cols_t = (
                df_t.select_dtypes(include=["number"]).dropna(axis=1).columns.tolist()
            )
            cols_t = [c for c in cols_t if c not in exclude_attributes]
            event_num_keys[event_type] = {
                col: (float(df_t[col].min()), float(df_t[col].max())) for col in cols_t
            }
        else:
            event_num_keys[event_type] = {}

    node_num_keys["EVENT"] = event_num_keys

    return node_num_keys


def construct_node_cat_keys(
    selected_object_types: List[str],
    selected_event_types: List[str],
    df_objects,
    df_events,
    object_type_column: str = "ocel:type",
    event_activity_column: str = "ocel:activity",
    exclude_attributes: List[str] = None,
) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
    """Construct node_cat_keys: categorical attributes mapped to unique values per type.

    Returns dict structure:
        node_cat_keys[node_type_key][type_name][column] = [unique_value_1, unique_value_2, ...]

    Where node_type_key is "OBJECT" or "EVENT", and type_name is either:
        - Generic "OBJECT" or "EVENT" (for all categorical columns)
        - Specific object type (e.g. "ProductionLot")
        - Specific activity/event type (e.g. "place order")
    """
    if exclude_attributes is None:
        exclude_attributes = []

    node_cat_keys = {}

    # Process OBJECT types
    object_cat_keys = {}

    # Generic OBJECT: all categorical (non-numeric) columns across all objects
    cat_cols = (
        df_objects.select_dtypes(exclude=["number"]).dropna(axis=1).columns.tolist()
    )
    cat_cols = [c for c in cat_cols if c not in exclude_attributes]
    object_cat_keys["OBJECT"] = {}
    for col in cat_cols:
        unique_values = df_objects[col].dropna().unique().tolist()
        if unique_values:
            object_cat_keys["OBJECT"][col] = unique_values

    # Per-type OBJECT: specific categorical columns for each object type
    for obj_type in selected_object_types:
        object_cat_keys[obj_type] = {}
        df_t = df_objects[df_objects[object_type_column] == obj_type]
        if not df_t.empty:
            cols_t = df_t.select_dtypes(exclude=["number"]).dropna().columns.tolist()
            cols_t = [c for c in cols_t if c not in exclude_attributes]
            object_cat_keys[obj_type] = {}
            for col in cols_t:
                unique_values = df_t[col].dropna().unique().tolist()
                if unique_values:
                    object_cat_keys[obj_type][col] = unique_values
        else:
            object_cat_keys[obj_type] = {}

    node_cat_keys["OBJECT"] = object_cat_keys

    # Process EVENT types
    event_cat_keys = {}

    # Generic EVENT: all categorical (non-numeric) columns across all events
    cat_cols = df_events.select_dtypes(exclude=["number"]).dropna().columns.tolist()
    cat_cols = [c for c in cat_cols if c not in exclude_attributes]
    event_cat_keys["EVENT"] = {}
    for col in cat_cols:
        unique_values = df_events[col].dropna().unique().tolist()
        if unique_values:
            event_cat_keys["EVENT"][col] = unique_values

    # Per-type EVENT: specific categorical columns for each activity/event type
    for event_type in selected_event_types:
        event_cat_keys[event_type] = {}
        df_t = df_events[df_events[event_activity_column] == event_type]
        if not df_t.empty:
            cols_t = (
                df_t.select_dtypes(exclude=["number"]).dropna(axis=1).columns.tolist()
            )
            cols_t = [c for c in cols_t if c not in exclude_attributes]
            event_cat_keys[obj_type][col] = {}
            for col in cols_t:
                unique_values = df_t[col].dropna().unique().tolist()
                if unique_values:
                    event_cat_keys[obj_type][col] = unique_values
        else:
            event_cat_keys[event_type] = {}

    node_cat_keys["EVENT"] = event_cat_keys

    return node_cat_keys


def build_process_execution_dataset(
    ocel_nx: Graph,
    trace_object_types: List[str],
    node_cat_keys: Dict[str, Dict[str, Dict[str, List[Any]]]],
    node_num_keys: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    events_to_trace: List[str],
    viewpoint: str,
    trace_target_activity_type: Optional[str] = None,
    trace_backward: bool = False,
    graph_y_function: Optional[Callable[[Graph, str], int | float]] = None,
    node_y_mapping: Optional[dict] = None,
    object_type_col: str = "ocel:type",
    event_activity_col: str = "ocel:activity",
    add_reverse_edges: bool = False,
    normalize: bool = False,
    one_hot_encoding: bool = False,
    path_pe_dataset: Optional[str] = None,
) -> Tuple[List[HeteroData], Metadata]:
    """
    Build a HeteroData dataset from process executions in an OCEL graph.

    This function extracts process executions from events in an OCEL graph,
    converts each to a HeteroData graph, and compiles them into a dataset
    with associated metadata.

    Args:
        ocel_nx: NetworkX OCEL graph to extract process executions from.
        object_types: List of object types to consider in the trace.
        target_activity_type: Activity type to end the trace. Can be None for full traces.
        backward: Whether to trace backward (True) or forward (False).
        graph_y_function: Optional callable that determines the target value for each graph.
            Should take (graph, event_id) and return an int/float/None.
            If None, y values will be set to NaN and can be filled later.
        node_cat_keys: Dict mapping node types to categorical attributes and their unique values.
        node_num_keys: Dict mapping node types to numeric attributes and their (min, max) bounds.
        events_to_trace: List of event IDs to extract process executions from.
        path_pe_dataset: Optional path to save the compiled dataset as a .pt file.
        viewpoint: The node type to use as the viewpoint (e.g., "PackingUnit").
        object_type_col: Column name for object type attribute (default: "ocel:type").
        event_activity_col: Column name for event activity attribute (default: "ocel:activity").
        add_reverse_edges: Whether to add reverse edges in the graph.
        normalize: Whether to normalize numeric features.

    Returns:
        Tuple of (dataset, metadata) where:
            - dataset: List of HeteroData objects, one per process execution.
            - metadata: Metadata object containing dataset information (node types, edge types,
              feature labels, etc.).

    Raises:
        RuntimeError: If no valid process executions are found.
        ValueError: If viewpoint type has no nodes in any extracted graph.
    """
    dataset = []
    node_types_set = set()
    edge_types_set = set()
    feat_label_dict = {}

    for idx, event in enumerate(events_to_trace):
        try:
            # Extract process execution
            G = extract_process_execution(
                ocel_nx,
                event,
                trace_object_types,
                trace_target_activity_type,
                backward=trace_backward,
            )

            # Build HeteroData graph
            hetero_data, n_types, e_types, _, feat_labels, _ = build_hetero_data(
                graph=G,
                node_num_keys=node_num_keys,
                node_cat_keys=node_cat_keys,
                object_type_col=object_type_col,
                event_activity_col=event_activity_col,
                viewpoint=viewpoint,
                node_y_mapping=node_y_mapping,
                add_reverse_edges=add_reverse_edges,
                normalize=normalize,
                one_hot_encoding=one_hot_encoding,
            )

            # Set graph-level y if graph_y_function was provided
            if graph_y_function is not None:
                y_value = graph_y_function(G, event)
                hetero_data.y = y_value

            dataset.append(hetero_data)
            node_types_set.update(n_types)
            edge_types_set.update(e_types)

            # Fill global feature/label dict if empty
            for k, v in feat_labels.items():
                if k not in feat_label_dict:
                    feat_label_dict[k] = v

            if idx % 50 == 0:
                print(f"Processed {idx} process executions")

        except Exception as e:
            print(f"Failed converting event {event} to HeteroData: {e}")
        finally:
            del G

    if not dataset:
        raise RuntimeError("No complete process executions found to build dataset")

    # Save dataset if path provided
    if path_pe_dataset:
        tsave(dataset, path_pe_dataset)

    # Create metadata
    metadata = Metadata(
        viewpoint=viewpoint,
        node_num_keys=node_num_keys,
        node_cat_keys=node_cat_keys,
        node_types=list(node_types_set),
        edge_types=list(edge_types_set),
        feat_label_dict=feat_label_dict,
        normalized=normalize,
        one_hot_encoding=one_hot_encoding,
    )

    return dataset, metadata


def generate_explanation(
    G: Graph,
    metadata: Metadata,
    model,
    object_type_col: str,
    event_activity_col: str,
    verbose=False,
) -> HeteroExplanation:
    device = next(model.parameters()).device

    data, _, _, _, feat_label_dict, node_label_dict = build_hetero_data(
        graph=G,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=object_type_col,
        event_activity_col=event_activity_col,
        viewpoint=metadata.viewpoint,
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
    )

    data = data.to(device)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
        node_mask_type="attributes",
        threshold_config=dict(
            threshold_type="topk",
            value=200,
        ),
    )
    # For a single graph, create a batch vector of zeros (all nodes belong to graph 0)
    batch_dict = {
        node_type: zeros(data[node_type].num_nodes, dtype=long, device=device)
        for node_type in metadata.node_types
    }
    explanation = explainer(
        x=data.x_dict,
        edge_index=data.edge_index_dict,
        batch_dict=batch_dict,
    )

    if verbose:
        top_nodes = get_nodes_by_importance(explanation, node_label_dict, top_k=20)

        print("Top nodes by importance:")
        for n in top_nodes:
            print(
                f"{n['label']} ({n['node_type']}:{n['node_index']}): {n['importance']:.6f}"
            )

        top_features = get_feature_labels_by_importance(
            explanation,
            metadata.feat_label_dict,
            one_hot_encoding=metadata.one_hot_encoding,
            top_k=10,
        )
        print("\nTop features by importance per node type:")
        for nt, feats in top_features.items():
            print(f"\nNode type: {nt}")
            for f in feats:
                print(f"  {f['feature']}: {f['importance']:.6f}")

    return (
        explanation,
        feat_label_dict,
        node_label_dict,
    )
