from networkx import Graph
from torch import long, zeros
from torch_geometric.explain import (
    Explainer,
    Explanation,
    GNNExplainer,
    HeteroExplanation,
)

from gnn.utils import Metadata
from gnn.hetero_graph_data import build_hetero_data, to_homogeneous_data
from tree_search.action_helpers import (
    get_nodes_by_importance,
    get_feature_labels_by_importance,
)


def generate_explanation(
    G: Graph,
    metadata: Metadata,
    model,
    object_type_col: str,
    event_activity_col: str,
    homogeneous: bool = False,
    verbose: bool = False,
) -> HeteroExplanation | Explanation:
    device = next(model.parameters()).device

    hetero_data, _, _, _, feat_label_dict, node_label_dict = build_hetero_data(
        graph=G,
        node_num_keys=metadata.node_num_keys,
        node_cat_keys=metadata.node_cat_keys,
        object_type_col=object_type_col,
        event_activity_col=event_activity_col,
        viewpoint=metadata.viewpoint,
        normalize=metadata.normalized,
        one_hot_encoding=metadata.one_hot_encoding,
        add_reverse_edges=metadata.add_reverse_edges,
    )

    hetero_data = hetero_data.to(device)

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

    if homogeneous:
        data = to_homogeneous_data(
            hetero_data,
            metadata.node_num_keys,
            metadata.node_cat_keys,
            metadata.node_types,
            metadata.one_hot_encoding,
            metadata.unique_node_type_attribute_columns,
        )
        batch = zeros(
            data.num_nodes if data.num_nodes else 0,
            dtype=long,
            device=device,
        )
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch,
        )
    else:
        batch_dict = {
            node_type: zeros(
                hetero_data[node_type].num_nodes
                if hetero_data[node_type].num_nodes
                else 0,
                dtype=long,
                device=device,
            )
            for node_type in metadata.node_types
        }
        explanation = explainer(
            x=hetero_data.x_dict,
            edge_index=hetero_data.edge_index_dict,
            batch_dict=batch_dict,
        )

    if verbose:
        top_nodes = get_nodes_by_importance(
            explanation,
            node_label_dict,
            top_k=20,
            metadata=metadata if homogeneous else None,
            hetero_data=hetero_data if homogeneous else None,
        )

        print("Top nodes by importance:")
        for n in top_nodes:
            print(
                f"{n['label']} ({n['node_type']}:{n['node_index']}): {n['importance']:.6f}"
            )

        top_features = get_feature_labels_by_importance(
            explanation,
            metadata.feat_label_dict,
            node_cat_keys=metadata.node_cat_keys,
            one_hot_encoding=metadata.one_hot_encoding,
            top_k=10,
            metadata=metadata if homogeneous else None,
            hetero_data=hetero_data if homogeneous else None,
        )
        print("\nTop features by importance per node type:")
        for nt, feats in top_features.items():
            print(f"\nNode type: {nt}")
            for f in feats:
                print(f"  {f['feature']}: {f['importance']:.6f}")

    return (
        explanation,
        hetero_data,
        feat_label_dict,
        node_label_dict,
    )
