from networkx import Graph

from gnn.hetero_graph_data import build_data, build_hetero_data, to_homogeneous_data


def test_build_data_creates_homogeneous_graph_with_consistent_feature_dimension():
    graph = Graph()
    graph.add_node(
        1,
        attr={
            "type": "OBJECT",
            "ocel:type": "OBJECT",
            "value": 1.0,
            "kind": "A",
        },
    )
    graph.add_node(
        2,
        attr={
            "type": "EVENT",
            "ocel:activity": "EVENT",
            "duration": 3.0,
            "activity": "x",
        },
    )
    graph.add_edge(1, 2, attr={"type": "rel"})

    node_num_keys = {
        "OBJECT": {"OBJECT": {"value": (0.0, 1.0)}},
        "EVENT": {"EVENT": {"duration": (0.0, 5.0)}},
    }
    node_cat_keys = {
        "OBJECT": {"OBJECT": {"kind": ["A", "B"]}},
        "EVENT": {"EVENT": {"activity": ["x", "y"]}},
    }

    data, node_types, edge_types, y_nodes, feat_labels, node_labels = build_data(
        graph=graph,
        node_num_keys=node_num_keys,
        node_cat_keys=node_cat_keys,
        object_type_col="ocel:type",
        event_activity_col="ocel:activity",
        viewpoint="EVENT",
        node_y_mapping={2: 1},
        add_reverse_edges=False,
        normalize=False,
        one_hot_encoding=False,
        node_types=["OBJECT", "EVENT"],
    )

    assert data.x.shape == (2, 2)
    assert data.edge_index.shape[1] == 1
    assert y_nodes == [2]
    assert set(node_types) == {"OBJECT", "EVENT"}
    assert feat_labels["OBJECT"] == ["value", "kind"]
    assert feat_labels["EVENT"] == ["duration", "activity"]


def test_to_homogeneous_data_pads_missing_node_types():
    graph = Graph()
    graph.add_node(
        1,
        attr={
            "type": "OBJECT",
            "ocel:type": "OBJECT",
            "value": 2.0,
            "kind": "B",
        },
    )

    node_num_keys = {
        "OBJECT": {"OBJECT": {"value": (0.0, 2.0)}},
        "EVENT": {"EVENT": {"duration": (0.0, 5.0)}},
    }
    node_cat_keys = {
        "OBJECT": {"OBJECT": {"kind": ["A", "B"]}},
        "EVENT": {"EVENT": {"activity": ["x", "y"]}},
    }

    hetero_data, node_types, edge_types, _, feat_labels, _ = build_hetero_data(
        graph=graph,
        node_num_keys=node_num_keys,
        node_cat_keys=node_cat_keys,
        object_type_col="ocel:type",
        event_activity_col="ocel:activity",
        viewpoint="OBJECT",
        add_reverse_edges=False,
        normalize=False,
        one_hot_encoding=False,
    )
    data = to_homogeneous_data(
        hetero_data,
        node_num_keys=node_num_keys,
        node_cat_keys=node_cat_keys,
        node_types=["OBJECT", "EVENT"],
        one_hot_encoding=False,
    )

    assert data.x.shape == (1, 2)
    assert data.edge_index.shape[1] == 0
    assert feat_labels["OBJECT"] == ["value", "kind"]
