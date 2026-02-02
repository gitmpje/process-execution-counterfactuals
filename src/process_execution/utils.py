import json
from pandas import Timestamp
from pm4py import OCEL
from pm4py.convert import convert_ocel_to_networkx
from networkx import read_graphml, Graph, MultiDiGraph, write_graphml
from typing import List, Tuple


def build_ocel_dfg(
    ocel: OCEL,
    selected_aggregation_activity_qualifier: List[Tuple[str, str]] = None,
    include_object_relations: bool = False,
) -> MultiDiGraph:
    """Build a Directly-Follows Graph (DFG) from an OCEL, including aggregation edges.
    Args:
        ocel (OCEL): The OCEL object.
        selected_aggregation_activity_qualifier (List[Tuple[str, str]], optional):
            List of (activity, qualifier) pairs for which to add aggregation DF edges.
            Defaults to [].
        include_object_relations (bool): Whether to add object to object relations to the graph.
    Returns:
        MultiDiGraph: The constructed OCEL DFG with aggregation edges."""

    selected_aggregation_activity_qualifier = (
        selected_aggregation_activity_qualifier
        if selected_aggregation_activity_qualifier
        else []
    )

    _ocel_nx = convert_ocel_to_networkx(ocel)

    # Workaround for https://github.com/process-intelligence-solutions/pm4py/issues/534
    ocel_nx = MultiDiGraph()
    ocel_nx.add_nodes_from(_ocel_nx.nodes(data=True))
    ocel_nx.add_edges_from(
        [e for e in _ocel_nx.edges(data=True) if e[-1]["attr"].get("type") != "DF"]
    )

    def lifecycle_max_lower_than(lif: List[str], e_prime: str):
        lif_int = [int(e) for e in lif]
        return str(max([e for e in lif_int if e < int(e_prime)]))

    agg_act_qual = [
        f"{act}-{qual}" for act, qual in selected_aggregation_activity_qualifier
    ]
    ocel.relations["activity-qualifier"] = (
        ocel.relations[ocel.event_activity] + "-" + ocel.relations[ocel.qualifier]
    )

    lifecycle = (
        ocel.relations.groupby(ocel.object_id_column)
        .agg(list)
        .to_dict()[ocel.event_id_column]
    )
    for obj in lifecycle:
        # Add DF edges
        lif = lifecycle[obj]
        for i in range(len(lif) - 1):
            ocel_nx.add_edge(lif[i], lif[i + 1], attr={"type": "DF", "object": obj})

        # Add aggregation DF edges
        for activity, qualifier in selected_aggregation_activity_qualifier:
            relations_obj = ocel.relations[ocel.relations[ocel.object_id_column] == obj]
            lif = lifecycle[obj]
            # Add DF_agg for selected aggregation events, taking into account the E2O qualifier
            for event in relations_obj[
                (relations_obj[ocel.event_activity] == activity)
                & (relations_obj[ocel.qualifier] == qualifier)
            ][ocel.event_id_column].values:
                # Get preceding event from events that are not in the selected aggregation activity - qualifier pairs
                lif_selected = relations_obj[
                    ~relations_obj["activity-qualifier"].isin(agg_act_qual)
                ][ocel.event_id_column].values
                ocel_nx.add_edge(
                    lifecycle_max_lower_than(lif_selected, event),
                    event,
                    attr={"type": "DF_agg", "object": obj},
                )

    if include_object_relations:
        ocel_nx_o2o = convert_ocel_to_networkx(ocel, "ocel_features_to_nx")
        ocel_nx.add_edges_from(ocel_nx_o2o.edges(data=True))

    return ocel_nx


# Store graph to GraphML format
def store_ocel_dfg_graphml(ocel_nx: MultiDiGraph, path_graphml: str):
    """Store OCEL DFG graph to GraphML format, handling JSON-serializable attributes.

    Args:
        ocel_nx (MultiDiGraph): The OCEL DFG graph.
        path_graphml (str): The file path to store the GraphML file.
    """

    # Convert timestamp attributes to ISO format strings
    for _, d in ocel_nx.nodes(data=True):
        if not d.get("attr", {}).get("ocel:timestamp"):
            continue
        if isinstance(d["attr"]["ocel:timestamp"], Timestamp):
            d["attr"]["ocel:timestamp"] = d["attr"]["ocel:timestamp"].isoformat()

    # Convert node attributes that are dictionaries into JSON strings so GraphML can store them
    for _, d in ocel_nx.nodes(data=True):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                try:
                    d[k] = json.dumps(v)
                except Exception:
                    d[k] = str(v)

    # Convert edge attributes (handle MultiDiGraph and DiGraph)
    try:
        # MultiGraph/MultiDiGraph edges include keys
        for _, _, _, ed in ocel_nx.edges(keys=True, data=True):
            for k, val in list(ed.items()):
                if isinstance(val, dict):
                    try:
                        ed[k] = json.dumps(val)
                    except Exception:
                        ed[k] = str(val)
    except TypeError:
        # fallback for Graph/DiGraph without keys
        for _, _, ed in ocel_nx.edges(data=True):
            for k, val in list(ed.items()):
                if isinstance(val, dict):
                    try:
                        ed[k] = json.dumps(val)
                    except Exception:
                        ed[k] = str(val)

    write_graphml(ocel_nx, path=path_graphml)


# Load the graphml and parse JSON attributes back
def load_graphml_with_json_attrs(path: str) -> Graph:
    """Read a GraphML file and attempt to JSON-decode any string attributes back into Python objects.

    Only replaces attribute values when json.loads returns a dict or list (to avoid converting plain strings).
    Works for Graph/DiGraph and MultiGraph/MultiDiGraph edge representations.
    """
    G = read_graphml(path)

    # Nodes
    for _, d in G.nodes(data=True):
        for k, v in list(d.items()):
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, (dict, list)):
                        d[k] = parsed
                except Exception:
                    # leave as string if it isn't JSON
                    pass

    # Edges (handle keyed MultiGraphs and non-keyed graphs)
    try:
        edges = list(G.edges(keys=True, data=True))
        keyed = True
    except TypeError:
        edges = list(G.edges(data=True))
        keyed = False

    if keyed:
        for _, _, _, ed in edges:
            for k, val in list(ed.items()):
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            ed[k] = parsed
                    except Exception:
                        pass
    else:
        for _, _, ed in edges:
            for k, val in list(ed.items()):
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            ed[k] = parsed
                    except Exception:
                        pass

    return G
