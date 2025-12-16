import json
from networkx import read_graphml, Graph


# Load the graphml and parse JSON attributes back
def load_graphml_with_json_attrs(path: str) -> Graph:
    """Read a GraphML file and attempt to JSON-decode any string attributes back into Python objects.

    Only replaces attribute values when json.loads returns a dict or list (to avoid converting plain strings).
    Works for Graph/DiGraph and MultiGraph/MultiDiGraph edge representations.
    """
    G = read_graphml(path)

    # Nodes
    for n, d in G.nodes(data=True):
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
        for u, v, key, ed in edges:
            for k, val in list(ed.items()):
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            ed[k] = parsed
                    except Exception:
                        pass
    else:
        for u, v, ed in edges:
            for k, val in list(ed.items()):
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            ed[k] = parsed
                    except Exception:
                        pass

    return G
