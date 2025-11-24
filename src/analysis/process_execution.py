from itertools import combinations
from typing import Generator, List, Set, Tuple
from networkx import all_simple_edge_paths, edge_subgraph, subgraph, DiGraph, Graph


class ProcessExecution(DiGraph):
    def construct_node_label(G: Graph):
        """
        Construct label based on attributes from node['attr'].
        - G (Graph): NetworkX OCEL graph
        """
        for _, data in list(G.nodes(data=True)):
            a = data.get("attr")
            if a is None:
                continue
            if isinstance(a, dict):
                node_type = a.get("type")
                type_activity = (
                    a.get("ocel:type")
                    if node_type == "OBJECT"
                    else a.get("ocel:activity")
                )
                data["label"] = f"{node_type}_{type_activity}"

    def construct_edge_label(G: Graph):
        """
        Construct label based on attributes from edge['attr'].
        - G (Graph): NetworkX OCEL graph
        """
        for _, _, data in list(G.edges(data=True)):
            a = data.get("attr")
            if a is None:
                continue
            if isinstance(a, dict):
                edge_type = a.get("type")
                edge_qualifier = a.get("qualifier")
                data["label"] = f"{edge_type}_{edge_qualifier}" if edge_qualifier else edge_type

    def select_node_attr(G: Graph, attr_key: str):
        """
        Add value from node['attr'][attr_key] as attribute 'selected_attr' of the nodes.
        - G (Graph): networkx OCEL graph
        """
        for node, data in list(G.nodes(data=True)):
            a = data.get("attr")
            if a is None:
                continue
            if isinstance(a, dict):
                data["selected_attr"] = a.get(attr_key)

    def extract_normalized_process_executions(
        self, e_prime: str, normalize_events: Set[str]
    ) -> Generator:
        """
        Given a process execution graph, return all process executions (sub)graphs
        normalized on the provided events.

        Args:
            e_prime (str): target event for which to extract the process execution(s).
            normalize_events (Set[str]): set of events on which to normalize the process execution.

        Returns:
            Generator[Graph]: generator of normalized process execution graphs
        """

        root_events = [n for n in self.nodes() if self.in_degree(n) == 0]
        paths_edges = []
        for e_r in root_events:
            paths_edges.extend(
                [
                    tuple(p)
                    for p in all_simple_edge_paths(self, source=e_r, target=e_prime)
                ]
            )

        complete_executions, _ = get_maximal_combinations_optimized(
            paths_edges, normalize_events
        )
        for complete_execution in complete_executions:
            yield edge_subgraph(
                self, [edge for path in complete_execution for edge in path]
            )


def check_event_object_qualifier(
    G: Graph, event: str, obj: str, event_object_qualifiers: List[str] = []
) -> bool:
    for u, v, d in G.out_edges(event, data=True):
        edge_type = d["attr"].get("type")
        qualifier = d["attr"].get("qualifier")
        activity = G.nodes()[event]["attr"].get("ocel:activity")
        if (
            (edge_type == "E2O")
            and (v == obj)
            and (
                (qualifier in event_object_qualifiers)
                or (activity, qualifier) in event_object_qualifiers
            )
        ):
            return True
    return False


def extract_process_execution(
    G: Graph,
    source_event: str,
    event_object_qualifiers: List[str] = [],
    target_activity_type: str = None,
) -> ProcessExecution:
    """
    G (Graph): NetworkX OCEL graph to extract process execution from;
    source_event (str): source event node to trace from;
    event_object_qualifiers (List[str|Tuple[str]]): list of event-object relationship qualifiers used to select objects for which to extract the process execution.
    target_activity_type (str): activity type to end the trace;
    """
    events_to_check = [source_event]
    nodes_traced = set()
    while events_to_check:
        event = events_to_check.pop()
        nodes_traced.add(event)

        # End trace when target activity type is encountered
        if target_activity_type and (
            G.nodes()[event]["attr"].get("ocel:activity") == target_activity_type
        ):
            continue

        # Add objects related to event to trace
        trace_objects = [
            e[1]
            for e in list(G.out_edges(event, data=True))
            if e[-1]["attr"].get("type") == "E2O"
        ]
        nodes_traced.update(trace_objects)

        trace_df_edges = [
            e
            for e in list(G.in_edges(event, data=True))
            if e[-1]["attr"].get("type") == "DF"
        ]

        # Filter traced events based on qualifier and activity
        selected_trace_events = set(
            [
                u
                for u, v, d in trace_df_edges
                if check_event_object_qualifier(
                    G, u, d["attr"]["object"], event_object_qualifiers
                )
            ]
        )

        events_to_check.extend(list(selected_trace_events - nodes_traced))

    return ProcessExecution(subgraph(G, nodes_traced))


def check_edges_normalize(edges: List[Tuple[str]], normalize_events: List[str]) -> bool:
    """
    Check if there are multiple edges with the same target that occurs in normalize_events.

    Args:
        edges (List[Tuple[str]]): set of edges.
        normalize_events (List[str]): set of events on which to normalize the process execution.

    Returns:
        boolean: True if there are multiple edges with the same target that occurs in normalize_events
    """
    for normalize_event in normalize_events:
        edges_to_normalize = {(u, v) for (u, v) in edges if v == normalize_event}

        # Check constraint: at most one distinct edge to normalize_event
        if len(edges_to_normalize) > 1:
            return True

    return False


def get_all_edges(paths):
    """Get union of all edges from a set of paths."""
    edges = set()
    for path in paths:
        edges.update(path)
    return edges


def is_valid_and_maximal(T, S, normalize_events) -> bool:
    """
    Check if combination T is valid (satisfies constraint) and maximal.
    """
    # Get all edges
    edges = get_all_edges(T)

    # Check constraint: at most one distinct edge to each normalize_event
    if check_edges_normalize(edges, normalize_events):
        return False

    # Check maximality: try to add each path not in T
    for path in S:
        if path not in T:
            # Check if we can add this path
            new_edges = edges | set(path)

            if not check_edges_normalize(new_edges, normalize_events):
                # We can add this path, so T is not maximal
                return False

    return True


def get_maximal_combinations_optimized(
    S: List[Tuple[str]], normalize_events: List[str]
):
    """
    Directly construct maximal valid combinations without generating all subsets.

    Args:
        S (List[Tuple[str]]): set of paths, where each path is a tuple of edges (u,v).
        normalize_events (List[str]): set of events on which to normalize the process execution.

    Returns:
        C_max: set of maximal combinations (frozensets of paths)
        E_C_max: dict mapping each maximal combination to its set of edges
    """
    # Group paths by the edges to normalize_events they contain
    # Key: frozenset of (normalize_event, edge) tuples
    # Value: list of paths
    paths_by_normalize_edges = {}
    paths_without_normalize_edges = []

    for path in S:
        # Find all edges targeting normalize_events in this path
        edges_to_normalize = []
        for normalize_event in normalize_events:
            edges = [(u, v) for (u, v) in path if v == normalize_event]
            for edge in edges:
                edges_to_normalize.append((normalize_event, edge))

        if len(edges_to_normalize) == 0:
            paths_without_normalize_edges.append(path)
        else:
            # Create a signature for this path based on its normalize edges
            signature = frozenset(edges_to_normalize)
            if signature not in paths_by_normalize_edges:
                paths_by_normalize_edges[signature] = []
            paths_by_normalize_edges[signature].append(path)

    C_max = set()
    E_C_max = {}

    # Strategy 1: Select all paths without edges to normalize_events
    candidate = frozenset(paths_without_normalize_edges)
    if is_valid_and_maximal(candidate, S, normalize_events):
        C_max.add(candidate)
        E_C_max[candidate] = get_all_edges(candidate)

    # Strategy 2: For each compatible combination of normalize edge signatures
    # Two signatures are compatible if they don't have conflicting edges to the same normalize_event
    signatures = list(paths_by_normalize_edges.keys())

    # Try each signature group
    for signature in signatures:
        candidate = set(paths_by_normalize_edges[signature])
        candidate.update(paths_without_normalize_edges)
        candidate = frozenset(candidate)

        if is_valid_and_maximal(candidate, S, normalize_events):
            C_max.add(candidate)
            E_C_max[candidate] = get_all_edges(candidate)

    # Strategy 3: Try compatible combinations of signatures
    # Two signatures are compatible if for each normalize_event, they have at most one distinct edge
    for r in range(2, len(signatures) + 1):
        for sig_combo in combinations(signatures, r):
            # Check if these signatures are compatible
            edges_by_normalize_event = {}
            compatible = True

            for sig in sig_combo:
                for normalize_event, edge in sig:
                    if normalize_event not in edges_by_normalize_event:
                        edges_by_normalize_event[normalize_event] = set()
                    edges_by_normalize_event[normalize_event].add(edge)

                    # If we have more than one distinct edge to this normalize_event, not compatible
                    if len(edges_by_normalize_event[normalize_event]) > 1:
                        compatible = False
                        break
                if not compatible:
                    break

            if compatible:
                # Combine all paths from these signatures
                candidate = set()
                for sig in sig_combo:
                    candidate.update(paths_by_normalize_edges[sig])
                candidate.update(paths_without_normalize_edges)
                candidate = frozenset(candidate)

                if is_valid_and_maximal(candidate, S, normalize_events):
                    C_max.add(candidate)
                    E_C_max[candidate] = get_all_edges(candidate)

    # Strategy 4: Empty set (if no paths can be combined)
    empty = frozenset()
    if is_valid_and_maximal(empty, S, normalize_events):
        C_max.add(empty)
        E_C_max[empty] = set()

    return C_max, E_C_max
