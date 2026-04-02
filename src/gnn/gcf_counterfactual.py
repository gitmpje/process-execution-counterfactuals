"""
GCFExplainer – local and global counterfactual explanations for HeteroData graphs.

Adapted from:
  Huang, Kosan et al. "Global Counterfactual Explainer for Graph Neural Networks."
  WSDM 2023.  https://dl.acm.org/doi/10.1145/3539597.3570376

Public API
──────────
  gcf_explain(predict_fn, graph, target_class, ...)
      Local explainer: find the minimal edit to flip one graph's prediction.

  gcf_explain_global(predict_fn, graphs, target_class, ...)
      Global explainer: find a small set of representative counterfactual
      graphs that collectively cover all input graphs.

Local algorithm overview
────────────────────────
1.  Start from the input graph G.
2.  Build a neighbourhood of G in edit-map space (single-step edits).
3.  Run a Vertex-Reinforced Random Walk (VRRW): at each step move to a
    neighbour with probability ∝ visit_count^α, with teleportation back
    to G with probability τ.  Neighbours beyond max_distance are pruned.
4.  Collect all visited graphs predicted as target_class → candidates.
5.  Return the candidate with minimum GED to G as a list of edits.

Global algorithm overview  (Section 3 of the paper)
────────────────────────────────────────────────────
1.  Run the VRRW simultaneously over all input graphs: at each teleport
    step, pick a random input graph as the new origin.  This builds a
    shared pool of counterfactual candidates across the whole dataset.
2.  Greedy summary selection: iteratively add the candidate C* that
    maximises  α·individual_coverage(C*) + (1-α)·marginal_gain(C*)
    until `summary_size` candidates are chosen.
3.  Return the summary as a list of CounterfactualEdit objects (one per
    selected representative), plus per-input-graph coverage metadata.

The module is intentionally self-contained: it only depends on
PyTorch, PyTorch Geometric, and the standard library.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch_geometric.data import HeteroData


# ─────────────────────────────────────────────────────────────────────────────
# Public result type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CounterfactualEdit:
    """
    Minimal set of edits that (when applied to `original`) causes `predict_fn`
    to return `target_class`.

    Every item in each list is a dict describing one atomic change:

    Node removal:
        {"action": "remove_node", "node_type": str, "node_idx": int}

    Node insertion  (feature vector for the new node):
        {"action": "add_node", "node_type": str, "features": Tensor}

    Edge removal:
        {"action": "remove_edge", "edge_type": tuple[str,str,str],
         "src": int, "dst": int}

    Edge insertion:
        {"action": "add_edge", "edge_type": tuple[str,str,str],
         "src": int, "dst": int,
         "features": Tensor | None}   # None when no edge attr present
    """

    edits: list[dict[str, Any]] = field(default_factory=list)
    target_class: int = -1
    original_class: int = -1
    counterfactual_found: bool = False

    def __repr__(self) -> str:
        lines = [
            f"CounterfactualEdit(target={self.target_class}, "
            f"original={self.original_class}, "
            f"found={self.counterfactual_found}, "
            f"n_edits={len(self.edits)})"
        ]
        for e in self.edits:
            lines.append(f"  {e}")
        return "\n".join(lines)


@dataclass
class GlobalCounterfactualResult:
    """
    Output of gcf_explain_global.

    Attributes
    ----------
    summary : list[CounterfactualEdit]
        The selected representative counterfactual edits (length ≤ summary_size).
        Each entry is the minimal edit from the closest input graph to that
        representative, so it is human-interpretable as a recourse rule.
    coverage : float
        Fraction of input graphs covered by the summary (i.e. for which at
        least one representative is within `distance_threshold` GED after
        applying its edits).
    covered_indices : list[int]
        Indices into the input `graphs` list that are covered.
    target_class : int
    """

    summary: list[CounterfactualEdit] = field(default_factory=list)
    coverage: float = 0.0
    covered_indices: list[int] = field(default_factory=list)
    target_class: int = -1

    def __repr__(self) -> str:
        return (
            f"GlobalCounterfactualResult("
            f"summary_size={len(self.summary)}, "
            f"coverage={self.coverage:.3f}, "
            f"covered={len(self.covered_indices)}, "
            f"target={self.target_class})"
        )


def _graph_key(graph: HeteroData) -> tuple:
    """
    Hashable fingerprint of a HeteroData graph based on its edge sets.
    Node features are intentionally excluded for efficiency; if your
    experiments need feature-level edits as the primary axis, extend this.
    """
    parts: list[tuple] = []
    for et in sorted(graph.edge_types):
        ei = graph[et].edge_index
        edges = frozenset(map(tuple, ei.t().tolist()))
        parts.append((et, edges))
    # Include node counts so graphs with different sizes are distinguished
    for nt in sorted(graph.node_types):
        parts.append((nt, graph[nt].num_nodes))
    return tuple(parts)


def _clone(graph: HeteroData) -> HeteroData:
    """Deep-copy a HeteroData graph."""
    return copy.deepcopy(graph)


# ─────────────────────────────────────────────────────────────────────────────
# Edit-map neighbourhood
# ─────────────────────────────────────────────────────────────────────────────


def _edge_removal_neighbours(graph: HeteroData) -> list[tuple[HeteroData, dict]]:
    """Yield all single-edge-removal neighbours."""
    neighbours = []
    for et in graph.edge_types:
        ei = graph[et].edge_index  # (2, E)
        has_attr = hasattr(graph[et], "edge_attr") and graph[et].edge_attr is not None
        E = ei.size(1)
        for i in range(E):
            g2 = _clone(graph)
            mask = torch.ones(E, dtype=torch.bool)
            mask[i] = False
            g2[et].edge_index = ei[:, mask]
            if has_attr:
                g2[et].edge_attr = graph[et].edge_attr[mask]
            edit = {
                "action": "remove_edge",
                "edge_type": et,
                "src": int(ei[0, i]),
                "dst": int(ei[1, i]),
            }
            neighbours.append((g2, edit))
    return neighbours


def _edge_insertion_neighbours(
    graph: HeteroData,
    max_new_edges: int = 5,
) -> list[tuple[HeteroData, dict]]:
    """
    Yield single-edge-insertion neighbours.
    To keep the search space tractable we only sample up to `max_new_edges`
    random candidate insertions per edge type.
    """
    neighbours = []
    for et in graph.edge_types:
        src_type, _, dst_type = et
        n_src = graph[src_type].num_nodes
        n_dst = graph[dst_type].num_nodes
        if n_src == 0 or n_dst == 0:
            continue
        ei = graph[et].edge_index
        existing = set(map(tuple, ei.t().tolist()))
        has_attr = hasattr(graph[et], "edge_attr") and graph[et].edge_attr is not None
        feat_dim = graph[et].edge_attr.size(1) if has_attr else None

        candidates = [
            (s, d) for s in range(n_src) for d in range(n_dst) if (s, d) not in existing
        ]
        random.shuffle(candidates)
        for s, d in candidates[:max_new_edges]:
            g2 = _clone(graph)
            new_edge = torch.tensor([[s], [d]], dtype=torch.long)
            g2[et].edge_index = torch.cat([ei, new_edge], dim=1)
            new_feat = None
            if has_attr:
                # Use the mean of existing edge features as a proxy
                mean_feat = graph[et].edge_attr.mean(0, keepdim=True)
                g2[et].edge_attr = torch.cat([graph[et].edge_attr, mean_feat], dim=0)
                new_feat = mean_feat.squeeze(0)
            edit = {
                "action": "add_edge",
                "edge_type": et,
                "src": s,
                "dst": d,
                "features": new_feat,
            }
            neighbours.append((g2, edit))
    return neighbours


def _node_removal_neighbours(graph: HeteroData) -> list[tuple[HeteroData, dict]]:
    """
    Yield single-node-removal neighbours.
    Removing a node also removes all edges incident to it.
    """
    neighbours = []
    for nt in graph.node_types:
        n = graph[nt].num_nodes
        for i in range(n):
            g2 = _clone(graph)
            # Remove node i from node type nt
            keep_mask = torch.ones(n, dtype=torch.bool)
            keep_mask[i] = False
            # Remap features
            g2[nt].x = graph[nt].x[keep_mask]
            g2[nt].num_nodes = int(keep_mask.sum())
            # Remove incident edges and re-index
            for et in graph.edge_types:
                src_type, _, dst_type = et
                ei = g2[et].edge_index.clone()
                if src_type == nt:
                    valid = ei[0] != i
                    ei = ei[:, valid]
                    if hasattr(g2[et], "edge_attr") and g2[et].edge_attr is not None:
                        g2[et].edge_attr = g2[et].edge_attr[valid]
                    # Re-index src
                    shift = (ei[0] > i).long()
                    ei[0] = ei[0] - shift
                if dst_type == nt:
                    valid2 = ei[1] != i
                    ei = ei[:, valid2]
                    if hasattr(g2[et], "edge_attr") and g2[et].edge_attr is not None:
                        g2[et].edge_attr = g2[et].edge_attr[valid2]
                    shift2 = (ei[1] > i).long()
                    ei[1] = ei[1] - shift2
                g2[et].edge_index = ei
            edit = {"action": "remove_node", "node_type": nt, "node_idx": i}
            neighbours.append((g2, edit))
    return neighbours


def _get_neighbours(
    graph: HeteroData,
    allow_node_removal: bool,
    allow_edge_insertion: bool,
    max_new_edges: int,
) -> list[tuple[HeteroData, dict]]:
    nbrs = _edge_removal_neighbours(graph)
    if allow_edge_insertion:
        nbrs += _edge_insertion_neighbours(graph, max_new_edges=max_new_edges)
    if allow_node_removal:
        nbrs += _node_removal_neighbours(graph)
    return nbrs


# ─────────────────────────────────────────────────────────────────────────────
# Graph-edit distance (structural, approximate)
# ─────────────────────────────────────────────────────────────────────────────


def _ged(g1: HeteroData, g2: HeteroData) -> int:
    """
    Approximate graph-edit distance based on edge-set symmetric difference.
    One unit of cost per inserted/deleted edge, plus per inserted/deleted node.
    """
    cost = 0
    for et in set(g1.edge_types) | set(g2.edge_types):
        e1 = (
            set(map(tuple, g1[et].edge_index.t().tolist()))
            if et in g1.edge_types
            else set()
        )
        e2 = (
            set(map(tuple, g2[et].edge_index.t().tolist()))
            if et in g2.edge_types
            else set()
        )
        cost += len(e1.symmetric_difference(e2))
    for nt in set(g1.node_types) | set(g2.node_types):
        n1 = g1[nt].num_nodes if nt in g1.node_types else 0
        n2 = g2[nt].num_nodes if nt in g2.node_types else 0
        cost += abs(n1 - n2)
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# Diff between two graphs → list of edits
# ─────────────────────────────────────────────────────────────────────────────


def _compute_edits(original: HeteroData, counterfactual: HeteroData) -> list[dict]:
    """
    Compute the list of atomic edits that transform `original` into
    `counterfactual`.  Handles edge additions/removals and node additions/
    removals (by count; features of added nodes use zero-vectors as placeholders
    since we don't track exact feature assignments across the random walk).
    """
    edits: list[dict] = []

    # ── Nodes ──────────────────────────────────────────────────────────────
    for nt in set(original.node_types) | set(counterfactual.node_types):
        n_orig = original[nt].num_nodes if nt in original.node_types else 0
        n_cf = counterfactual[nt].num_nodes if nt in counterfactual.node_types else 0
        if n_cf < n_orig:
            for idx in range(n_cf, n_orig):
                edits.append(
                    {"action": "remove_node", "node_type": nt, "node_idx": idx}
                )
        elif n_cf > n_orig:
            feat_dim = (
                original[nt].x.size(1)
                if nt in original.node_types and hasattr(original[nt], "x")
                else 1
            )
            for _ in range(n_cf - n_orig):
                edits.append(
                    {
                        "action": "add_node",
                        "node_type": nt,
                        "features": torch.zeros(feat_dim),
                    }
                )

    # ── Edges ───────────────────────────────────────────────────────────────
    for et in set(original.edge_types) | set(counterfactual.edge_types):
        e_orig = (
            set(map(tuple, original[et].edge_index.t().tolist()))
            if et in original.edge_types
            else set()
        )
        e_cf = (
            set(map(tuple, counterfactual[et].edge_index.t().tolist()))
            if et in counterfactual.edge_types
            else set()
        )
        for s, d in e_orig - e_cf:
            edits.append({"action": "remove_edge", "edge_type": et, "src": s, "dst": d})
        for s, d in e_cf - e_orig:
            edits.append(
                {
                    "action": "add_edge",
                    "edge_type": et,
                    "src": s,
                    "dst": d,
                    "features": None,
                }
            )

    return edits


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────


def gcf_explain(
    predict_fn: Callable[[HeteroData], int],
    graph: HeteroData,
    target_class: int,
    *,
    # VRRW hyper-parameters (paper defaults)
    num_steps: int = 50_000,  # M  – total random-walk steps
    teleport_prob: float = 0.1,  # τ  – probability of teleporting back to origin
    alpha: float = 0.5,  # α  – reinforcement weight in transition prob
    distance_threshold: float = 0.1,  # θ  – max normalised GED to count as "covered"
    max_distance: int
    | None = None,  # hard GED cap from origin; defaults to total_edges
    # Edit-space options
    allow_node_removal: bool = True,
    allow_edge_insertion: bool = True,
    max_new_edges_per_type: int = 5,  # cap random edge-insertion candidates
    # Misc
    seed: int = 42,
) -> CounterfactualEdit:
    """
    Find a minimal set of graph edits that causes `model` to predict
    `target_class` for the (modified) `graph`.

    Parameters
    ----------
    predict_fn     : Callable that takes a HeteroData graph and returns a
                     predicted class index (int).  Wrap your model however
                     it needs to be called, e.g.:
                         predict_fn = lambda g: int(model(g).argmax())
    graph          : The HeteroData instance to explain.
    target_class   : Desired output class (0 or 1, or any valid class idx).
    num_steps      : Number of VRRW iterations.
    teleport_prob  : Probability of teleporting back to origin per step.
    alpha          : Exponent controlling how strongly past visits are reinforced.
    distance_threshold : Maximum GED (as a fraction of total edges) within which
                         a counterfactual is considered a valid recourse for
                         the original graph.  Used only for early stopping.
    max_distance       : Hard upper bound on the GED between the current walk
                         position and the origin.  Neighbours that would exceed
                         this distance are pruned before sampling, preventing
                         the walk from drifting into degenerate (near-empty)
                         graphs.  Defaults to `total_edges` (i.e. at most as
                         many edits as there are edges in the original graph).
    allow_node_removal : Include node-deletion moves in the edit map.
    allow_edge_insertion : Include edge-insertion moves in the edit map.
    max_new_edges_per_type : How many random edge-insertion candidates to
                             enumerate per edge type per step (performance knob).
    seed           : Random seed for reproducibility.

    Returns
    -------
    CounterfactualEdit
        Dataclass containing the list of edits (possibly empty if none found)
        and metadata.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    original_class = predict_fn(graph)
    result = CounterfactualEdit(
        target_class=target_class,
        original_class=original_class,
    )

    if original_class == target_class:
        # Already predicts the desired class – zero edits needed.
        result.counterfactual_found = True
        return result

    # ── Total edge count for normalising GED ────────────────────────────────
    total_edges = (
        sum(graph[et].edge_index.size(1) for et in graph.edge_types) or 1
    )  # guard against empty graphs

    # Hard GED cap: never walk more than this many edits away from origin.
    # This is the paper's bounded search space (Section 3.2.1).
    _max_dist = max_distance if max_distance is not None else total_edges

    # ── VRRW state ──────────────────────────────────────────────────────────
    current = _clone(graph)
    origin = graph  # teleportation anchor
    current_ged = 0  # GED of current position from origin (maintained cheaply)

    # visit_counts[key] → how many times we visited that graph state
    visit_counts: dict[tuple, float] = {}
    origin_key = _graph_key(origin)
    visit_counts[origin_key] = 1.0

    # Track the best counterfactual found so far (minimum GED to original)
    best_cf: HeteroData | None = None
    best_ged: int = int(1e9)

    # ── Walk ─────────────────────────────────────────────────────────────────
    for step in range(num_steps):
        # 1. Teleport back to origin with probability τ
        if random.random() < teleport_prob:
            current = _clone(origin)
            current_ged = 0
            continue

        # 2. Enumerate single-edit neighbours and prune those that would
        #    exceed the hard distance cap from origin.
        candidates = _get_neighbours(
            current,
            allow_node_removal=allow_node_removal,
            allow_edge_insertion=allow_edge_insertion,
            max_new_edges=max_new_edges_per_type,
        )

        # Each neighbour is exactly one edit away from current, so its GED
        # from origin is at most current_ged + 1 (and at least current_ged - 1).
        # We compute exact GED only for candidates that could still be in-bounds,
        # which avoids calling predict_fn on degenerate graphs entirely.
        neighbours: list[
            tuple[HeteroData, dict, int]
        ] = []  # (graph, edit, ged_from_origin)
        for nbr, edit in candidates:
            ged = _ged(origin, nbr)
            if ged <= _max_dist:
                neighbours.append((nbr, edit, ged))

        if not neighbours:
            # All moves are out of bounds — teleport back.
            current = _clone(origin)
            current_ged = 0
            continue

        # 3. Compute transition weights  w_i = visit_count(n_i)^α
        weights = []
        for nbr, _edit, _ged_val in neighbours:
            k = _graph_key(nbr)
            cnt = visit_counts.get(k, 1.0)
            weights.append(cnt**alpha)

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        # 4. Sample next state
        chosen_idx = random.choices(range(len(neighbours)), weights=probs, k=1)[0]
        next_graph, _, next_ged = neighbours[chosen_idx]
        next_key = _graph_key(next_graph)

        # 5. Update reinforcement counts
        visit_counts[next_key] = visit_counts.get(next_key, 1.0) + 1.0

        # 6. Check if this neighbour is a valid counterfactual.
        #    Wrap in try/except so a broken graph (e.g. isolated nodes that
        #    cause the model to error) never crashes the search — treat it as
        #    non-counterfactual and teleport back to a safe state.
        try:
            pred = predict_fn(next_graph)
        except Exception:
            current = _clone(origin)
            current_ged = 0
            continue

        if pred == target_class:
            if next_ged < best_ged:
                best_ged = next_ged
                best_cf = _clone(next_graph)

            # Single-edit counterfactual — optimal, stop immediately.
            if next_ged == 1:
                break

        current = next_graph
        current_ged = next_ged

        # ── Early stopping every 1000 steps ─────────────────────────────────
        if (step + 1) % 1000 == 0 and best_cf is not None:
            norm_ged = best_ged / total_edges
            if norm_ged <= distance_threshold:
                break

    # ── Build result ─────────────────────────────────────────────────────────
    if best_cf is not None:
        result.counterfactual_found = True
        result.edits = _compute_edits(origin, best_cf)
    else:
        result.counterfactual_found = False
        result.edits = []

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Global explainer
# ─────────────────────────────────────────────────────────────────────────────


def _vrrw_global(
    predict_fn: Callable[[HeteroData], int],
    graphs: list[HeteroData],
    target_class: int,
    *,
    num_steps: int,
    teleport_prob: float,
    alpha: float,
    max_distance: int,
    allow_node_removal: bool,
    allow_edge_insertion: bool,
    max_new_edges_per_type: int,
) -> list[HeteroData]:
    """
    Multi-origin VRRW over a collection of input graphs.

    At each teleportation event a random input graph is chosen as the new
    anchor (instead of always returning to a single origin), so the walk
    explores the edit-map neighbourhood of the *entire* dataset.

    Returns the list of all visited graphs that are predicted as target_class
    (the global counterfactual candidate pool).
    """
    # Shared visit counts across all origins
    visit_counts: dict[tuple, float] = {_graph_key(g): 1.0 for g in graphs}

    # Candidate pool: key → graph  (deduped by structure)
    candidates: dict[tuple, HeteroData] = {}

    # Start from a random input graph
    origin_idx = random.randrange(len(graphs))
    current = _clone(graphs[origin_idx])
    current_origin = graphs[origin_idx]

    # Per-origin GED budget — use the same default as the local explainer
    for step in range(num_steps):
        # 1. Teleport: pick a new random input graph as anchor
        if random.random() < teleport_prob:
            origin_idx = random.randrange(len(graphs))
            current_origin = graphs[origin_idx]
            current = _clone(current_origin)
            continue

        # 2. Enumerate neighbours; prune those beyond the distance budget
        raw_nbrs = _get_neighbours(
            current,
            allow_node_removal=allow_node_removal,
            allow_edge_insertion=allow_edge_insertion,
            max_new_edges=max_new_edges_per_type,
        )
        neighbours: list[tuple[HeteroData, dict, int]] = []
        for nbr, edit in raw_nbrs:
            ged = _ged(current_origin, nbr)
            if ged <= max_distance:
                neighbours.append((nbr, edit, ged))

        if not neighbours:
            origin_idx = random.randrange(len(graphs))
            current_origin = graphs[origin_idx]
            current = _clone(current_origin)
            continue

        # 3. Transition weights ∝ visit_count^α
        weights = [
            visit_counts.get(_graph_key(nbr), 1.0) ** alpha for nbr, _, _ in neighbours
        ]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        # 4. Sample
        chosen_idx = random.choices(range(len(neighbours)), weights=probs, k=1)[0]
        next_graph, _, _ = neighbours[chosen_idx]
        next_key = _graph_key(next_graph)

        # 5. Reinforce
        visit_counts[next_key] = visit_counts.get(next_key, 1.0) + 1.0

        # 6. Check if counterfactual; guard against broken graphs
        try:
            pred = predict_fn(next_graph)
        except Exception:
            origin_idx = random.randrange(len(graphs))
            current_origin = graphs[origin_idx]
            current = _clone(current_origin)
            continue

        if pred == target_class:
            candidates[next_key] = next_graph

        current = next_graph

    return list(candidates.values())


def _covers(
    candidate: HeteroData,
    input_graph: HeteroData,
    distance_threshold: int,
) -> bool:
    """
    A candidate counterfactual 'covers' an input graph when their GED is
    within the distance threshold (paper Section 3.1).
    """
    return _ged(candidate, input_graph) <= distance_threshold


def _greedy_summary(
    candidates: list[HeteroData],
    graphs: list[HeteroData],
    summary_size: int,
    distance_threshold: int,
    alpha: float,
) -> list[int]:
    """
    Greedy submodular maximisation to select `summary_size` candidates that
    maximise weighted coverage (paper Algorithm 2).

    Objective for adding candidate c to current summary S:
        f(c | S) = α · individual_coverage(c)
                 + (1 - α) · marginal_gain(c | S)

    individual_coverage(c) = |{g : covers(c, g)}| / |graphs|
    marginal_gain(c | S)   = |covered_by(c) \ already_covered(S)| / |graphs|

    Returns the indices (into `candidates`) of the chosen representatives.
    """
    n = len(graphs)
    # Pre-compute coverage sets: candidate_idx → set of graph indices covered
    cov_sets: list[set[int]] = [
        {i for i, g in enumerate(graphs) if _covers(c, g, distance_threshold)}
        for c in candidates
    ]

    selected: list[int] = []
    already_covered: set[int] = set()

    for _ in range(min(summary_size, len(candidates))):
        best_score = -1.0
        best_idx = -1
        for j, cov in enumerate(cov_sets):
            if j in selected:
                continue
            individual = len(cov) / n
            marginal = len(cov - already_covered) / n
            score = alpha * individual + (1.0 - alpha) * marginal
            if score > best_score:
                best_score = score
                best_idx = j
        if best_idx == -1 or best_score == 0.0:
            break
        selected.append(best_idx)
        already_covered |= cov_sets[best_idx]

    return selected


def gcf_explain_global(
    predict_fn: Callable[[HeteroData], int],
    graphs: list[HeteroData],
    target_class: int,
    *,
    # Summary
    summary_size: int = 10,
    # VRRW hyper-parameters (paper defaults)
    num_steps: int = 50_000,
    teleport_prob: float = 0.1,
    alpha: float = 0.5,  # also used in greedy summary objective
    distance_threshold: int = 5,  # θ – absolute GED (not normalised here)
    max_distance: int | None = None,
    # Edit-space options
    allow_node_removal: bool = True,
    allow_edge_insertion: bool = True,
    max_new_edges_per_type: int = 5,
    # Misc
    seed: int = 42,
) -> GlobalCounterfactualResult:
    """
    Find a small set of representative counterfactual graphs that globally
    explain the model's behaviour across all `graphs`.

    This implements the full GCFExplainer algorithm (WSDM 2023):
      1. Multi-origin VRRW to build a counterfactual candidate pool.
      2. Greedy submodular summary selection from that pool.

    Parameters
    ----------
    predict_fn      : Callable[[HeteroData], int] — same contract as in
                      gcf_explain.
    graphs          : All input graphs that share the same predicted class.
                      The walk explores the edit-map neighbourhood of the
                      entire collection.
    target_class    : The desired (counterfactual) class.
    summary_size    : Maximum number of representative counterfactuals to
                      return (c in the paper).
    num_steps       : Total VRRW steps shared across all input graphs.
    teleport_prob   : τ — probability of teleporting to a random input graph.
    alpha           : α — balances individual vs. marginal coverage in the
                      greedy objective, and controls VRRW reinforcement.
    distance_threshold : θ — maximum GED for a candidate to 'cover' an input
                      graph.  Unlike the local explainer this is an absolute
                      integer count (edges + nodes), not a fraction.
    max_distance    : Hard cap on how far the walker can drift from its
                      current anchor.  Defaults to distance_threshold.
    allow_node_removal, allow_edge_insertion, max_new_edges_per_type, seed:
                      Same as gcf_explain.

    Returns
    -------
    GlobalCounterfactualResult
        .summary         — list of CounterfactualEdit, one per representative.
        .coverage        — fraction of input graphs covered by the summary.
        .covered_indices — which input graphs are covered.
        .target_class    — the requested target class.
    """
    if not graphs:
        raise ValueError("graphs must be a non-empty list.")

    random.seed(seed)
    torch.manual_seed(seed)

    _max_dist = max_distance if max_distance is not None else distance_threshold

    # ── Phase 1: multi-origin VRRW → candidate pool ─────────────────────────
    candidates = _vrrw_global(
        predict_fn,
        graphs,
        target_class,
        num_steps=num_steps,
        teleport_prob=teleport_prob,
        alpha=alpha,
        max_distance=_max_dist,
        allow_node_removal=allow_node_removal,
        allow_edge_insertion=allow_edge_insertion,
        max_new_edges_per_type=max_new_edges_per_type,
    )

    result = GlobalCounterfactualResult(target_class=target_class)

    if not candidates:
        return result

    # ── Phase 2: greedy summary selection ───────────────────────────────────
    selected_indices = _greedy_summary(
        candidates,
        graphs,
        summary_size=summary_size,
        distance_threshold=distance_threshold,
        alpha=alpha,
    )

    # ── Phase 3: build edits for each representative ─────────────────────────
    # For each selected candidate, find the closest input graph and express
    # the representative as edits relative to that graph.
    for idx in selected_indices:
        cf_graph = candidates[idx]
        # Closest input graph by GED
        closest_graph = min(graphs, key=lambda g: _ged(g, cf_graph))
        original_class = -1
        try:
            original_class = predict_fn(closest_graph)
        except Exception:
            pass
        edits = _compute_edits(closest_graph, cf_graph)
        result.summary.append(
            CounterfactualEdit(
                edits=edits,
                target_class=target_class,
                original_class=original_class,
                counterfactual_found=True,
            )
        )

    # ── Phase 4: compute coverage ────────────────────────────────────────────
    covered: set[int] = set()
    for idx in selected_indices:
        cf_graph = candidates[idx]
        for i, g in enumerate(graphs):
            if _covers(cf_graph, g, distance_threshold):
                covered.add(i)

    result.covered_indices = sorted(covered)
    result.coverage = len(covered) / len(graphs)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: apply edits to produce the actual counterfactual graph
# ─────────────────────────────────────────────────────────────────────────────


def apply_edits(graph: HeteroData, result: CounterfactualEdit) -> HeteroData:
    """
    Apply the edits from a CounterfactualEdit to `graph` and return the
    modified HeteroData object.
    """
    g = _clone(graph)
    for edit in result.edits:
        action = edit["action"]

        if action == "remove_edge":
            et = edit["edge_type"]
            s, d = edit["src"], edit["dst"]
            ei = g[et].edge_index
            mask = ~((ei[0] == s) & (ei[1] == d))
            g[et].edge_index = ei[:, mask]
            if hasattr(g[et], "edge_attr") and g[et].edge_attr is not None:
                g[et].edge_attr = g[et].edge_attr[mask]

        elif action == "add_edge":
            et = edit["edge_type"]
            s, d = edit["src"], edit["dst"]
            new_edge = torch.tensor([[s], [d]], dtype=torch.long)
            g[et].edge_index = torch.cat([g[et].edge_index, new_edge], dim=1)
            if (
                edit.get("features") is not None
                and hasattr(g[et], "edge_attr")
                and g[et].edge_attr is not None
            ):
                g[et].edge_attr = torch.cat(
                    [g[et].edge_attr, edit["features"].unsqueeze(0)], dim=0
                )

        elif action == "remove_node":
            nt = edit["node_type"]
            idx = edit["node_idx"]
            n = g[nt].num_nodes
            mask = torch.ones(n, dtype=torch.bool)
            mask[idx] = False
            if hasattr(g[nt], "x") and g[nt].x is not None:
                g[nt].x = g[nt].x[mask]
            g[nt].num_nodes = int(mask.sum())
            for et in g.edge_types:
                src_t, _, dst_t = et
                ei = g[et].edge_index
                if src_t == nt:
                    valid = ei[0] != idx
                    ei = ei[:, valid]
                    ei[0] -= (ei[0] > idx).long()
                if dst_t == nt:
                    valid2 = ei[1] != idx
                    ei = ei[:, valid2]
                    ei[1] -= (ei[1] > idx).long()
                g[et].edge_index = ei

        elif action == "add_node":
            nt = edit["node_type"]
            feat = edit["features"].unsqueeze(0)
            if hasattr(g[nt], "x") and g[nt].x is not None:
                g[nt].x = torch.cat([g[nt].x, feat], dim=0)
            g[nt].num_nodes = g[nt].num_nodes + 1

    return g


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run as script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HANConv

    def _make_graph(seed: int) -> HeteroData:
        torch.manual_seed(seed)
        g = HeteroData()
        g["paper"].x = torch.randn(5, 8)
        g["author"].x = torch.randn(3, 8)
        g["paper", "cites", "paper"].edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )
        g["author", "writes", "paper"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 2, 4]], dtype=torch.long
        )
        return g

    data = _make_graph(0)

    class SimpleHeteroGNN(nn.Module):
        def __init__(self, metadata):
            super().__init__()
            self.conv = HANConv(-1, 16, heads=1, metadata=metadata)
            self.fc = nn.Linear(16, 2)

        def forward(self, x_dict, edge_index_dict):
            out = self.conv(x_dict, edge_index_dict)
            all_x = torch.cat(list(out.values()), dim=0)
            return self.fc(all_x.mean(0, keepdim=True))

    model = SimpleHeteroGNN(data.metadata())
    model.eval()

    def predict_fn(g: HeteroData) -> int:
        with torch.no_grad():
            out = model(g.x_dict, g.edge_index_dict)
            return int(out.argmax(dim=-1).item())

    target = 1 - predict_fn(data)

    # ── Local ────────────────────────────────────────────────────────────────
    print("=== Local GCFExplainer ===")
    local_result = gcf_explain(
        predict_fn, data, target_class=target, num_steps=500, seed=0
    )
    print(local_result)
    if local_result.counterfactual_found:
        cf = apply_edits(data, local_result)
        print(f"  CF prediction: {predict_fn(cf)}  (target: {target})")

    # ── Global ───────────────────────────────────────────────────────────────
    print("\n=== Global GCFExplainer ===")
    dataset = [_make_graph(i) for i in range(5)]
    global_result = gcf_explain_global(
        predict_fn,
        dataset,
        target_class=target,
        summary_size=3,
        num_steps=500,
        seed=0,
    )
    print(global_result)
    for i, edit in enumerate(global_result.summary):
        print(f"  Representative {i}: {edit}")
