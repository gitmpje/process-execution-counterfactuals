"""
nseg_hetero_counterfactual.py
─────────────────────────────
Native PyTorch Geometric re-implementation of NSEG for heterogeneous graphs
with HANConv.

Usage
─────
    from nseg_hetero_counterfactual import NSEGHetero, generate_counterfactual

    result = generate_counterfactual(
        hetero_data       = data,
        model             = trained_model,
        num_epochs        = 300,
        lr                = 5e-3,
        objective         = "PNS",
        alpha_e           = 0.01,
        alpha_f           = 0.01,
        beta_e            = 0.01,
        beta_f            = 0.01,
        edge_threshold    = 0.5,
        feature_threshold = 0.5,
    )
    print(result.summary())
    cf_graph = result.apply_to(data)

Tuning guide
────────────
If you still get everything masked after the fix:
  • Run explainer.diagnose() first — it prints loss components and mask stats
    without running the full optimisation.
  • Start with objective="sufficiency" only, alpha_e=alpha_f=0 (no sparsity).
    If the sufficiency loss alone can't be minimised, your model forward pass
    has a problem (wrong loss type, edge_weight not propagated, etc.).
  • Increase alpha_e / alpha_f gradually once sufficiency works.
  • "necessity" on its own should drive masks toward 0 (zeroing is cheap).
    Use "PNS" to balance both objectives.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData


# ─────────────────────────────────────────────────────────────────────────────
# Result data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EdgeChange:
    """One edge whose mask fell below the threshold (removed in the CF)."""

    edge_type: Tuple[str, str, str]
    src_node: int
    dst_node: int
    mask_value: float


@dataclass
class FeatureChange:
    """One (node, feature-dim) pair zeroed in the counterfactual."""

    node_type: str
    node_idx: int
    feat_dim: int
    mask_value: float
    original_value: float


@dataclass
class CounterfactualResult:
    """
    Everything needed to understand and reconstruct the counterfactual.

    Key attributes
    ──────────────
    original_pred     : model output on the original graph
    cf_pred           : model output on the counterfactual graph
    removed_edges     : edges removed in the CF (mask < edge_threshold)
    kept_edges        : edges retained
    edge_masks        : Dict[edge_type → Tensor(E,)] sigmoid mask per edge
    zeroed_features   : (node, dim) pairs zeroed in the CF
    kept_features     : (node, dim) pairs kept
    feature_masks     : Dict[node_type → Tensor(N, F)] sigmoid mask
    loss_history      : list of (total, suff, nec, sparse, ent) per epoch
    """

    original_pred: Tensor
    cf_pred: Optional[Tensor]

    removed_edges: List[EdgeChange]
    kept_edges: List[EdgeChange]
    edge_masks: Dict[Tuple[str, str, str], Tensor]
    edge_threshold: float

    zeroed_features: List[FeatureChange]
    kept_features: List[FeatureChange]
    feature_masks: Dict[str, Tensor]
    feature_threshold: float

    # Each entry: (total, suff, nec, sparsity, entropy)
    loss_history: List[Tuple[float, float, float, float, float]] = field(
        default_factory=list
    )

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "  NSEG Counterfactual Result  (native PyG implementation)",
            "=" * 65,
            f"  Original prediction  : {self.original_pred.tolist()}",
            f"  Counterfactual pred  : "
            f"{self.cf_pred.tolist() if self.cf_pred is not None else 'N/A'}",
        ]
        if self.loss_history:
            t0 = self.loss_history[0]
            tf = self.loss_history[-1]
            lines.append(f"  Loss (total)         : {t0[0]:.4f} → {tf[0]:.4f}")
            lines.append(
                f"  Loss components (final): "
                f"suff={tf[1]:.4f}  nec={tf[2]:.4f}  "
                f"sparse={tf[3]:.4f}  ent={tf[4]:.4f}"
            )
        lines += [
            "",
            "  -- Edge counterfactual --",
            f"  Threshold            : {self.edge_threshold}",
            f"  Total edges          : {len(self.removed_edges) + len(self.kept_edges)}",
            f"  Edges REMOVED (CF)   : {len(self.removed_edges)}",
            f"  Edges KEPT    (CF)   : {len(self.kept_edges)}",
            "  Changes:",
        ]
        if not self.removed_edges:
            lines.append("    (none)")
        else:
            for ec in sorted(self.removed_edges, key=lambda e: e.mask_value):
                src_t, rel, dst_t = ec.edge_type
                lines.append(
                    f"    [{rel}]  {src_t}[{ec.src_node}]"
                    f" -> {dst_t}[{ec.dst_node}]  mask={ec.mask_value:.4f}"
                )
        lines += [
            "",
            "  -- Feature counterfactual --",
            f"  Threshold            : {self.feature_threshold}",
            f"  Total (node,feat)    : "
            f"{len(self.zeroed_features) + len(self.kept_features)}",
            f"  Features ZEROED (CF) : {len(self.zeroed_features)}",
            "  Top-20 changes (lowest mask = most counterfactual):",
        ]
        shown = sorted(self.zeroed_features, key=lambda f: f.mask_value)[:20]
        if not shown:
            lines.append("    (none)")
        else:
            for fc in shown:
                lines.append(
                    f"    {fc.node_type}[{fc.node_idx}] dim={fc.feat_dim}"
                    f"  orig={fc.original_value:.4f}  mask={fc.mask_value:.4f}"
                )
            if len(self.zeroed_features) > 20:
                lines.append(f"    ... and {len(self.zeroed_features) - 20} more")
        lines.append("=" * 65)
        return "\n".join(lines)

    def apply_to(self, hetero_data: HeteroData) -> HeteroData:
        """Return a deep copy with the counterfactual applied."""
        cf = copy.deepcopy(hetero_data)
        self._apply_edges(cf)
        self._apply_features(cf)
        return cf

    def _apply_edges(self, cf: HeteroData) -> None:
        remove_by_type: Dict[Tuple[str, str, str], set] = {}
        for ec in self.removed_edges:
            remove_by_type.setdefault(ec.edge_type, set()).add(
                (ec.src_node, ec.dst_node)
            )
        for etype, pairs in remove_by_type.items():
            store = cf[etype]
            ei = store.edge_index
            keep = torch.ones(ei.size(1), dtype=torch.bool)
            for e in range(ei.size(1)):
                if (int(ei[0, e]), int(ei[1, e])) in pairs:
                    keep[e] = False
            store.edge_index = ei[:, keep]
            if hasattr(store, "edge_attr") and store.edge_attr is not None:
                store.edge_attr = store.edge_attr[keep]

    def _apply_features(self, cf: HeteroData) -> None:
        for fc in self.zeroed_features:
            store = cf[fc.node_type]
            if hasattr(store, "x") and store.x is not None:
                store.x[fc.node_idx, fc.feat_dim] = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_pred": self.original_pred.tolist(),
            "cf_pred": self.cf_pred.tolist() if self.cf_pred is not None else None,
            "edge_threshold": self.edge_threshold,
            "feature_threshold": self.feature_threshold,
            "edge_masks": {str(k): v.tolist() for k, v in self.edge_masks.items()},
            "feature_masks": {k: v.tolist() for k, v in self.feature_masks.items()},
            "removed_edges": [
                {
                    "edge_type": list(ec.edge_type),
                    "src_node": ec.src_node,
                    "dst_node": ec.dst_node,
                    "mask_value": ec.mask_value,
                }
                for ec in self.removed_edges
            ],
            "zeroed_features": [
                {
                    "node_type": fc.node_type,
                    "node_idx": fc.node_idx,
                    "feat_dim": fc.feat_dim,
                    "original_value": fc.original_value,
                    "mask_value": fc.mask_value,
                }
                for fc in self.zeroed_features
            ],
            "loss_history": self.loss_history,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss helper — handles logits OR log-softmax output
# ─────────────────────────────────────────────────────────────────────────────


def _ce_loss(logits_or_logprobs: Tensor, target: Tensor) -> Tensor:
    """
    Cross-entropy that works whether the model returns raw logits or
    log-softmax probabilities.

    Detection heuristic: if all values are ≤ 0 and the row sums to ≈ 0 in
    exp-space (i.e. logsumexp ≈ 0), it looks like log-probabilities.
    This is reliable for the typical binary/multi-class classification case.
    """
    y = logits_or_logprobs
    if y.ndim == 1:
        y = y.unsqueeze(0)
    # Check if output looks like log-probabilities (all ≤ 0, logsumexp ≈ 0)
    # if y.max().item() <= 0.0 and abs(y.logsumexp(dim=-1).mean().item()) < 0.1:
    #     return F.nll_loss(y, target)
    return F.cross_entropy(y, target)


# ─────────────────────────────────────────────────────────────────────────────
# Masked HeteroData builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_masked_data(
    data: HeteroData,
    edge_masks: Dict[Tuple[str, str, str], Tensor],
    feat_masks: Dict[str, Tensor],
    complement: bool,
    device: torch.device,
) -> HeteroData:
    """
    Build a soft-masked HeteroData for one forward pass.

    Feature masking
    ───────────────
    x_eff = mask_f * x            (sufficiency)
    x_eff = (1 - mask_f) * x      (necessity / complement)

    Edge masking via neighbour feature pre-scaling
    ───────────────────────────────────────────────
    HANConv (and most hetero convs) do not read edge_weight. We therefore
    implement edge masking by scaling each source node's features by the
    *mean* of its outgoing edge masks before the conv. This is equivalent
    to soft-removing edges while keeping the graph structure intact (needed
    for gradients to flow through the conv).

    Concretely, for each edge type (src→dst):
        scale[src_node] = mean of edge_masks for edges leaving that src node
        x_src_eff *= scale[src_node]   (broadcast over feature dim)

    This gives the edge mask a real gradient path through the model.
    """
    # ── per-node edge-mask scale factors ─────────────────────────────────────
    # Accumulate the mean outgoing mask per source node, per node type.
    node_edge_scale: Dict[str, Tensor] = {}

    for ntype in data.node_types:
        n = data[ntype].num_nodes
        node_edge_scale[ntype] = torch.ones(n, device=device)

    for etype in data.edge_types:
        src_type, _, _ = etype
        ei = data[etype].edge_index.to(device)  # (2, E)
        m_e = edge_masks[etype].to(device)  # (E,)
        if complement:
            m_e = 1.0 - m_e

        n_src = data[src_type].num_nodes
        # Sum of masks per source node
        scale = torch.zeros(n_src, device=device)
        count = torch.zeros(n_src, device=device)
        scale.scatter_add_(0, ei[0], m_e)
        count.scatter_add_(0, ei[0], torch.ones_like(m_e))
        # Nodes with no outgoing edges of this type keep scale 1
        has_edges = count > 0
        scale[has_edges] = scale[has_edges] / count[has_edges]
        scale[~has_edges] = 1.0
        # Multiply into existing scale (handles multiple edge types)
        node_edge_scale[src_type] = node_edge_scale[src_type] * scale

    # ── build masked HeteroData ───────────────────────────────────────────────
    masked = HeteroData()

    for ntype in data.node_types:
        store = data[ntype]
        m_f = feat_masks[ntype].to(device)  # (N, F)
        if complement:
            m_f = 1.0 - m_f
        x = (
            store.x.to(device)
            if (hasattr(store, "x") and store.x is not None)
            else torch.zeros(store.num_nodes, 1, device=device)
        )
        # Apply feature mask then edge scale (both differentiable)
        edge_scale = node_edge_scale[ntype].unsqueeze(-1)  # (N, 1)
        masked[ntype].x = m_f * x * edge_scale
        masked[ntype].num_nodes = store.num_nodes

    for etype in data.edge_types:
        store = data[etype]
        masked[etype].edge_index = store.edge_index.to(device)
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            masked[etype].edge_attr = store.edge_attr.to(device)

    return masked


# ─────────────────────────────────────────────────────────────────────────────
# NSEGHetero
# ─────────────────────────────────────────────────────────────────────────────


class NSEGHetero:
    """
    Native PyG NSEG for heterogeneous graphs with HANConv.

    Parameters
    ──────────
    model             : nn.Module — HeteroData → Tensor (logits or log-probs).
    hetero_data       : HeteroData — single graph (not batched).
    num_epochs        : int — optimisation steps.
    lr                : float — Adam learning rate.
    objective         : "sufficiency" | "necessity" | "PNS"
    alpha_e           : float — edge sparsity coefficient (default 0.01).
    beta_e            : float — edge entropy coefficient (default 0.01).
    alpha_f           : float — feature sparsity coefficient (default 0.01).
    beta_f            : float — feature entropy coefficient (default 0.01).
    edge_threshold    : float — mask < threshold → edge removed in CF.
    feature_threshold : float — mask < threshold → feature zeroed in CF.
    explain_features  : bool — if False, feature masks are fixed at 1.
    log_every         : int — print loss breakdown every N epochs (0 = silent).
    device            : torch.device — defaults to CUDA if available.
    model_kwargs      : extra kwargs forwarded to model.forward().

    Tuning alpha / beta
    ───────────────────
    These control how sparse / binary the masks become:
      • alpha too high → everything gets zeroed (sparsity wins)
      • alpha too low  → masks stay near 0.5, no clear explanation
      • beta too high  → masks snap to 0/1 too early (before prediction loss
                         has had a chance to guide them)
    A safe starting point: alpha_e=alpha_f=0.01, beta_e=beta_f=0.01.
    Use diagnose() to check loss component magnitudes before tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        hetero_data: HeteroData,
        num_epochs: int = 300,
        lr: float = 5e-3,
        objective: str = "PNS",
        alpha_e: float = 0.01,
        beta_e: float = 0.01,
        alpha_f: float = 0.01,
        beta_f: float = 0.01,
        edge_threshold: float = 0.5,
        feature_threshold: float = 0.5,
        explain_features: bool = True,
        log_every: int = 50,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        target_class: Optional[int] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device).eval()
        self.data = hetero_data.to(device)
        self.num_epochs = num_epochs
        self.lr = lr
        self.objective = objective
        self.alpha_e = alpha_e
        self.beta_e = beta_e
        self.alpha_f = alpha_f
        self.beta_f = beta_f
        self.edge_threshold = edge_threshold
        self.feature_threshold = feature_threshold
        self.explain_features = explain_features
        self.log_every = log_every
        self.model_kwargs = model_kwargs or {}

        for p in self.model.parameters():
            p.requires_grad_(False)

        # Compute original prediction once
        with torch.no_grad():
            self._orig_pred: Tensor = self._forward(self.data)
        pred_flat = self._orig_pred.squeeze()
        self._target_class: int = (
            target_class if target_class is not None else int(pred_flat.argmax())
        )
        # Counterfactual target = the class the model should predict AFTER masking.
        # For binary: the other class. For multi-class: the second-highest class.
        n_classes = int(pred_flat.shape[0]) if pred_flat.ndim > 0 else 2
        if n_classes == 2:
            self._cf_target_class: int = 1 - self._target_class
        else:
            sorted_classes = pred_flat.argsort(descending=True)
            self._cf_target_class = int(sorted_classes[1])

        # Initialise mask logits at 0  →  sigmoid = 0.5
        self._edge_logits: Dict[Tuple[str, str, str], nn.Parameter] = {
            etype: nn.Parameter(
                torch.zeros(self.data[etype].edge_index.size(1), device=device)
            )
            for etype in self.data.edge_types
        }

        self._feat_logits: Dict[str, nn.Parameter] = {}
        for ntype in self.data.node_types:
            store = self.data[ntype]
            has_x = hasattr(store, "x") and store.x is not None
            if not explain_features or not has_x:
                # Fixed at sigmoid(+10) ≈ 1 — no masking
                N = store.num_nodes
                F_ = store.x.size(1) if has_x else 1
                self._feat_logits[ntype] = nn.Parameter(
                    torch.full((N, F_), 10.0, device=device),
                    requires_grad=False,
                )
            else:
                N, F_ = store.x.shape
                self._feat_logits[ntype] = nn.Parameter(
                    torch.zeros(N, F_, device=device)
                )

    # ── forward helper ────────────────────────────────────────────────────────

    def _forward(self, data: HeteroData) -> Tensor:
        """Call model, trying HeteroData directly then common dict signatures."""
        try:
            out = self.model(data.x_dict, **self.model_kwargs)
        except TypeError:
            try:
                out = self.model(data.x_dict, data.edge_index_dict, **self.model_kwargs)
            except TypeError:
                batch_dict = data.get("batch_dict", None)
                if batch_dict is None:
                    batch_dict = {
                        ntype: torch.zeros(
                            data[ntype].num_nodes, dtype=torch.long, device=data[ntype].x.device
                        )
                        for ntype in data.node_types
                    }
                out = self.model(
                    data.x_dict,
                    data.edge_index_dict,
                    batch_dict,
                    **self.model_kwargs,
                )
        if out.ndim == 1:
            out = out.unsqueeze(0)
        return out

    # ── mask accessors ────────────────────────────────────────────────────────

    def _edge_masks(self) -> Dict[Tuple[str, str, str], Tensor]:
        return {et: torch.sigmoid(lg) for et, lg in self._edge_logits.items()}

    def _feat_masks(self) -> Dict[str, Tensor]:
        return {nt: torch.sigmoid(lg) for nt, lg in self._feat_logits.items()}

    # ── individual loss terms ─────────────────────────────────────────────────

    def _loss_prediction(
        self,
        e_masks: Dict[Tuple[str, str, str], Tensor],
        f_masks: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Returns (sufficiency_loss, necessity_loss) — both scalar tensors."""
        target = torch.tensor([self._target_class], device=self.device)
        zero = torch.tensor(0.0, device=self.device)

        l_suff = zero
        if self.objective in ("sufficiency", "PNS"):
            suff_data = _build_masked_data(
                self.data, e_masks, f_masks, complement=False, device=self.device
            )
            y_suff = self._forward(suff_data)
            l_suff = _ce_loss(y_suff, target)

        l_nec = zero
        if self.objective in ("necessity", "PNS"):
            nec_data = _build_masked_data(
                self.data, e_masks, f_masks, complement=True, device=self.device
            )
            y_nec = self._forward(nec_data)
            # Necessity: complement should predict the CF class (not original).
            # Using the CF target directly is more stable than negating the
            # original-class CE, because -CE is unbounded below and causes the
            # optimiser to drive all masks to 0 (empty graph → wrong class
            # trivially).  Pushing toward the specific other class is a bounded,
            # well-defined objective and prevents the collapse.
            cf_target = torch.tensor([self._cf_target_class], device=self.device)
            l_nec = _ce_loss(y_nec, cf_target)

        return l_suff, l_nec

    def _loss_sparsity(
        self,
        e_masks: Dict[Tuple[str, str, str], Tensor],
        f_masks: Dict[str, Tensor],
    ) -> Tensor:
        """Normalised L1 — stays O(1) regardless of graph size."""
        loss = torch.tensor(0.0, device=self.device)
        total_e = sum(m.numel() for m in e_masks.values())
        if total_e > 0:
            loss = loss + self.alpha_e * (
                sum(m.sum() for m in e_masks.values()) / total_e
            )
        if self.explain_features:
            total_f = sum(
                m.numel()
                for m in f_masks.values()
                if m.requires_grad or m.grad_fn is not None
            )
            trainable_f = {
                nt: m
                for nt, m in f_masks.items()
                if self._feat_logits[nt].requires_grad
            }
            if trainable_f:
                total_f = sum(m.numel() for m in trainable_f.values())
                loss = loss + self.alpha_f * (
                    sum(m.sum() for m in trainable_f.values()) / total_f
                )
        return loss

    def _loss_entropy(
        self,
        e_masks: Dict[Tuple[str, str, str], Tensor],
        f_masks: Dict[str, Tensor],
    ) -> Tensor:
        """Binary entropy regularisation — pushes masks toward 0 or 1."""
        eps = 1e-8

        def _h(m: Tensor) -> Tensor:
            return -(m * (m + eps).log() + (1 - m) * (1 - m + eps).log()).mean()

        loss = torch.tensor(0.0, device=self.device)
        for m in e_masks.values():
            loss = loss - self.beta_e * _h(m)
        trainable_f = {
            nt: m for nt, m in f_masks.items() if self._feat_logits[nt].requires_grad
        }
        for m in trainable_f.values():
            loss = loss - self.beta_f * _h(m)
        return loss

    # ── diagnostics ───────────────────────────────────────────────────────────

    def diagnose(self) -> None:
        """
        Print a snapshot of loss components and mask statistics at the current
        mask values (logits = 0, i.e. all masks = 0.5 before any training).

        Use this to check that:
          (a) prediction loss is non-zero and of reasonable magnitude
          (b) sparsity and entropy losses are << prediction loss
          (c) the model forward pass completes without error
        """
        print("=" * 60)
        print("  NSEGHetero diagnostics")
        print("=" * 60)
        print(f"  Original prediction : {self._orig_pred.tolist()}")
        print(f"  Target class        : {self._target_class}")
        print(f"  CF target class     : {self._cf_target_class}")
        print(f"  Objective           : {self.objective}")
        print()

        e_masks = self._edge_masks()
        f_masks = self._feat_masks()

        l_suff, l_nec = self._loss_prediction(e_masks, f_masks)
        l_sparse = self._loss_sparsity(e_masks, f_masks)
        l_ent = self._loss_entropy(e_masks, f_masks)
        total = l_suff + l_nec + l_sparse + l_ent

        print("  Loss @ init (all masks=0.5):")
        print(f"    sufficiency : {l_suff.item():.4f}")
        print(f"    necessity   : {l_nec.item():.4f}")
        print(f"    sparsity    : {l_sparse.item():.4f}")
        print(f"    entropy     : {l_ent.item():.4f}")
        print(f"    TOTAL       : {total.item():.4f}")
        print()

        # Sanity: what does model predict on fully-zeroed input?
        zero_data = HeteroData()
        for ntype in self.data.node_types:
            store = self.data[ntype]
            N = store.num_nodes
            F_ = store.x.size(1) if (hasattr(store, "x") and store.x is not None) else 1
            zero_data[ntype].x = torch.zeros(N, F_, device=self.device)
            zero_data[ntype].num_nodes = N
        for etype in self.data.edge_types:
            zero_data[etype].edge_index = self.data[etype].edge_index.to(self.device)
        with torch.no_grad():
            y_zero = self._forward(zero_data)
        print(f"  Model output on fully-zeroed input: {y_zero.tolist()}")
        print("  (If this is the SAME class as original, necessity will be")
        print("   very hard to satisfy and the optimiser may over-zero.)")
        print()

        print("  Graph size:")
        for ntype in self.data.node_types:
            store = self.data[ntype]
            F_ = store.x.size(1) if (hasattr(store, "x") and store.x is not None) else 0
            print(f"    {ntype}: {store.num_nodes} nodes × {F_} features")
        for etype in self.data.edge_types:
            E = self.data[etype].edge_index.size(1)
            print(f"    {etype}: {E} edges")
        print("=" * 60)

    # ── main optimisation ─────────────────────────────────────────────────────

    def explain(self, **extra_model_kwargs: Any) -> CounterfactualResult:
        """Run optimisation and return a CounterfactualResult."""
        if extra_model_kwargs:
            self.model_kwargs = {**self.model_kwargs, **extra_model_kwargs}

        all_params = list(self._edge_logits.values()) + [
            p for p in self._feat_logits.values() if p.requires_grad
        ]
        optimiser = torch.optim.Adam(all_params, lr=self.lr)

        loss_history: List[Tuple[float, float, float, float, float]] = []

        pbar = tqdm(total=self.num_epochs)
        for epoch in range(self.num_epochs):
            optimiser.zero_grad()

            e_masks = self._edge_masks()
            f_masks = self._feat_masks()

            l_suff, l_nec = self._loss_prediction(e_masks, f_masks)
            l_sparse = self._loss_sparsity(e_masks, f_masks)
            l_ent = self._loss_entropy(e_masks, f_masks)
            loss = l_suff + l_nec + l_sparse + l_ent

            loss.backward()
            optimiser.step()

            record = (
                loss.item(),
                l_suff.item(),
                l_nec.item(),
                l_sparse.item(),
                l_ent.item(),
            )
            loss_history.append(record)

            pbar.update(1)
            if self.log_every > 0 and (epoch + 1) % self.log_every == 0:
                # Also print mean mask values to detect collapse
                mean_e = (
                    torch.stack([m.detach().mean() for m in e_masks.values()])
                    .mean()
                    .item()
                )
                trainable_f = [
                    m
                    for nt, m in f_masks.items()
                    if self._feat_logits[nt].requires_grad
                ]
                mean_f = (
                    torch.stack([m.detach().mean() for m in trainable_f]).mean().item()
                    if trainable_f
                    else float("nan")
                )
                print(
                    f"  [NSEG] {epoch + 1:>4d}/{self.num_epochs}"
                    f"  loss={loss.item():.4f}"
                    f"  (suf={l_suff.item():.3f}"
                    f"  nec={l_nec.item():.3f}"
                    f"  spar={l_sparse.item():.3f}"
                    f"  ent={l_ent.item():.3f})"
                    f"  mean_mask_e={mean_e:.3f}"
                    f"  mean_mask_f={mean_f:.3f}"
                )

        pbar.close()

        # ── extract final masks ───────────────────────────────────────────────
        with torch.no_grad():
            final_edge_masks = {
                et: torch.sigmoid(lg).cpu() for et, lg in self._edge_logits.items()
            }
            final_feat_masks = {
                nt: torch.sigmoid(lg).cpu() for nt, lg in self._feat_logits.items()
            }

        # ── classify edges ────────────────────────────────────────────────────
        removed_edges: List[EdgeChange] = []
        kept_edges: List[EdgeChange] = []
        for etype in self.data.edge_types:
            ei = self.data[etype].edge_index.cpu()
            mask = final_edge_masks[etype]
            for e in range(ei.size(1)):
                mv = mask[e].item()
                ec = EdgeChange(
                    edge_type=etype,
                    src_node=int(ei[0, e]),
                    dst_node=int(ei[1, e]),
                    mask_value=mv,
                )
                (removed_edges if mv < self.edge_threshold else kept_edges).append(ec)

        # ── classify features ─────────────────────────────────────────────────
        zeroed_features: List[FeatureChange] = []
        kept_features: List[FeatureChange] = []
        for ntype in self.data.node_types:
            store = self.data[ntype]
            if not (hasattr(store, "x") and store.x is not None):
                continue
            x = store.x.cpu()
            mask = final_feat_masks[ntype]
            for n in range(x.size(0)):
                for f in range(x.size(1)):
                    mv = mask[n, f].item()
                    fc = FeatureChange(
                        node_type=ntype,
                        node_idx=n,
                        feat_dim=f,
                        mask_value=mv,
                        original_value=x[n, f].item(),
                    )
                    (
                        zeroed_features
                        if mv < self.feature_threshold
                        else kept_features
                    ).append(fc)

        # ── counterfactual prediction ─────────────────────────────────────────
        cf_pred: Optional[Tensor] = None
        try:
            partial = CounterfactualResult(
                original_pred=self._orig_pred.cpu(),
                cf_pred=None,
                removed_edges=removed_edges,
                kept_edges=kept_edges,
                edge_masks=final_edge_masks,
                edge_threshold=self.edge_threshold,
                zeroed_features=zeroed_features,
                kept_features=kept_features,
                feature_masks=final_feat_masks,
                feature_threshold=self.feature_threshold,
                loss_history=loss_history,
            )
            cf_data = partial.apply_to(self.data)
            with torch.no_grad():
                cf_pred = self._forward(cf_data).cpu()
        except Exception as exc:
            print(f"[WARNING] CF prediction failed: {exc}", file=sys.stderr)

        return CounterfactualResult(
            original_pred=self._orig_pred.cpu(),
            cf_pred=cf_pred,
            removed_edges=removed_edges,
            kept_edges=kept_edges,
            edge_masks=final_edge_masks,
            edge_threshold=self.edge_threshold,
            zeroed_features=zeroed_features,
            kept_features=kept_features,
            feature_masks=final_feat_masks,
            feature_threshold=self.feature_threshold,
            loss_history=loss_history,
        )

    def __call__(self, **kwargs: Any) -> CounterfactualResult:
        return self.explain(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def generate_counterfactual(
    hetero_data: HeteroData,
    model: nn.Module,
    num_epochs: int = 300,
    lr: float = 5e-3,
    objective: str = "PNS",
    alpha_e: float = 0.01,
    beta_e: float = 0.01,
    alpha_f: float = 0.01,
    beta_f: float = 0.01,
    edge_threshold: float = 0.5,
    feature_threshold: float = 0.5,
    explain_features: bool = True,
    log_every: int = 50,
    device: Optional[torch.device] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    run_diagnostics: bool = False,
    target_class: Optional[int] = None,
) -> CounterfactualResult:
    """
    Convenience wrapper. All parameters are forwarded to NSEGHetero.

    Set run_diagnostics=True to print loss component breakdown before
    optimisation starts — useful for tuning alpha/beta.

    target_class: override the class to explain (default: model's argmax).
    """
    explainer = NSEGHetero(
        model=model,
        hetero_data=hetero_data,
        num_epochs=num_epochs,
        lr=lr,
        objective=objective,
        alpha_e=alpha_e,
        beta_e=beta_e,
        alpha_f=alpha_f,
        beta_f=beta_f,
        edge_threshold=edge_threshold,
        feature_threshold=feature_threshold,
        explain_features=explain_features,
        log_every=log_every,
        device=device,
        model_kwargs=model_kwargs,
        target_class=target_class,
    )
    if run_diagnostics:
        explainer.diagnose()
    return explainer.explain()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.nn import HANConv, Linear

    torch.manual_seed(0)

    data = HeteroData()
    data["paper"].x = torch.randn(5, 16)
    data["author"].x = torch.randn(3, 16)
    data["paper", "cites", "paper"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )
    data["author", "writes", "paper"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 2, 4]], dtype=torch.long
    )

    class ToyHAN(nn.Module):
        def __init__(self):
            super().__init__()
            metadata = (
                ["paper", "author"],
                [("paper", "cites", "paper"), ("author", "writes", "paper")],
            )
            self.conv = HANConv(16, 8, metadata=metadata, heads=2)
            self.lin = Linear(8, 2)

        def forward(self, d: HeteroData) -> Tensor:
            out = self.conv({nt: d[nt].x for nt in d.node_types}, d.edge_index_dict)
            return self.lin(out["paper"].mean(dim=0, keepdim=True))

    model = ToyHAN()

    result = generate_counterfactual(
        hetero_data=data,
        model=model,
        num_epochs=150,
        lr=5e-3,
        objective="PNS",
        alpha_e=0.01,
        beta_e=0.01,
        alpha_f=0.01,
        beta_f=0.01,
        edge_threshold=0.5,
        feature_threshold=0.5,
        explain_features=True,
        log_every=50,
        run_diagnostics=True,
    )

    print(result.summary())
    cf = result.apply_to(data)
    print("\nEdge counts — original → CF:")
    for et in data.edge_types:
        print(f"  {et}: {data[et].edge_index.size(1)} → {cf[et].edge_index.size(1)}")
