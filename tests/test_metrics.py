import os
import sys

import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from evaluation.metrics import compute_proximity, matched_diff_to_edits


def test_compute_proximity_pads_mismatched_graph_sizes():
    feat_orig = torch.ones(6, 5)
    feat_cf = torch.zeros(7, 6)
    adj_orig = torch.eye(6)
    adj_cf = torch.eye(7)

    metrics = compute_proximity(feat_orig, adj_orig, feat_cf, adj_cf)

    assert set(metrics.keys()) == {"dist_x", "dist_a", "proximity_x", "proximity_a"}
    assert metrics["dist_x"] >= 0.0
    assert 0.0 <= metrics["proximity_x"] <= 1.0
    assert 0.0 <= metrics["proximity_a"] <= 1.0


def test_matched_diff_to_edits_pads_mismatched_feature_dims():
    feat_orig = torch.ones(2, 4)
    feat_cf = torch.zeros(3, 5)
    adj_orig = torch.eye(2)
    adj_cf = torch.eye(3)

    edits = matched_diff_to_edits(adj_orig, adj_cf, feat_orig, feat_cf)

    assert isinstance(edits, list)
