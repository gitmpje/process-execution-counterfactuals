"""
scenario_01_object_attr_change.py
==================================
Scenario: Object node attribute change
---------------------------------------
Anomaly  : `PO.items` is set higher than the actual number of items linked to
           that PO, so `complete_order` can never fire (not enough shipped items
           to satisfy the guard).
Label    : abnormal = True  when PO.items > actual item count
Counterfactual expected: decrease `PO.items` to match actual item count.

Only the token-generation step differs from the base process; all events are
identical.
"""

import random
from base import (
    build_model,
    run_simulation,
    make_po_tokens,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = dict(
    n_pos=100,
    items_per_po=2,
    simtime=400,
    anomaly_prob=0.3,
    output_prefix="scenario_01",
    seed=101,
    send_reminder_mandatory=True,
)


def inject_anomalies(po_tokens, cfg):
    """
    For anomalous POs, increase `PO.items` by 1 so the guard in
    complete_order_guard can never be satisfied.

    Returns
    -------
    po_tokens : list[dict]   – modified in-place and returned
    labels    : dict[po_id -> bool]  – True = anomalous
    """
    rng = random.Random(cfg["seed"])
    labels = {}
    for po in po_tokens:
        is_anomalous = rng.random() < cfg["anomaly_prob"]
        labels[po["PO_id"]] = is_anomalous
        if is_anomalous:
            po["items"] += 1  # inflate: now expects more items than exist
            po["is_anomalous"] = True
        else:
            po["is_anomalous"] = False
    return po_tokens, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sim = build_model(CONFIG)

    # Replace default PO tokens with anomaly-injected ones
    # Clear the place first, then re-populate
    sim.PO_arrival.marking.clear()

    po_tokens = make_po_tokens(
        CONFIG["n_pos"], CONFIG["items_per_po"], seed=CONFIG["seed"]
    )
    po_tokens, labels = inject_anomalies(po_tokens, CONFIG)

    anomaly_count = sum(labels.values())
    print(
        f"Injected {anomaly_count}/{CONFIG['n_pos']} anomalous POs "
        f"(PO.items inflated by 1)"
    )

    for po in po_tokens:
        sim.PO_arrival.put(po)

    run_simulation(sim, output_suffix="")

    # Persist ground-truth labels alongside the OCEL
    import json

    label_path = f"{CONFIG['output_prefix']}-labels.json"
    with open(label_path, "w") as f:
        json.dump(
            {
                "scenario": "object_attr_change",
                "description": "PO.items inflated; complete_order cannot fire",
                "counterfactual": "decrease PO.items to actual item count",
                "labels": {str(k): v for k, v in labels.items()},
            },
            f,
            indent=2,
        )
    print(f"Labels saved to {label_path}")
