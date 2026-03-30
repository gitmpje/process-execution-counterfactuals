"""
scenario_07_event_deletion.py
================================
Scenario: Event node deletion
--------------------------------
Anomaly  : `send_reminder` fires even for POs that should not need it —
           i.e. for anomalous POs the reminder fires multiple times (or fires
           when it normally would not).
Label    : abnormal = True when a PO's reminder_count > 1 at complete_order.
Counterfactual expected: delete `send_reminder` event node.

Implementation: widen the reminder guard so anomalous POs can fire it twice
(reminder_count ∈ {0, 1}) while normal POs can only fire it once
(reminder_count == 0).  The pay_order guard is adjusted accordingly.
"""

import random
from base import (
    build_model,
    run_simulation,
    make_po_tokens,
    make_item_tokens,
)

CONFIG = dict(
    n_pos=100,
    items_per_po=2,
    simtime=400,
    anomaly_prob=0.3,
    output_prefix="scenario_07",
    seed=808,
    send_reminder_mandatory=True,
)

EXTRA_REMINDER_THRESHOLD = 2  # anomalous POs can send up to 2 reminders


def make_anomalous_ids(po_tokens, cfg):
    rng = random.Random(cfg["seed"])
    return {po["PO_id"] for po in po_tokens if rng.random() < cfg["anomaly_prob"]}


if __name__ == "__main__":
    sim = build_model(CONFIG)

    sim.PO_arrival.marking.clear()
    sim.item_arrival.marking.clear()

    po_tokens = make_po_tokens(
        CONFIG["n_pos"], CONFIG["items_per_po"], seed=CONFIG["seed"]
    )
    item_tokens = make_item_tokens(
        CONFIG["n_pos"], CONFIG["items_per_po"], seed=CONFIG["seed"]
    )

    anomalous_ids = make_anomalous_ids(po_tokens, CONFIG)
    labels = {po["PO_id"]: (po["PO_id"] in anomalous_ids) for po in po_tokens}

    print(
        f"Injected {len(anomalous_ids)}/{CONFIG['n_pos']} anomalous POs "
        f"(extra send_reminder → reminder_count ≥ 2)"
    )

    for po in po_tokens:
        sim.PO_arrival.put(po)
    for it in item_tokens:
        sim.item_arrival.put(it)

    # Patch reminder guard:
    #   normal POs    → reminder fires if reminder_count == 0  (once)
    #   anomalous POs → reminder fires if reminder_count  < 2  (twice)
    def selective_reminder_guard(billed_PO):
        po_id = billed_PO["PO_id"]
        if po_id in anomalous_ids:
            return billed_PO["reminder_count"] < EXTRA_REMINDER_THRESHOLD
        return billed_PO["reminder_count"] == 0

    # Patch pay_order guard:
    #   normal POs    → pay after 1 reminder
    #   anomalous POs → pay only after 2 reminders
    def selective_pay_guard(billed_PO):
        po_id = billed_PO["PO_id"]
        if po_id in anomalous_ids:
            return billed_PO["reminder_count"] >= EXTRA_REMINDER_THRESHOLD
        return billed_PO["reminder_count"] >= 1

    sim.send_reminder_event.guard = selective_reminder_guard
    sim.pay_order_event.guard = selective_pay_guard

    run_simulation(sim)

    import json

    label_path = f"{CONFIG['output_prefix']}-labels.json"
    with open(label_path, "w") as f:
        json.dump(
            {
                "scenario": "event_insertion",
                "description": (
                    f"Extra send_reminder for anomalous POs; reminder_count ≥ {EXTRA_REMINDER_THRESHOLD}"
                ),
                "counterfactual": "insert send_reminder event node into normal trace",
                "labels": {str(k): v for k, v in labels.items()},
            },
            f,
            indent=2,
        )
    print(f"Labels saved to {label_path}")
