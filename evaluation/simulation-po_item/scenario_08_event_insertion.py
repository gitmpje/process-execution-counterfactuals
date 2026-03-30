"""
scenario_08_event_insertion.py
===============================
Scenario: Event node insertion
-------------------------------
Anomaly  : `send_reminder` is skipped for anomalous POs.  Those POs jump
           directly from send_invoice → pay_order without the reminder step,
           so `PO.reminder_count` remains 0 in the OCEL.
Label    : abnormal = True when a PO reaches complete_order with
           reminder_count == 0  (reminder was never sent).
Counterfactual expected: insert the `send_reminder` event node.

Implementation: we configure `send_reminder_mandatory=False` globally in base
so no reminder fires, then selectively re-enable it for *normal* POs by
wrapping the guard.
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
    output_prefix="scenario_08",
    seed=707,
    # Start with reminder OFF globally; we will enable it for normal POs only
    send_reminder_mandatory=False,
)


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

    # Anomalous POs: reminder is SKIPPED (reminder_count stays 0)
    # Normal POs   : reminder fires (reminder_count becomes 1)
    anomalous_ids = make_anomalous_ids(po_tokens, CONFIG)
    labels = {po["PO_id"]: (po["PO_id"] in anomalous_ids) for po in po_tokens}

    print(
        f"Injected {len(anomalous_ids)}/{CONFIG['n_pos']} anomalous POs "
        f"(send_reminder skipped → reminder_count stays 0)"
    )

    for po in po_tokens:
        sim.PO_arrival.put(po)
    for it in item_tokens:
        sim.item_arrival.put(it)

    # Patch reminder guard: fires only for NORMAL POs (reminder_count == 0 AND not anomalous)
    def selective_reminder_guard(billed_PO):
        if billed_PO["PO_id"] in anomalous_ids:
            return False  # skip reminder for anomalous POs
        return billed_PO["reminder_count"] == 0

    # Patch pay_order guard: normal POs need reminder first; anomalous can pay immediately
    def selective_pay_guard(billed_PO):
        if billed_PO["PO_id"] in anomalous_ids:
            return True  # pay without reminder
        return billed_PO["reminder_count"] >= 1

    sim.send_reminder_event.guard = selective_reminder_guard
    sim.pay_order_event.guard = selective_pay_guard

    run_simulation(sim)

    import json

    label_path = f"{CONFIG['output_prefix']}-labels.json"
    with open(label_path, "w") as f:
        json.dump(
            {
                "scenario": "event_deletion",
                "description": "send_reminder skipped for anomalous POs; reminder_count stays 0",
                "counterfactual": "delete send_reminder event node from trace",
                "labels": {str(k): v for k, v in labels.items()},
            },
            f,
            indent=2,
        )
    print(f"Labels saved to {label_path}")
