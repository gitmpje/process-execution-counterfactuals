"""
scenario_06_event_substitution.py
===================================
Scenario: Event node substitution
-----------------------------------
Anomaly  : For anomalous POs, `send_reminder` fires *before* `send_invoice`
           instead of after it.  The normal flow is:

             place_order → send_invoice → send_reminder → pay_order → complete_order

           The anomalous flow is:

             place_order → send_reminder → send_invoice → pay_order → complete_order

           Both events fire at their natural delays; no timestamp magnitudes
           are manipulated.  The only observable difference is event ordering.

Label    : abnormal = True when send_reminder precedes send_invoice in the trace.
Counterfactual expected: swap send_reminder and send_invoice in the trace.

Implementation
--------------
A `send_reminder_early` event is added as a self-loop on pipeline_PO,
firing only for anomalous POs (reminder_count == 0).  It uses the same
event name "send_reminder" so it appears identically in the OCEL.
The normal send_reminder on billed_PO is suppressed for anomalous POs
so the reminder does not fire a second time after send_invoice.
pay_order is also adjusted to allow firing without a reminder for
anomalous POs (since reminder_count was already incremented early).
"""

import json
import os
import random
from ocpn_prototypes import OCPNEvent
from base import (
    build_model,
    run_simulation,
    make_po_tokens,
    make_item_tokens,
    send_reminder_behavior,
)

CONFIG = dict(
    n_pos=100,
    items_per_po=2,
    simtime=400,
    anomaly_prob=0.3,
    output_prefix="scenario_06",
    seed=606,
    send_reminder_mandatory=True,
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

    anomalous_ids = make_anomalous_ids(po_tokens, CONFIG)
    labels = {po["PO_id"]: (po["PO_id"] in anomalous_ids) for po in po_tokens}

    print(
        f"Injected {len(anomalous_ids)}/{CONFIG['n_pos']} anomalous POs "
        f"(send_reminder fires before send_invoice)"
    )

    for po in po_tokens:
        sim.PO_arrival.put(po)
    for it in item_tokens:
        sim.item_arrival.put(it)

    # ------------------------------------------------------------------
    # Add send_reminder_early: self-loop on pipeline_PO, anomalous POs only
    # ------------------------------------------------------------------
    OCPNEvent(
        model=sim.model,
        incoming_vars=[(sim.pipeline_PO, False)],
        outgoing_vars=[(sim.pipeline_PO, False)],
        name="send_reminder_early",  # same name: appears as send_reminder in OCEL
        guard=lambda pipeline_PO: (
            pipeline_PO["PO_id"] in anomalous_ids and pipeline_PO["reminder_count"] == 0
        ),
        behavior=send_reminder_behavior,
    )

    # ------------------------------------------------------------------
    # Suppress normal send_reminder on billed_PO for anomalous POs
    # (reminder already fired early; don't fire it again)
    # ------------------------------------------------------------------
    sim.send_reminder_event.guard = lambda billed_PO: (
        billed_PO["PO_id"] not in anomalous_ids and billed_PO["reminder_count"] == 0
    )

    # ------------------------------------------------------------------
    # pay_order: anomalous POs already have reminder_count >= 1 from the
    # early reminder, so the standard guard (>= 1) works unchanged.
    # No patch needed.
    # ------------------------------------------------------------------

    run_simulation(sim)

    # Replace internal event name to align simulated events with OCEL activity labels
    # (send_reminder_early is logically the same as send_reminder in this scenario)
    ocel_path = f"{CONFIG['output_prefix']}.json"
    if os.path.exists(ocel_path):
        with open(ocel_path, "r", encoding="utf-8") as f:
            ocel_json = f.read()
        if "send_reminder_early" in ocel_json:
            ocel_json = ocel_json.replace("send_reminder_early", "send_reminder")
            with open(ocel_path, "w", encoding="utf-8") as f:
                f.write(ocel_json)
            print(
                f"Patched event names in {ocel_path}: send_reminder_early -> send_reminder"
            )
    else:
        print(f"Warning: expected OCEL file not found: {ocel_path}")

    label_path = f"{CONFIG['output_prefix']}_labels.json"
    with open(label_path, "w") as f:
        json.dump(
            {
                "scenario": "event_substitution",
                "description": (
                    "send_reminder fires before send_invoice for anomalous POs; "
                    "no delays or token values are manipulated"
                ),
                "counterfactual": "swap send_reminder and send_invoice in the trace",
                "labels": {str(k): v for k, v in labels.items()},
            },
            f,
            indent=2,
        )
    print(f"Labels saved to {label_path}")
