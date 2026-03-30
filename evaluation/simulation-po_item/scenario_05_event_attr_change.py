"""
scenario_05_event_attr_change.py
==================================
Scenario: Event node attribute change
---------------------------------------
Anomaly  : For anomalous POs, there is an outlier time gap between send_reminder
           and pay_order.  This manifests in the OCEL as an unusually large
           timestamp on pay_order relative to send_reminder.
Label    : abnormal = True for POs whose pay_order fires much later than
           expected after send_reminder.
Counterfactual expected: decrease pay_order.timestamp to a normal value.
"""

import random
from simpn.simulator import SimToken
from base import (
    build_model,
    run_simulation,
    make_po_tokens,
    make_item_tokens,
)

CONFIG = dict(
    n_pos=100,
    items_per_po=2,
    simtime=600,  # longer simtime to let outlier POs complete
    anomaly_prob=0.3,
    output_prefix="scenario_05",
    seed=505,
    send_reminder_mandatory=True,
)

NORMAL_REMINDER_DELAY = 2  # base process: send_reminder delay
OUTLIER_REMINDER_DELAY = (
    50  # anomalous: billed_PO held up, so pay_order fires ~50 units later
)


def make_anomalous_ids(po_tokens, cfg):
    rng = random.Random(cfg["seed"])
    return {po["PO_id"] for po in po_tokens if rng.random() < cfg["anomaly_prob"]}


def patched_send_reminder_behavior(anomalous_ids):
    """
    Emit the billed_PO token with an outlier delay for anomalous POs.
    This delays the availability of billed_PO after send_reminder, which in
    turn delays when pay_order can fire — shifting pay_order's timestamp later.
    """

    def behavior(billed_PO):
        billed_PO["reminder_count"] += 1
        delay = (
            OUTLIER_REMINDER_DELAY
            if billed_PO["PO_id"] in anomalous_ids
            else NORMAL_REMINDER_DELAY
        )
        return [SimToken(billed_PO, delay=delay)]

    return behavior


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
        f"(send_reminder→pay_order gap inflated by {OUTLIER_REMINDER_DELAY} time units)"
    )

    for po in po_tokens:
        sim.PO_arrival.put(po)
    for it in item_tokens:
        sim.item_arrival.put(it)

    # Patch send_reminder behavior: outlier delay on billed_PO token for anomalous POs
    sim.send_reminder_event.behavior = patched_send_reminder_behavior(anomalous_ids)

    run_simulation(sim)

    import json

    label_path = f"{CONFIG['output_prefix']}-labels.json"
    with open(label_path, "w") as f:
        json.dump(
            {
                "scenario": "event_attr_change",
                "description": (
                    f"send_reminder emits billed_PO with delay={OUTLIER_REMINDER_DELAY} for anomalous POs "
                    f"(normal delay={NORMAL_REMINDER_DELAY}), causing pay_order to fire "
                    f"~{OUTLIER_REMINDER_DELAY} time units later than expected"
                ),
                "counterfactual": "decrease pay_order.timestamp to normal range",
                "labels": {str(k): v for k, v in labels.items()},
            },
            f,
            indent=2,
        )
    print(f"Labels saved to {label_path}")
