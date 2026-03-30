"""
scenario_03_object_deletion.py
================================
Scenario: Object node deletion
--------------------------------
Anomaly  : An extra item is injected for an anomalous PO (so it has
           `items_per_po + 1` items in the system but `PO.items == items_per_po`).
           The extra item flows through pick_item → ship_item but is never
           consumed by complete_order (guard checks exact count), so it
           remains in shipped_item forever.
Label    : abnormal = True for POs with a surplus item (detectable as an item
           linked to a PO that is never consumed by complete_order).
Counterfactual expected: delete the surplus item node.
"""

import random
from base import (
    build_model,
    run_simulation,
    make_po_tokens,
    make_item_tokens,
    ITEM_CATALOGUE,
)

CONFIG = dict(
    n_pos=100,
    items_per_po=2,
    simtime=400,
    anomaly_prob=0.3,
    output_prefix="scenario_03",
    seed=303,
    send_reminder_mandatory=True,
)


def inject_anomalies(po_tokens, item_tokens, cfg):
    rng = random.Random(cfg["seed"])
    labels = {po["PO_id"]: False for po in po_tokens}

    next_item_id = max(it["item_id"] for it in item_tokens) + 1

    extra_items = []
    for po in po_tokens:
        if rng.random() >= cfg["anomaly_prob"]:
            continue
        po_id = po["PO_id"]
        pair_idx = (po_id - 1) % len(ITEM_CATALOGUE)
        # Add a third item from the same catalogue pair
        extra_name = ITEM_CATALOGUE[pair_idx][0] + "_extra"
        extra_items.append(
            {
                "object_type": "item",
                "PO_id": po_id,
                "item_name": extra_name,
                "item_id": next_item_id,
                "is_anomalous": True,
            }
        )
        next_item_id += 1
        labels[po_id] = True

    return item_tokens + extra_items, labels


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

    item_tokens, labels = inject_anomalies(po_tokens, item_tokens, CONFIG)

    anomaly_count = sum(labels.values())
    print(
        f"Injected {anomaly_count}/{CONFIG['n_pos']} anomalous POs "
        f"(surplus item added, never consumed by complete_order)"
    )

    for po in po_tokens:
        sim.PO_arrival.put(po)
    for it in item_tokens:
        sim.item_arrival.put(it)

    run_simulation(sim)

    import json

    label_path = f"{CONFIG['output_prefix']}-labels.json"
    with open(label_path, "w") as f:
        json.dump(
            {
                "scenario": "object_deletion",
                "description": "Extra item present; never consumed by complete_order",
                "counterfactual": "delete surplus item node",
                "labels": {str(k): v for k, v in labels.items()},
            },
            f,
            indent=2,
        )
    print(f"Labels saved to {label_path}")
