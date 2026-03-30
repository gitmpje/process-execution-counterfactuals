"""
scenario_04_object_insertion.py
=================================
Scenario: Object node insertion
---------------------------------
Anomaly  : One item is removed from an anomalous PO, so that PO has only
           `items_per_po - 1` items in the system while `PO.items` still
           expects `items_per_po`.  The complete_order guard can never fire.
Label    : abnormal = True for POs with a missing item.
Counterfactual expected: insert a new item node for the missing item, linked
                         to the PO and to complete_order.
"""

import random
from base import build_model, run_simulation, make_po_tokens, make_item_tokens

CONFIG = dict(
    n_pos=100,
    items_per_po=2,
    simtime=400,
    anomaly_prob=0.3,
    output_prefix="scenario_04",
    seed=404,
    send_reminder_mandatory=True,
)


def inject_anomalies(po_tokens, item_tokens, cfg):
    """
    Drop one item per anomalous PO.
    """
    rng = random.Random(cfg["seed"])
    labels = {po["PO_id"]: False for po in po_tokens}

    # Build lookup
    po_to_item_indices = {}
    for idx, item in enumerate(item_tokens):
        po_to_item_indices.setdefault(item["PO_id"], []).append(idx)

    indices_to_remove = set()
    for po in po_tokens:
        if rng.random() >= cfg["anomaly_prob"]:
            continue
        po_id = po["PO_id"]
        own = po_to_item_indices.get(po_id, [])
        if not own:
            continue
        # Remove the last item of this PO
        indices_to_remove.add(own[-1])
        labels[po_id] = True

    filtered = [
        it for idx, it in enumerate(item_tokens) if idx not in indices_to_remove
    ]
    return filtered, labels


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
        f"(one item removed; complete_order cannot fire)"
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
                "scenario": "object_insertion",
                "description": "Item missing for PO; complete_order cannot fire",
                "counterfactual": "insert missing item node linked to PO and complete_order",
                "labels": {str(k): v for k, v in labels.items()},
            },
            f,
            indent=2,
        )
    print(f"Labels saved to {label_path}")
