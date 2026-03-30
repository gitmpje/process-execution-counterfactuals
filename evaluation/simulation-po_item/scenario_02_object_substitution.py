"""
scenario_02_object_substitution.py
====================================
Scenario: Object node substitution
------------------------------------
Anomaly  : For anomalous POs, one of their own items is replaced by a real item
           that genuinely belongs to a *different* PO — i.e. the wrong physical
           object is picked, shipped, and completed under the anomalous PO.
           The substituted item retains its original `PO_id` (pointing to the
           donor PO) throughout; it is NOT re-labelled.  This means the OCEL
           trace for the anomalous PO contains an item node whose `PO_id`
           differs from the PO it was processed under — exactly the structural
           signal the GNN and counterfactual search should detect.

           The anomalous PO still has the correct item *count* (one own item +
           one substituted item), so `complete_order` fires.  The donor PO
           loses one item and therefore its `complete_order` cannot fire unless
           a separate replacement arrives (not modelled here — it simply stalls).

Mechanism : We patch both `place_order` and `complete_order` guards/behaviors
           on the SimulationModel so that, for anomalous POs, an item from the
           donor PO is accepted in place of one of the PO's own items.  The
           donor item is identified by a pre-computed mapping stored in a
           closure, keeping the scenario file self-contained.

Label    : abnormal = True for POs that receive a substituted (donor) item.
Counterfactual expected: replace the donor item node with the correct item
           (one whose PO_id matches the anomalous PO).
"""

import random
from simpn.simulator import SimToken
from base import build_model, run_simulation, make_po_tokens, make_item_tokens

CONFIG = dict(
    n_pos=100,
    items_per_po=2,
    simtime=400,
    anomaly_prob=0.3,
    output_prefix="scenario_02",
    seed=202,
    send_reminder_mandatory=True,
)


def build_substitution_map(po_tokens, item_tokens, cfg):
    """
    Decide which POs are anomalous and, for each, which donor item (belonging
    to a different PO) will be substituted in.

    Returns
    -------
    labels          : dict[po_id -> bool]
    substitution    : dict[anomalous_po_id -> donor_item_id]
        Maps each anomalous PO to the item_id of the donor item it will receive.
    donor_item_lookup : dict[item_id -> item dict]
        Quick access to full donor item dicts by item_id.
    """
    rng = random.Random(cfg["seed"])
    n = len(po_tokens)
    labels = {po["PO_id"]: False for po in po_tokens}

    # po_id -> list of item dicts that belong to it
    po_to_items = {}
    for item in item_tokens:
        po_to_items.setdefault(item["PO_id"], []).append(item)

    donor_item_lookup = {item["item_id"]: item for item in item_tokens}
    substitution = {}  # anomalous_po_id -> donor_item_id

    for i, po in enumerate(po_tokens):
        if rng.random() >= cfg["anomaly_prob"]:
            continue
        po_id = po["PO_id"]
        # Donor: the next PO in the list (wrap around)
        donor_po_id = po_tokens[(i + 1) % n]["PO_id"]
        donor_items = po_to_items.get(donor_po_id, [])
        if not donor_items:
            continue
        # Use the first item of the donor PO as the substitute
        substitution[po_id] = donor_items[0]["item_id"]
        labels[po_id] = True

    return labels, substitution, donor_item_lookup


def patched_place_order_guard(anomalous_po_ids, substitution, donor_item_lookup):
    """
    For anomalous POs: fire when the item_queue contains the designated donor
    item (regardless of its PO_id) plus at least one own item.
    For normal POs: standard guard (at least one matching item).
    """

    def guard(PO, item_queue):
        po_id = PO["PO_id"]
        if po_id not in anomalous_po_ids:
            # Normal guard
            return any(tok.value["PO_id"] == po_id for tok in item_queue)
        # Anomalous guard: need the specific donor item AND one own item
        donor_id = substitution[po_id]
        has_donor = any(tok.value["item_id"] == donor_id for tok in item_queue)
        has_own = any(tok.value["PO_id"] == po_id for tok in item_queue)
        return has_donor and has_own

    return guard


def patched_place_order_behavior(anomalous_po_ids, substitution):
    """
    For anomalous POs: move the donor item + one own item into pipeline_item,
    leaving remaining own items back in item_arrival.
    For normal POs: standard behavior.
    """

    def behavior(PO, item_queue):
        po_id = PO["PO_id"]
        delay = 5
        put_pipeline = []
        put_back = []

        if po_id not in anomalous_po_ids:
            # Normal behavior
            for item in item_queue:
                if item.value["PO_id"] == po_id:
                    put_pipeline.append(SimToken(item.value, delay=delay))
                else:
                    put_back.append(SimToken(item.value, delay=delay))
            return [SimToken(PO, delay=delay), put_pipeline, put_back]

        # Anomalous behavior: pick one own item + the donor item
        donor_id = substitution[po_id]
        own_taken = False
        donor_taken = False
        for item in item_queue:
            iid = item.value["item_id"]
            own = item.value["PO_id"] == po_id
            is_donor = iid == donor_id

            if is_donor and not donor_taken:
                put_pipeline.append(SimToken(item.value, delay=delay))
                donor_taken = True
            elif own and not own_taken:
                put_pipeline.append(SimToken(item.value, delay=delay))
                own_taken = True
            else:
                put_back.append(SimToken(item.value, delay=delay))

        return [SimToken(PO, delay=delay), put_pipeline, put_back]

    return behavior


def patched_complete_order_guard(anomalous_po_ids, substitution):
    """
    For anomalous POs: count both own items and the donor item toward the
    required item count.
    """

    def guard(paid_PO, shipped_item):
        po_id = paid_PO["PO_id"]
        if po_id not in anomalous_po_ids:
            matching = [it for it in shipped_item if it.value["PO_id"] == po_id]
            return len(matching) == paid_PO["items"]
        donor_id = substitution[po_id]
        relevant = [
            it
            for it in shipped_item
            if it.value["PO_id"] == po_id or it.value["item_id"] == donor_id
        ]
        return len(relevant) == paid_PO["items"]

    return guard


def patched_complete_order_behavior(anomalous_po_ids, substitution):
    """
    For anomalous POs: consume both own items and the donor item into
    completed_item so the donor item appears in the completed trace.
    """

    def behavior(paid_PO, shipped_item):
        po_id = paid_PO["PO_id"]
        complete_items = []
        put_back = []

        if po_id not in anomalous_po_ids:
            for item in shipped_item:
                if item.value["PO_id"] == po_id:
                    complete_items.append(SimToken(item.value, delay=2))
                else:
                    put_back.append(SimToken(item.value))
            return [SimToken(paid_PO, delay=5), complete_items, put_back]

        donor_id = substitution[po_id]
        for item in shipped_item:
            own = item.value["PO_id"] == po_id
            is_donor = item.value["item_id"] == donor_id
            if own or is_donor:
                complete_items.append(SimToken(item.value, delay=2))
            else:
                put_back.append(SimToken(item.value))
        return [SimToken(paid_PO, delay=5), complete_items, put_back]

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

    labels, substitution, donor_item_lookup = build_substitution_map(
        po_tokens, item_tokens, CONFIG
    )
    anomalous_po_ids = set(po_id for po_id, v in labels.items() if v)

    anomaly_count = len(anomalous_po_ids)
    print(
        f"Injected {anomaly_count}/{CONFIG['n_pos']} anomalous POs "
        f"(one own item replaced by a real item from a donor PO)"
    )
    for po_id in sorted(anomalous_po_ids):
        did = substitution[po_id]
        ditem = donor_item_lookup[did]
        print(
            f"  PO {po_id} ← donor item {did} "
            f"('{ditem['item_name']}', originally from PO {ditem['PO_id']})"
        )

    for po in po_tokens:
        sim.PO_arrival.put(po)
    for it in item_tokens:
        sim.item_arrival.put(it)

    # Patch all four touched guards/behaviors in one place
    sim.place_order_event.guard = patched_place_order_guard(
        anomalous_po_ids, substitution, donor_item_lookup
    )
    sim.place_order_event.behavior = patched_place_order_behavior(
        anomalous_po_ids, substitution
    )
    sim.complete_order_event.guard = patched_complete_order_guard(
        anomalous_po_ids, substitution
    )
    sim.complete_order_event.behavior = patched_complete_order_behavior(
        anomalous_po_ids, substitution
    )

    run_simulation(sim)

    import json

    label_path = f"{CONFIG['output_prefix']}-labels.json"
    with open(label_path, "w") as f:
        json.dump(
            {
                "scenario": "object_substitution",
                "description": (
                    "A real item from a donor PO is processed under the anomalous PO. "
                    "The donor item retains its original PO_id in the OCEL trace, making "
                    "the substitution structurally visible. complete_order fires (count ok) "
                    "but the item graph contains a cross-PO edge."
                ),
                "counterfactual": (
                    "Replace the donor item node with the correct item "
                    "(PO_id matching the anomalous PO)"
                ),
                "substitution_map": {str(k): v for k, v in substitution.items()},
                "labels": {str(k): v for k, v in labels.items()},
            },
            f,
            indent=2,
        )
    print(f"Labels saved to {label_path}")
