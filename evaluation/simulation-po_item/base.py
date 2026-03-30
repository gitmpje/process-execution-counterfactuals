"""
base.py – Shared foundation for all PO/Item OCPN simulation scenarios.

Each scenario file imports `build_model(config)` from here, which constructs
the model, places tokens, wires up all standard events, and returns the
fully-configured objects needed to run or post-process a simulation.

Scenario files override behaviour by passing a `config` dict and/or by
monkey-patching the returned event handles before calling `run_simulation()`.

Config keys (all optional, with defaults shown):
    n_pos          (int)   – number of POs to generate             [20]
    items_per_po   (int)   – items per PO in the normal case        [2]
    simtime        (int)   – simulation wall-clock time             [200]
    anomaly_prob   (float) – probability that a given PO is abnormal[0.3]
    output_prefix  (str)   – file-name stem for saved OCEL files    ["ocel_report"]
    seed           (int)   – random seed for reproducibility        [42]
    send_reminder_mandatory (bool) – reminder always fires before pay [True]
"""

import random
from simpn.simulator import SimProblem, SimToken
from ocpn_prototypes import OCPNVar, OCPNEvent
from ocpn_reporter import OCELReporter

# ---------------------------------------------------------------------------
# Item catalogue – one pair per PO slot, used to build realistic tokens
# ---------------------------------------------------------------------------
ITEM_CATALOGUE = [
    ("wheel_front", "wheel_back"),
    ("brakes_front", "brakes_back"),
    ("frame", "handlebar"),
    ("saddle", "seatpost"),
    ("chain", "cassette"),
    ("pedal_left", "pedal_right"),
    ("fork", "headset"),
    ("bottom_bracket", "crank"),
    ("derailleur_front", "derailleur_back"),
    ("cable_brake", "cable_gear"),
]


def _default_config():
    return dict(
        n_pos=20,
        items_per_po=2,
        simtime=200,
        anomaly_prob=0.3,
        output_prefix="ocel_report",
        seed=42,
        send_reminder_mandatory=True,
    )


# ---------------------------------------------------------------------------
# Token factories
# ---------------------------------------------------------------------------


def make_po_tokens(n_pos, items_per_po, *, seed=42):
    """Return a list of plain PO dicts (no anomaly applied)."""
    rng = random.Random(seed)
    _ = rng  # seed captured, not used here but kept for reproducibility parity
    return [
        {
            "object_type": "PO",
            "PO_id": po_id,
            "items": items_per_po,
            "reminder_count": 0,
        }
        for po_id in range(1, n_pos + 1)
    ]


def make_item_tokens(n_pos, items_per_po, *, seed=42):
    """Return a list of plain item dicts (no anomaly applied)."""
    rng = random.Random(seed)
    _ = rng
    items = []
    item_id = 1
    for po_id in range(1, n_pos + 1):
        pair_idx = (po_id - 1) % len(ITEM_CATALOGUE)
        pair = ITEM_CATALOGUE[pair_idx]
        for slot in range(items_per_po):
            items.append(
                {
                    "object_type": "item",
                    "PO_id": po_id,
                    "item_name": pair[slot % len(pair)],
                    "item_id": item_id,
                }
            )
            item_id += 1
    return items


# ---------------------------------------------------------------------------
# Guard / behaviour helpers (shared, reusable)
# ---------------------------------------------------------------------------


def process_items_guard(PO, item_queue):
    po_id = PO["PO_id"]
    return any(tok.value["PO_id"] == po_id for tok in item_queue)


def process_items_behavior(PO, item_queue):
    put_pipeline = []
    put_back = []
    delay = 5
    for item in item_queue:
        if item.value["PO_id"] == PO["PO_id"]:
            put_pipeline.append(SimToken(item.value, delay=delay))
        else:
            put_back.append(SimToken(item.value, delay=delay))
    return [SimToken(PO, delay=delay), put_pipeline, put_back]


def complete_order_guard(paid_PO, shipped_item):
    matching = [it for it in shipped_item if it.value["PO_id"] == paid_PO["PO_id"]]
    return len(matching) == paid_PO["items"]


def complete_order_behavior(paid_PO, shipped_item):
    complete_items = []
    put_back = []
    for item in shipped_item:
        if item.value["PO_id"] == paid_PO["PO_id"]:
            complete_items.append(SimToken(item.value, delay=2))
        else:
            put_back.append(SimToken(item.value))
    return [SimToken(paid_PO, delay=5), complete_items, put_back]


def send_reminder_behavior(billed_PO):
    billed_PO["reminder_count"] += 1
    return [SimToken(billed_PO, delay=0)]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
class SimulationModel:
    """
    Container holding all places, events, and the SimProblem instance so
    scenario files can access and patch individual components easily.
    """

    __slots__ = (
        "model",
        "cfg",
        # places
        "PO_arrival",
        "item_arrival",
        "pipeline_PO",
        "pipeline_item",
        "billed_PO",
        "inventory",
        "paid_PO",
        "shipped_item",
        "completed_PO",
        "completed_item",
        # events
        "place_order_event",
        "send_invoice_event",
        "send_reminder_event",
        "pay_order_event",
        "pick_item_event",
        "ship_item_event",
        "complete_order_event",
        # reporter (set during run)
        "reporter",
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = SimProblem()
        self.reporter = None


def build_model(config=None):
    """
    Construct and return a fully initialised :class:`SimulationModel`.

    Scenario files call this, then (optionally) modify guards/behaviours on
    the returned object before handing it to :func:`run_simulation`.
    """
    cfg = _default_config()
    if config:
        cfg.update(config)

    random.seed(cfg["seed"])

    sim = SimulationModel(cfg)
    m = sim.model

    # ------------------------------------------------------------------
    # Places
    # ------------------------------------------------------------------
    sim.PO_arrival = OCPNVar(m, object_type="PO", _id="PO_arrival")
    sim.item_arrival = OCPNVar(m, object_type="item", _id="item_arrival")

    sim.pipeline_PO = OCPNVar(m, object_type="PO", _id="pipeline_PO")
    sim.pipeline_item = OCPNVar(m, object_type="item", _id="pipeline_item")

    sim.billed_PO = OCPNVar(m, object_type="PO", _id="billed_PO")
    sim.inventory = OCPNVar(m, object_type="item", _id="inventory")

    sim.paid_PO = OCPNVar(m, object_type="PO", _id="paid_PO")
    sim.shipped_item = OCPNVar(m, object_type="item", _id="shipped_item")

    sim.completed_PO = OCPNVar(m, object_type="PO", _id="completed_PO")
    sim.completed_item = OCPNVar(m, object_type="item", _id="completed_item")

    # ------------------------------------------------------------------
    # Initial tokens  (scenarios override these lists before calling
    #                  build_model, or patch tokens after, or replace
    #                  make_po_tokens / make_item_tokens entirely)
    # ------------------------------------------------------------------
    po_tokens = make_po_tokens(cfg["n_pos"], cfg["items_per_po"], seed=cfg["seed"])
    item_tokens = make_item_tokens(cfg["n_pos"], cfg["items_per_po"], seed=cfg["seed"])

    for po in po_tokens:
        sim.PO_arrival.put(po)
    for it in item_tokens:
        sim.item_arrival.put(it)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    sim.place_order_event = OCPNEvent(
        model=m,
        incoming_vars=[(sim.PO_arrival, False), (sim.item_arrival, True)],
        outgoing_vars=[(sim.pipeline_PO, False), (sim.pipeline_item, True)],
        name="place_order",
        guard=process_items_guard,
        behavior=process_items_behavior,
    )

    sim.send_invoice_event = OCPNEvent(
        model=m,
        incoming_vars=[(sim.pipeline_PO, False)],
        outgoing_vars=[(sim.billed_PO, False)],
        name="send_invoice",
        guard=None,
        behavior=lambda pipeline_PO: [SimToken(pipeline_PO, delay=2)],
    )

    if cfg["send_reminder_mandatory"]:
        # Reminder is mandatory (fires exactly once, blocks pay_order until done)
        reminder_guard = lambda billed_PO: billed_PO["reminder_count"] == 0
        pay_guard = lambda billed_PO: billed_PO["reminder_count"] >= 1
    else:
        # Reminder is optional / skipped
        reminder_guard = lambda billed_PO: False  # never fires
        pay_guard = lambda billed_PO: True  # always allowed

    sim.send_reminder_event = OCPNEvent(
        model=m,
        incoming_vars=[(sim.billed_PO, False)],
        outgoing_vars=[(sim.billed_PO, False)],
        name="send_reminder",
        guard=reminder_guard,
        behavior=send_reminder_behavior,
    )

    sim.pay_order_event = OCPNEvent(
        model=m,
        incoming_vars=[(sim.billed_PO, False)],
        outgoing_vars=[(sim.paid_PO, False)],
        name="pay_order",
        guard=pay_guard,
        behavior=lambda billed_PO: [SimToken(billed_PO, delay=0)],
    )

    sim.pick_item_event = OCPNEvent(
        model=m,
        incoming_vars=[(sim.pipeline_item, False)],
        outgoing_vars=[(sim.inventory, False)],
        name="pick_item",
        guard=None,
        behavior=lambda inventory: [SimToken(inventory, delay=7)],
    )

    sim.ship_item_event = OCPNEvent(
        model=m,
        incoming_vars=[(sim.inventory, False)],
        outgoing_vars=[(sim.shipped_item, False)],
        name="ship_item",
        guard=None,
        behavior=lambda inventory: [SimToken(inventory, delay=7)],
    )

    sim.complete_order_event = OCPNEvent(
        model=m,
        incoming_vars=[(sim.paid_PO, False), (sim.shipped_item, True)],
        outgoing_vars=[(sim.completed_PO, False), (sim.completed_item, True)],
        name="complete_order",
        guard=complete_order_guard,
        behavior=complete_order_behavior,
    )

    return sim


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_simulation(sim, output_suffix=""):
    """
    Run the simulation contained in *sim* and persist OCEL output.

    Parameters
    ----------
    sim : SimulationModel
        Built (and optionally patched) model.
    output_suffix : str
        Appended to the output_prefix so each scenario writes distinct files.
    """
    prefix = sim.cfg["output_prefix"] + (f"_{output_suffix}" if output_suffix else "")
    reporter = OCELReporter(sim.model)
    sim.reporter = reporter

    print(f"\n[{prefix}] Running simulation (simtime={sim.cfg['simtime']})…")
    sim.model.simulate(sim.cfg["simtime"], [reporter])

    reporter.save_ocel20(f"{prefix}.json")
    print(f"[{prefix}] Saved {prefix}.json")

    return reporter
