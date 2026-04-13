# OCPN Counterfactual Validation – Simulation Suite

## Overview

This suite generates labelled OCEL datasets for validating an object-centric
counterfactual search algorithm. Each of the eight scenarios injects a
specific type of anomaly into a Purchase Order / Item shipment process,
produces an OCEL log, and writes a companion `_labels.json` file that maps
every PO to a boolean `is_anomalous` flag.

```text
simulation/
├── base.py  # Shared model, helpers, runner
├── scenario_01_object_attr_change.py
├── scenario_02_object_substitution.py
├── scenario_03_object_deletion.py
├── scenario_04_object_insertion.py
├── scenario_05_event_attr_change.py
├── scenario_06_event_substitution.py
├── scenario_07_event_deletion.py
├── scenario_08_event_insertion.py
└── run_all_scenarios.py
```

## Running

```bash
# Run all 8 scenarios (output to current directory)
python run_all_scenarios.py

# Run specific scenarios only
python run_all_scenarios.py --scenarios 1 3 7

# Other options:
--outdir ../data/raw # write output to a dedicated folder
--skip-simulation # skip running the simulation and use existing data set
--config-file config.yaml # YAML file in which the scripts to run are defined

# Run a single scenario directly
python scenario_01_object_attr_change.py
```

Each scenario writes two files:

| File | Description |
|---|---|
| `scenario_NN.json`     | OCEL 2.0 standard |
| `scenario_NN-labels.json` | Ground-truth anomaly labels per PO |

## Scenario Catalogue

Scenarios for validation and benchmarking of search algorithm for
Object-Centric counterfactuals.

Base process: Purchase Order with Item shipment [van der Aalst and Berti (2020)](https://journals.sagepub.com/doi/pdf/10.3233/FI-2020-1946?casa_token=P6Pqcrv2AzgAAAAA:A1Hm5nOtbRluAN-rdCY-SMXlzLgl3C6g3SfsQCVEaXKKqnqzBzPHyW_uYd3Fnf88QbE2TFT2Eazj6K8).

| # | Category | Anomaly injected | `complete_order` fires? | Counterfactual |
|---|---|---|---|---|
| 01 | Object attr change | `PO.items` inflated by 1 | ✗ | Decrease `PO.items` |
| 02 | Object substitution | One item's `PO_id` re-pointed to wrong PO | ✓ (but with wrong item) | Replace with correct `item` |
| 03 | Object deletion | Extra surplus item injected | ✓ (PO completes; item stranded) | Delete surplus item node |
| 04 | Object insertion | One item removed from PO | ✗ | Insert missing item node |
| 05 | Event attr change | `pay_order` delay ×10 | ✓ (delayed) | Decrease `pay_order.timestamp` |
| 06 | Event substitution | `send_reminder` fires before `send_invoice` | ✓ | Swap events in trace |
| 07 | Event deletion | Extra `send_reminder` fires; `reminder_count` ≥ 2 | ✓ | Delete `send_reminder` node |
| 08 | Event insertion | `send_reminder` skipped; `reminder_count` stays 0 | ✓ | Insert `send_reminder` node |

## Architecture: `base.py`

`base.py` exposes two public functions:

```python
sim = build_model(config: dict) -> SimulationModel
reporter = run_simulation(sim, output_suffix="") -> OCELReporter
```

`SimulationModel` holds references to every place and event, making it easy
for scenario files to patch individual components:

```python
# Example: replace the complete_order behavior before running
sim.complete_order_event.behavior = my_custom_behavior
run_simulation(sim)
```

### Config keys

| Key | Default | Description |
|---|---|---|
| `n_pos` | 100 | Number of POs generated |
| `items_per_po` | 2 | Items per PO (normal case) |
| `simtime` | 200 | Simulation time horizon |
| `anomaly_prob` | 0.3 | Fraction of anomalous POs |
| `output_prefix` | `"ocel_report"` | Stem of output file names |
| `seed` | 42 | Random seed |
| `send_reminder_mandatory` | `True` | Whether reminder must fire before pay |

## GNN Training Notes

Each `-labels.json` file has the structure:

```json
{
  "scenario": "object_attr_change",
  "description": "...",
  "counterfactual": "...",
  "labels": {
    "1": false,
    "2": true,
    ...
  }
}
```

The OCEL log can be loaded with `pm4py` to extract object-centric event graphs
for GNN training. Labels align on `PO_id`. With `n_pos=100` and
`anomaly_prob=0.3` each scenario produces roughly 30 positive and 70 negative
examples; adjust `n_pos` upward for larger datasets.
