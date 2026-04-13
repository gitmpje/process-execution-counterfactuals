"""
run_all_scenarios.py
=====================
Convenience script: run all eight validation scenarios in sequence.
Each scenario is launched as a subprocess so its `if __name__ == "__main__"`
block executes correctly and output files land in the right directory.

Usage
-----
    cd simulation/
    python run_all_scenarios.py [--scenarios 1 3 5] [--outdir ../data/raw]

Options
-------
    --scenarios   Space-separated list of scenario numbers to run (default: all)
    --config-file YAML file in which the scripts to run are defined (default: config.yaml)
    --outdir      Directory to write output files (default: current dir)
    --run-id      Identifier for the run used as suffix for the result files from search_counterfactual (default: None)
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
import yaml

from typing import List

SCENARIO_MODULES = {
    1: (
        "scenario_01_object_attr_change.py",
        "Object attribute change (PO.items inflated)",
    ),
    2: ("scenario_02_object_substitution.py", "Object substitution (wrong item)"),
    3: ("scenario_03_object_deletion.py", "Object deletion (surplus item)"),
    4: ("scenario_04_object_insertion.py", "Object insertion (missing item)"),
    5: ("scenario_05_event_attr_change.py", "Event attribute change (outlier delay)"),
    6: (
        "scenario_06_event_substitution.py",
        "Event substitution (reminder after pay)",
    ),
    7: ("scenario_07_event_deletion.py", "Event deletion (extra reminder)"),
    8: ("scenario_08_event_insertion.py", "Event insertion (reminder skipped)"),
}

SIM_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_scenario_output_prefix(script_path):
    spec = importlib.util.spec_from_file_location("scenario_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONFIG"):
        raise RuntimeError(f"Scenario script {script_path} has no CONFIG")

    output_prefix = module.CONFIG.get("output_prefix")
    if not output_prefix:
        raise RuntimeError(
            f"CONFIG['output_prefix'] not set in scenario script {script_path}"
        )
    return output_prefix


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCPN validation scenarios.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=int,
        default=list(SCENARIO_MODULES.keys()),
        help="Scenario numbers to run (default: all)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.yaml",
        help="YAML file in which the scripts to run are defined (default: config.yaml)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/",
        help="Output directory for OCEL and label files (default: data/)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifier for the run used as suffix for the result files from search_counterfactual (default: None)",
    )
    return parser.parse_args()


def run_scenario(number, outdir, script_path):
    t0 = time.time()

    os.makedirs(outdir, exist_ok=True)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.abspath(outdir),  # output files written here
        env={**os.environ, "PYTHONPATH": SIM_DIR},  # so `import base` works
    )

    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"process exited with code {result.returncode}")

    print(f"  ✓ Simulation scenario {number:02d} completed in {elapsed:.1f}s")

def run_scripts(number, script_path, scripts: List[str], run_id: str):
    # Use scenario CONFIG output_prefix for scripts.
    output_prefix = _read_scenario_output_prefix(script_path)
    print(f"  Using output_prefix='{output_prefix}'")

    # Run the scripts in order (build_dataset -> train_gnn -> search_counterfactual).
    env = {
        **os.environ,
        "PYTHONPATH": SIM_DIR,
        "SCENARIO_PREFIX": output_prefix,
        "RUN_ID": run_id,
    }
    for pipeline_script in scripts:
        t0_step = time.time()
        print(f"  ---> Running {pipeline_script} for scenario {number}...")
        result = subprocess.run(
            [sys.executable, os.path.join(SIM_DIR, pipeline_script)],
            cwd=SIM_DIR,
            env=env,
        )
        elapsed_step = time.time() - t0_step
        if result.returncode != 0:
            raise RuntimeError(
                f"{pipeline_script} failed for scenario {number} with code {result.returncode}"
            )
        print(f"  ✓ {pipeline_script} completed in {elapsed_step:.1f}s")


def main():
    args = parse_args()
    total_start = time.time()
    failed = []

    # Load configuration file
    config_file = os.path.join(args.config_file)
    with open(config_file) as f:
        cfg_run = yaml.safe_load(f)["run_scenarios"]

    # Clear results file
    with open("data/cf_results.txt", "w") as f:
        f.write("")

    for number in sorted(args.scenarios):
        if number not in SCENARIO_MODULES:
            print(f"[WARN] Unknown scenario {number}, skipping.")
            continue
        try:
            filename, description = SCENARIO_MODULES[number]
            print(f"\n{'=' * 60}")
            print(f"  Scenario {number:02d}: {description}")
            print(f"{'=' * 60}")

            script_path = os.path.join(SIM_DIR, filename)
            if cfg_run["simulate"]:
                run_scenario(number, args.outdir, script_path)
            run_scripts(number, script_path, cfg_run.get("scripts", []), args.run_id)
        except Exception as exc:
            print(f"[ERROR] Scenario {number} failed: {exc}")
            failed.append(number)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  All done in {total_elapsed:.1f}s.")
    if failed:
        print(f"  Failed scenarios: {failed}")
    else:
        print("  All scenarios completed successfully.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
