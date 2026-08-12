"""Smoke test for the experiment ladder: run_all must produce a 4-row ablation table."""
import csv
import os
import subprocess
import sys


def test_run_all_produces_ablation_table(tmp_path):
    env = dict(os.environ)
    env["FRANQ_RESULTS"] = str(tmp_path / "results")
    env["FRANQ_DATASET"] = "sample"

    proc = subprocess.run(
        [sys.executable, "-m", "franq_ext.experiments.run_all"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    ablation = tmp_path / "results" / "tables" / "ablation.csv"
    assert ablation.exists(), "ablation.csv was not written"
    with open(ablation, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # The four ladder rungs, in order.
    assert [r["short_name"] for r in rows] == ["B0_franq", "A1_router", "A2_graph", "A3_full"]
    # The full system fixes errors (accuracy up) with zero regret on the sample.
    full = rows[-1]
    assert float(full["factual_acc_after"]) >= float(full["factual_acc_before"])
    assert float(full["correction_regret"]) == 0.0

    # The regret analysis table (safe vs reckless) was written too.
    assert (tmp_path / "results" / "tables" / "regret_analysis.csv").exists()
