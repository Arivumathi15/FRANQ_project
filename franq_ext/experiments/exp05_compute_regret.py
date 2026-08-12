"""05 - Correction-regret analysis (the paper's headline novelty).

The safety claim is measured, not asserted: we run the FULL system twice on the SAME data,
once with the safety-checked acceptance rule and once with a reckless "replace on any change"
rule, and compare their MEASURED correction-regret and factual accuracy. The safety-checked
loop should match (or beat) reckless on regret while not losing accuracy — the collateral
damage that no compared baseline (RAC, QuCo-RAG, HalluEntity, RHD) measures.

A tiny hand-built unit check (`_illustrative_unit`) is kept only as a sanity anchor for the
regret formula; the reported result is the real dataset measurement above.
"""
from __future__ import annotations

import csv
import os

from franq_ext.experiments import common
from franq_ext.experiments.conditions import full
from franq_ext.verification import ReVerifier
from franq_ext.scoring import LexicalAlignScorer
from franq_ext.schema import Fact


def _illustrative_unit() -> tuple[float, float]:
    """Formula sanity anchor (NOT a dataset result): a fixed correct fact left untouched
    gives regret 0; the same correct fact overwritten with a wrong value gives regret 1."""
    gold = [("Einstein", "birth year", "1879"), ("Einstein", "birth city", "Ulm")]
    before = {("einstein", "birth year"): "1879", ("einstein", "birth city"): "Munich"}
    rv = ReVerifier(LexicalAlignScorer())
    safe = [Fact("Einstein", "birth year", "1879"), Fact("Einstein", "birth city", "Ulm")]
    reckless = [Fact("Einstein", "birth year", "1899"), Fact("Einstein", "birth city", "Ulm")]
    return (rv.evaluate(safe, gold, before).correction_regret,
            rv.evaluate(reckless, gold, before).correction_regret)


def main() -> None:
    common.ensure_dirs()
    name, examples = common.dataset_from_env()
    seed = common_seed()

    # REAL measurement: same data, two acceptance policies.
    safe_cfg = full(seed)                       # correction.safety_checked = True
    res_safe = common.run_condition(
        "A3_full", "Full system (safety-checked)", safe_cfg, examples, seed=seed
    )

    reckless_cfg = full(seed)
    reckless_cfg.correction.safety_checked = False
    res_reckless = common.run_condition(
        "A3_reckless", "Full system (reckless)", reckless_cfg, examples, seed=seed
    )

    unit_safe, unit_reckless = _illustrative_unit()

    path = os.path.join(common.TABLES_DIR, "regret_analysis.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "correction_regret", "correction_success",
                    "factual_acc_after", "n_corrected", "note"])
        w.writerow(["safety_checked", f"{res_safe.correction_regret:.3f}",
                    f"{res_safe.correction_success:.3f}", f"{res_safe.factual_acc_after:.3f}",
                    res_safe.n_corrected, f"measured on {name}"])
        w.writerow(["reckless", f"{res_reckless.correction_regret:.3f}",
                    f"{res_reckless.correction_success:.3f}",
                    f"{res_reckless.factual_acc_after:.3f}",
                    res_reckless.n_corrected, f"measured on {name}"])
        w.writerow(["unit_check_safe", f"{unit_safe:.3f}", "", "", "",
                    "formula anchor: correct fact untouched"])
        w.writerow(["unit_check_reckless", f"{unit_reckless:.3f}", "", "", "",
                    "formula anchor: correct fact overwritten"])

    print("[05] Correction-regret analysis (measured on real data)")
    print(f"     safety-checked: regret={res_safe.correction_regret:.3f}  "
          f"success={res_safe.correction_success:.3f}  "
          f"acc_after={res_safe.factual_acc_after:.3f}  corrected={res_safe.n_corrected}")
    print(f"     reckless:       regret={res_reckless.correction_regret:.3f}  "
          f"success={res_reckless.correction_success:.3f}  "
          f"acc_after={res_reckless.factual_acc_after:.3f}  corrected={res_reckless.n_corrected}")
    print(f"     formula anchor -> safe: {unit_safe:.3f}   reckless: {unit_reckless:.3f}")
    print(f"     wrote {path}")


def common_seed() -> int:
    return int(os.environ.get("FRANQ_SEED", "13"))


if __name__ == "__main__":
    main()
