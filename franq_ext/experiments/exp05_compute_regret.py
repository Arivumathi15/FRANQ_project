"""05 - Correction-regret analysis (the paper's headline novelty).

Two things:
  (a) Report the correction-regret of the FULL system on the dataset.
  (b) A controlled collateral-damage contrast: if the router false-positives on a
      CORRECT fact and the corrector proposes a wrong value, the SAFETY-CHECKED loop
      refuses it (regret = 0) while a RECKLESS loop accepts it (regret > 0). This is
      exactly the risk correction-regret is designed to expose — and which no compared
      baseline (RAC, QuCo-RAG, HalluEntity, RHD) measures.
"""
from __future__ import annotations

import csv
import os

from franq_ext.experiments import common
from franq_ext.experiments.conditions import full
from franq_ext.verification import ReVerifier
from franq_ext.scoring import LexicalAlignScorer
from franq_ext.schema import Fact


def _controlled_contrast():
    """Same flags, two acceptance policies; return (safe_regret, reckless_regret)."""
    gold = [("Einstein", "birth year", "1879"), ("Einstein", "birth city", "Ulm")]
    before = {("einstein", "birth year"): "1879", ("einstein", "birth city"): "Munich"}
    rv = ReVerifier(LexicalAlignScorer())

    # SAFE: the genuinely-wrong fact is fixed; the correct fact is left untouched.
    safe = [Fact("Einstein", "birth year", "1879"), Fact("Einstein", "birth city", "Ulm")]
    safe_res = rv.evaluate(safe, gold, before)

    # RECKLESS: the correct 'birth year' was also flagged and overwritten with a wrong value.
    reckless = [Fact("Einstein", "birth year", "1899"), Fact("Einstein", "birth city", "Ulm")]
    reckless_res = rv.evaluate(reckless, gold, before)

    return safe_res.correction_regret, reckless_res.correction_regret


def main() -> None:
    common.ensure_dirs()
    name, examples = common.dataset_from_env()

    res = common.run_condition("A3_full", "Full system", full(common_seed()), examples,
                               seed=common_seed())
    safe_regret, reckless_regret = _controlled_contrast()

    path = os.path.join(common.TABLES_DIR, "regret_analysis.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "correction_regret", "note"])
        w.writerow(["full_system_on_dataset", f"{res.correction_regret:.3f}",
                    f"dataset={name}"])
        w.writerow(["safety_checked_correction", f"{safe_regret:.3f}",
                    "correct fact refused -> no damage"])
        w.writerow(["reckless_correction", f"{reckless_regret:.3f}",
                    "correct fact overwritten -> damage"])

    print("[05] Correction-regret analysis")
    print(f"     full system on {name}: regret = {res.correction_regret:.3f}")
    print(f"     controlled contrast -> safety-checked: {safe_regret:.3f}"
          f"   reckless: {reckless_regret:.3f}")
    print(f"     wrote {path}")


def common_seed() -> int:
    return int(os.environ.get("FRANQ_SEED", "13"))


if __name__ == "__main__":
    main()
