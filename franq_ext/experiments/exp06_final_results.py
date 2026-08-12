"""06 - Aggregate the ladder into the final ablation table + figures.

Reads results/raw/all_conditions.csv (written by 01-04), orders it by the ladder, writes
results/tables/ablation.csv, prints the table, and renders figures.
"""
from __future__ import annotations

import csv
import os

from franq_ext.experiments import common
from franq_ext.experiments.conditions import LADDER
from franq_ext.experiments import plots

_ORDER = [short for (short, _label, _b) in LADDER]
_COLS = ["short_name", "auroc", "prr", "ece",
         "factual_acc_before", "factual_acc_after",
         "correction_success", "correction_regret", "n_flagged", "n_corrected"]


def _load_rows() -> list[dict]:
    path = os.path.join(common.RAW_DIR, "all_conditions.csv")
    if not os.path.exists(path):
        raise SystemExit("No results found. Run 01-04 first (or python -m franq_ext.experiments.run_all).")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # De-dup (keep last) and order by ladder.
    by_name = {r["short_name"]: r for r in rows}
    return [by_name[s] for s in _ORDER if s in by_name]


def _fmt(v):
    try:
        return f"{float(v):.3f}"
    except (ValueError, TypeError):
        return str(v)


def main() -> None:
    common.ensure_dirs()
    rows = _load_rows()

    out = os.path.join(common.TABLES_DIR, "ablation.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Pretty print.
    print("\n================ ABLATION TABLE ================")
    header = f"{'condition':<12} {'AUROC':>7} {'PRR':>7} {'ECE':>7} " \
             f"{'acc_b':>7} {'acc_a':>7} {'cor_suc':>8} {'regret':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['short_name']:<12} {_fmt(r['auroc']):>7} {_fmt(r['prr']):>7} "
              f"{_fmt(r['ece']):>7} {_fmt(r['factual_acc_before']):>7} "
              f"{_fmt(r['factual_acc_after']):>7} {_fmt(r['correction_success']):>8} "
              f"{_fmt(r['correction_regret']):>7}")
    print(f"\nWrote {out}")

    figs = plots.plot_ablation(rows, common.FIGURES_DIR)
    for f in figs:
        print(f"     figure: {f}")


if __name__ == "__main__":
    main()
