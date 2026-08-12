"""Helper shared by scripts 01-04: run a single ladder rung and record it."""
from __future__ import annotations

import os

from franq_ext.experiments import common


def run_rung(index: int, short_name: str, label: str, cfg_builder) -> None:
    common.ensure_dirs()
    name, examples = common.dataset_from_env()
    cfg = cfg_builder(cfg_seed())
    # Apply env overrides BEFORE saving, so the saved config records the REAL backends /
    # models / n_samples actually used (not the offline defaults) — needed for the paper.
    common.apply_backend_env(cfg)
    cfg.to_json(os.path.join(common.RAW_DIR, f"{index:02d}_{short_name}.config.json"))
    res = common.run_condition(short_name, label, cfg, examples, seed=cfg_seed())
    common.save_row(short_name, res, os.path.join(common.RAW_DIR, f"{index:02d}_{short_name}.csv"))
    common.append_master(short_name, res)
    print(f"[{index:02d}] {label}  (dataset={name}, n_facts={res.n_facts})")
    print(f"     AUROC={res.auroc:.3f}  PRR={res.prr:.3f}  ECE={res.ece:.3f}")
    print(f"     factual_acc: {res.factual_acc_before:.3f} -> {res.factual_acc_after:.3f}"
          f"   correction_success={res.correction_success:.3f}"
          f"   correction_regret={res.correction_regret:.3f}")
    print(f"     flagged={res.n_flagged}  corrected={res.n_corrected}")


def cfg_seed() -> int:
    return int(os.environ.get("FRANQ_SEED", "13"))
