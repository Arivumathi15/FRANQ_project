"""Evaluation harness: run one Config over a set of examples and aggregate all metrics.

This is what every experiment script calls. One ablation condition == one Config, so the
whole ablation table is just `evaluate_config` run under different Configs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from franq_ext.config import Config
from franq_ext.pipeline import Pipeline
from franq_ext.schema import Example
from franq_ext.metrics import (
    auroc,
    prr,
    expected_calibration_error,
    correction_regret,
)


@dataclass
class EvalResult:
    condition: str
    n_examples: int
    n_facts: int
    # Detection (router quality) — FRANQ metrics.
    auroc: float
    prr: float
    ece: float
    # Factuality / correction.
    factual_acc_before: float
    factual_acc_after: float
    correction_success: float   # of wrong facts, fraction correct after
    correction_regret: float    # the safety metric
    n_flagged: int
    n_corrected: int

    def as_row(self) -> dict:
        return asdict(self)


def evaluate_config(cfg: Config, examples: list[Example], condition: str = "",
                    router=None, fact_source_factory=None) -> EvalResult:
    pipeline = Pipeline(cfg, router=router, fact_source_factory=fact_source_factory)
    return evaluate_with_pipeline(pipeline, examples, condition)


def evaluate_with_pipeline(pipeline, examples: list[Example], condition: str = "") -> EvalResult:
    """Evaluate using an already-built pipeline (so its models load only once)."""
    error_probs, wrong_before = [], []
    correct_before, correct_after = [], []
    regret_pairs = []
    n_flagged = n_corrected = 0

    import os
    progress = int(os.environ.get("FRANQ_PROGRESS_EVERY", "25"))
    total = len(examples)
    for i, ex in enumerate(examples):
        if progress and (i % progress == 0):
            print(f"    [{condition}] example {i}/{total}...", flush=True)
        out = pipeline.run(ex)
        by_key = {f.key(): f for f in out.facts}
        for pf in out.reverify.per_fact:
            fact = by_key[pf["fact"]]
            error_probs.append(fact.error_prob if fact.error_prob is not None else 0.0)
            wrong_before.append(0 if pf["correct_before"] else 1)
            correct_before.append(1 if pf["correct_before"] else 0)
            correct_after.append(1 if pf["correct_after"] else 0)
            regret_pairs.append((pf["correct_before"], pf["correct_after"]))
        n_flagged += sum(1 for f in out.facts if f.flagged)
        n_corrected += sum(1 for f in out.facts if f.corrected)

    wb = np.asarray(wrong_before)
    cb = np.asarray(correct_before)
    ca = np.asarray(correct_after)

    # Correction success = among facts wrong before, fraction correct after.
    n_wrong = int(wb.sum())
    success = float(ca[wb == 1].mean()) if n_wrong > 0 else float("nan")

    return EvalResult(
        condition=condition,
        n_examples=len(examples),
        n_facts=len(wrong_before),
        auroc=auroc(error_probs, wb),
        prr=prr(error_probs, wb),
        ece=expected_calibration_error(error_probs, wb),
        factual_acc_before=float(cb.mean()) if len(cb) else float("nan"),
        factual_acc_after=float(ca.mean()) if len(ca) else float("nan"),
        correction_success=success,
        correction_regret=correction_regret(regret_pairs),
        n_flagged=n_flagged,
        n_corrected=n_corrected,
    )
