"""Detection metrics from the FRANQ base paper: AUROC and PRR.

Both measure how well an error/uncertainty score ranks *wrong* facts above correct ones.
The router's `error_prob` is the score; the label is whether the fact was actually wrong.

  * AUROC - standard ranking quality (0.5 = chance, 1.0 = perfect).
  * PRR (Prediction Rejection Ratio) - how much of the way to an ORACLE rejector the
    score gets you: 1.0 = as good as rejecting true errors first, 0.0 = no better than
    random. This is FRANQ's headline UQ metric.
"""
from __future__ import annotations

import numpy as np


def auroc(scores, wrong) -> float:
    from sklearn.metrics import roc_auc_score

    wrong = np.asarray(wrong, dtype=int)
    if len(set(wrong.tolist())) < 2:
        return float("nan")  # undefined with a single class
    return float(roc_auc_score(wrong, np.asarray(scores, dtype=float)))


def _rejection_area(order: np.ndarray, risk: np.ndarray) -> float:
    """Mean retained risk as we reject samples in `order` (most-suspect first)."""
    n = len(risk)
    risk_ordered = risk[order]
    total = risk.sum()
    curve = []
    for k in range(n):  # reject k samples, retain n-k
        retained = n - k
        rejected_risk = risk_ordered[:k].sum()
        curve.append((total - rejected_risk) / retained)
    return float(np.trapz(curve, dx=1.0 / max(n - 1, 1)))


def prr(uncertainty, wrong) -> float:
    """Prediction Rejection Ratio in (-inf, 1]; higher is better."""
    u = np.asarray(uncertainty, dtype=float)
    risk = np.asarray(wrong, dtype=int)
    n = len(risk)
    if n < 2 or risk.sum() == 0 or risk.sum() == n:
        return float("nan")
    model = _rejection_area(np.argsort(-u), risk)      # reject most-uncertain first
    oracle = _rejection_area(np.argsort(-risk), risk)   # reject true errors first
    base = float(risk.mean())                           # random ordering ~ flat at base
    if base == oracle:
        return float("nan")
    return float((base - model) / (base - oracle))
