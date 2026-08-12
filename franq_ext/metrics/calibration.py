"""Expected Calibration Error (ECE).

Quantifies whether the router's confidence is trustworthy: among facts it called wrong
with ~70% probability, are ~70% actually wrong? Lower ECE = better calibrated. This is
the number that backs Pillar 2's 'calibrated' claim in the paper.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Last bin is closed on the right so p == 1.0 is included.
        in_bin = (probs > lo) & (probs <= hi) if hi < 1.0 else (probs > lo) & (probs <= hi + 1e-9)
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = probs[in_bin].mean()
        accuracy = labels[in_bin].mean()  # observed frequency of the positive class
        ece += (count / n) * abs(confidence - accuracy)
    return float(ece)
