"""Pillar 2 - Learned, Calibrated Uncertainty Router.

Replaces FRANQ's fixed IF-ELSE with a trained classifier that outputs a *calibrated*
probability that a fact is wrong. Calibration (isotonic/sigmoid via CalibratedClassifierCV)
is what makes the confidence trustworthy — a claim the paper makes explicitly.

The router does two things per fact:
  * error_prob : calibrated P(fact is wrong)  -> used to flag facts for correction.
  * route      : an interpretable tag for which UQ signal dominated the decision,
                 mirroring FRANQ's semantic-entropy vs token-probability choice.
"""
from __future__ import annotations

import numpy as np

from franq_ext.schema import Fact
from franq_ext.router.features import fact_features


class CalibratedRouter:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None
        self._prior = 0.5  # fallback error rate before training

    @staticmethod
    def _balanced_weights(y: np.ndarray) -> np.ndarray:
        """Class-balanced sample weights so a skewed label distribution (common on hard QA,
        where most generated facts are wrong) can't collapse the base learner to a constant."""
        classes = np.array([0, 1])
        freq = np.array([(y == c).mean() for c in classes])
        weight_for = {c: (0.5 / f if f > 0 else 0.0) for c, f in zip(classes, freq)}
        return np.array([weight_for[c] for c in y])

    def fit(self, X, y) -> "CalibratedRouter":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._prior = float(y.mean()) if len(y) else 0.5
        if len(set(y.tolist())) < 2:
            # Degenerate: only one class seen. Keep the prior; skip fitting.
            self.model = None
            return self

        n_min = int(np.bincount(y).min())
        base = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, random_state=self.cfg.seed
        )
        # KEY calibration fix: class-balancing (via sample_weight) must touch only the BASE
        # learner, never the calibration step. Reweighting the calibrator distorts the very
        # probabilities it is meant to make faithful to the true base rate — that was inflating
        # ECE above the fixed-rule baseline. So we fit the base on balanced data, then calibrate
        # on an UNWEIGHTED held-out split (cv="prefit"), which sees the real class frequency.
        if n_min >= 4 and len(y) >= 20:
            X_tr, X_cal, y_tr, y_cal = train_test_split(
                X, y, test_size=0.3, random_state=self.cfg.seed, stratify=y
            )
            base.fit(X_tr, y_tr, sample_weight=self._balanced_weights(y_tr))
            cal = CalibratedClassifierCV(base, method=self.cfg.router.calibration, cv="prefit")
            cal.fit(X_cal, y_cal)  # NO sample_weight -> honest calibration
            self.model = cal
        else:
            # Too small for a held-out calibration split: fall back to cross-val calibration,
            # still unweighted so the probabilities stay calibrated to the true frequency.
            cv = max(2, min(3, n_min))
            self.model = CalibratedClassifierCV(
                base, method=self.cfg.router.calibration, cv=cv
            )
            self.model.fit(X, y)
        return self

    def predict_error_prob(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.model is None:
            return np.full(len(X), self._prior, dtype=float)
        return self.model.predict_proba(X)[:, 1]

    # ---------------------------------------------------------------- per-fact
    @staticmethod
    def _route_tag(fact: Fact) -> str:
        se = fact.semantic_entropy or 0.0
        tok_unc = 1.0 - (fact.parametric_uq or 0.0)
        return "semantic_entropy" if se >= tok_unc else "token_prob"

    def route_facts(self, facts: list[Fact], graph=None, flag_threshold: float = 0.5) -> list[Fact]:
        if not facts:
            return facts
        use_graph = self.cfg.router.use_graph_features
        X = [fact_features(f, graph, use_graph) for f in facts]
        probs = self.predict_error_prob(X)
        for f, p in zip(facts, probs):
            f.error_prob = float(p)
            f.route = self._route_tag(f)
            f.flagged = bool(p >= flag_threshold)
        return facts
