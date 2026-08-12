"""Train the Pillar-2 calibrated router from labeled examples.

Runs the signal stages over a set of examples, labels each fact against gold (wrong/right),
and fits a CalibratedRouter. `augment` adds small Gaussian jitter copies so calibration has
enough data on tiny sets (e.g. the offline demo); real experiment runs pass augment=0.
"""
from __future__ import annotations

import numpy as np

from franq_ext.config import Config
from franq_ext.pipeline import Pipeline
from franq_ext.router import CalibratedRouter, fact_features, label_fact
from franq_ext.schema import Example


def collect_router_dataset(pipeline: Pipeline, examples: list[Example]):
    """Return (X, y) of labeled fact-feature rows from `examples`."""
    use_graph = pipeline.cfg.router.use_graph_features
    X, y = [], []
    for ex in examples:
        facts, graph, _ = pipeline.compute_signals(ex)
        for f in facts:
            label = label_fact(f, ex.gold_facts)
            if label is None:
                continue
            X.append(fact_features(f, graph, use_graph))
            y.append(label)
    return X, y


def _augment(X, y, factor: int, sigma: float, seed: int):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    Xs, ys = [X], [y]
    for _ in range(factor):
        noise = rng.normal(0.0, sigma, size=X.shape)
        Xs.append(np.clip(X + noise, 0.0, 1.0))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def train_router(cfg: Config, examples: list[Example], pipeline: Pipeline | None = None,
                 augment: int = 0, sigma: float = 0.05,
                 fact_source_factory=None) -> CalibratedRouter:
    pipeline = pipeline or Pipeline(cfg, fact_source_factory=fact_source_factory)
    X, y = collect_router_dataset(pipeline, examples)
    if augment > 0 and X:
        X, y = _augment(X, y, augment, sigma, cfg.seed)
    return CalibratedRouter(cfg).fit(X, y)
