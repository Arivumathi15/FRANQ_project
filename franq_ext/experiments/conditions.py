"""The ablation ladder. Each condition is just a Config; the difference between two
adjacent rungs isolates exactly one pillar.

  B0 franq_baseline   : fixed IF-ELSE router, no correction        (the FRANQ baseline)
  A1 +router          : learned calibrated router, NO graph feats  (isolates Pillar 2)
  A2 +graph           : learned router WITH graph features         (isolates Pillar 1)
  A3 full             : A2 + budget-bounded correction + regret    (isolates Pillar 3)

Read the improvements down the ladder:
  Pillar 2 = A1 - B0,  Pillar 1 = A2 - A1,  Pillar 3 = A3 - A2 (accuracy/regret).
"""
from __future__ import annotations

from franq_ext.config import Config


def franq_baseline(seed: int = 13) -> Config:
    cfg = Config(seed=seed)
    cfg.router.kind = "fixed"
    cfg.correction.enabled = False
    return cfg


def learned_router_no_graph(seed: int = 13) -> Config:
    cfg = Config(seed=seed)
    cfg.router.kind = "learned"
    cfg.router.use_graph_features = False
    cfg.correction.enabled = False
    return cfg


def learned_router_with_graph(seed: int = 13) -> Config:
    cfg = Config(seed=seed)
    cfg.router.kind = "learned"
    cfg.router.use_graph_features = True
    cfg.correction.enabled = False
    return cfg


def full(seed: int = 13) -> Config:
    cfg = Config(seed=seed)
    cfg.router.kind = "learned"
    cfg.router.use_graph_features = True
    cfg.correction.enabled = True
    return cfg


# Ordered ladder: (short_name, label, builder)
LADDER = [
    ("B0_franq", "FRANQ (fixed rule, no correction)", franq_baseline),
    ("A1_router", "+ Learned router (Pillar 2)", learned_router_no_graph),
    ("A2_graph", "+ Graph features (Pillar 1)", learned_router_with_graph),
    ("A3_full", "+ Correction + regret (Pillar 3)", full),
]
