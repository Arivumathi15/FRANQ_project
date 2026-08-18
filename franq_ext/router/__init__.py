from franq_ext.router.features import (
    FEATURE_NAMES,
    fact_features,
    label_fact,
    build_training_matrix,
)
from franq_ext.router.calibrated_router import CalibratedRouter
from franq_ext.router.fixed_rule_router import FixedRuleRouter


def build_router(cfg):
    """Factory: Pillar-2 learned router, or the FRANQ fixed-rule baseline."""
    if cfg.router.kind == "fixed":
        return FixedRuleRouter(
            faithful_threshold=cfg.router.fixed_faithful_threshold,
            flag_threshold=cfg.router.flag_threshold,
        )
    return CalibratedRouter(cfg)


__all__ = [
    "FEATURE_NAMES",
    "fact_features",
    "label_fact",
    "build_training_matrix",
    "CalibratedRouter",
    "FixedRuleRouter",
    "build_router",
]
