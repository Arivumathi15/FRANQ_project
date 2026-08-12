"""franq_ext: Dependency-Aware Entity-Attribute Verification and Targeted
Self-Correction for Faithful Retrieval-Augmented Generation.

An extension of FRANQ (arXiv:2505.21072) with three pillars:
  * Pillar 1 - Dependency-aware entity-attribute verification graph.
  * Pillar 2 - Learned, calibrated uncertainty router.
  * Pillar 3 - Budget-bounded targeted correction loop + correction-regret metric.
"""

__version__ = "0.1.0"

from franq_ext.config import Config, default_config

__all__ = ["Config", "default_config", "__version__"]
