"""Central configuration for franq_ext.

Every tunable knob lives here so experiment runs are reproducible: a run saves its
`Config` to disk, and re-loading it reproduces the exact setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class LLMConfig:
    # "mock" for offline dev/tests/demo; "hf" for the real Colab/Kaggle GPU runs.
    backend: Literal["mock", "hf"] = "mock"
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_new_tokens: int = 128
    temperature: float = 0.7
    device: str = "cuda"  # ignored by the mock backend


@dataclass
class RetrieverConfig:
    backend: Literal["dense", "lexical"] = "lexical"  # lexical needs no model download
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5


@dataclass
class ScoringConfig:
    # "nli"        -> modern NLI entailment scorer (recommended for GPU runs; torch 2.x OK)
    # "alignscore" -> the original AlignScore checkpoint (legacy; needs torch<2, Python<3.11)
    # "lexical"    -> dependency-light offline heuristic (dev/tests/demo)
    backend: Literal["nli", "alignscore", "lexical"] = "lexical"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    device: str = "cuda"
    # A fact is "faithful" when its support score is >= this threshold.
    faithful_threshold: float = 0.5


@dataclass
class UQConfig:
    n_samples: int = 5          # k generations for semantic entropy
    paraphrase_prompts: int = 3  # ways of asking the same question


@dataclass
class RouterConfig:
    # "learned" = Pillar 2 (calibrated classifier). "fixed" = FRANQ baseline IF-ELSE.
    kind: Literal["learned", "fixed"] = "learned"
    calibration: Literal["isotonic", "sigmoid"] = "isotonic"
    # Pillar 1 ablation switch: when False, the graph-derived features
    # (dependency_strength, entity_suspicion) are zeroed out for the learned router.
    use_graph_features: bool = True
    # FRANQ fixed rule: if AlignScore faithfulness < this, trust semantic-entropy UQ;
    # otherwise trust token-probability UQ.
    fixed_faithful_threshold: float = 0.5


@dataclass
class CorrectionConfig:
    enabled: bool = True
    budget: int = 3             # max targeted retrieval attempts per flagged fact
    flag_threshold: float = 0.5  # router error-probability above which a fact is corrected
    # When True, only accept a replacement that is BETTER supported than the old value
    # (this is what keeps correction-regret low). False = reckless: replace on any change.
    safety_checked: bool = True


@dataclass
class Config:
    seed: int = 13
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    uq: UQConfig = field(default_factory=UQConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(
            seed=raw.get("seed", 13),
            llm=LLMConfig(**raw.get("llm", {})),
            retriever=RetrieverConfig(**raw.get("retriever", {})),
            scoring=ScoringConfig(**raw.get("scoring", {})),
            uq=UQConfig(**raw.get("uq", {})),
            router=RouterConfig(**raw.get("router", {})),
            correction=CorrectionConfig(**raw.get("correction", {})),
        )


def default_config() -> Config:
    """The offline, dependency-light default used by tests and the demo."""
    return Config()
