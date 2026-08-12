"""Faithfulness scoring (paper Section 5: AlignScore).

Faithfulness = is a fact actually supported by the retrieved evidence? (Distinct from
factuality = is it true in the world.) This is FRANQ's core signal and the gate its
fixed rule switches on.

Two backends behind one interface:
  * RealAlignScorer  - the published AlignScore model (used on GPU runs).
  * LexicalAlignScorer - a dependency-light heuristic for offline dev/tests/demo. It keys
    on whether the claimed value is *asserted* (not negated) in the evidence, which is
    enough to make the demo behave correctly (a value the evidence denies scores low).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from franq_ext.schema import Fact

_NEGATIONS = ("not", "never", "no longer", "isn't", "wasn't", "rather than", "instead of")
_WORD = re.compile(r"[a-z0-9]+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class AlignScorer(ABC):
    @abstractmethod
    def score_claim(self, entity: str, attribute: str, value: str, contexts: list[str]) -> float:
        """Support for '<entity> <attribute> = <value>' given evidence, in [0,1]."""

    def score_fact(self, fact: Fact, contexts: list[str]) -> float:
        return self.score_claim(fact.entity, fact.attribute, fact.value, contexts)


class LexicalAlignScorer(AlignScorer):
    # Tunable support levels for the offline heuristic.
    SUPPORTED = 0.85
    NEGATED = 0.15
    ABSENT = 0.25

    def score_claim(self, entity: str, attribute: str, value: str, contexts: list[str]) -> float:
        value_l = value.strip().lower()
        claim_words = _words(f"{entity} {attribute} {value}")
        occurrence_scores: list[float] = []
        for ctx in contexts:
            for sent in _sentences(ctx):
                sent_l = sent.lower()
                if value_l and value_l in sent_l:
                    negated = any(n in sent_l for n in _NEGATIONS)
                    overlap = len(claim_words & _words(sent)) / max(len(claim_words), 1)
                    base = self.NEGATED if negated else self.SUPPORTED
                    # Small overlap bonus (asserted) so scores aren't purely binary.
                    score = base + (0.0 if negated else 0.10 * overlap)
                    occurrence_scores.append(min(score, 1.0))
        if not occurrence_scores:
            # Value never mentioned: weak (absent) support, but NOT actively denied.
            return float(self.ABSENT)
        # Best assertion wins; a value only ever denied stays at the negated (low) level.
        return float(max(occurrence_scores))


class NLIAlignScorer(AlignScorer):
    """Faithfulness via a modern NLI entailment model (recommended for GPU runs).

    Faithfulness ~= does the evidence ENTAIL the claim? We score entailment probability of
    the hypothesis 'The <attribute> of <entity> is <value>' against each evidence passage
    (premise) and take the max — a claim supported by ANY passage is faithful. This is the
    same NLI principle AlignScore is built on, but it installs cleanly on torch 2.x /
    Python 3.12 where the original AlignScore package does not.
    """

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        # Locate the 'entailment' class index robustly from the model's label map.
        id2label = self.model.config.id2label
        self.entail_idx = next(
            (i for i, lab in id2label.items() if "entail" in str(lab).lower()), 0
        )

    def score_claim(self, entity: str, attribute: str, value: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.25
        hypothesis = f"The {attribute} of {entity} is {value}."
        best = 0.0
        for premise in contexts:
            inputs = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[0]
            probs = self.torch.softmax(logits, dim=-1)
            best = max(best, float(probs[self.entail_idx].item()))
        return best


class RealAlignScorer(AlignScorer):
    """Wraps the published AlignScore checkpoint. Legacy: needs torch<2 / Python<3.11."""

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        from alignscore import AlignScore  # lazy

        self._scorer = AlignScore(
            model="roberta-large",
            batch_size=8,
            device=device,
            ckpt_path=model_path,
            evaluation_mode="nli_sp",
        )

    def score_claim(self, entity: str, attribute: str, value: str, contexts: list[str]) -> float:
        claim = f"The {attribute} of {entity} is {value}."
        evidence = "\n".join(contexts) if contexts else ""
        if not evidence:
            return 0.25
        scores = self._scorer.score(contexts=[evidence], claims=[claim])
        return float(scores[0])


_SCORER_CACHE: dict = {}


def build_scorer(cfg) -> AlignScorer:
    # Cached so the NLI model loads once per process, not once per condition.
    key = (cfg.backend, getattr(cfg, "nli_model", ""), getattr(cfg, "device", "cuda"))
    if key in _SCORER_CACHE:
        return _SCORER_CACHE[key]
    if cfg.backend == "nli":
        scorer: AlignScorer = NLIAlignScorer(cfg.nli_model, getattr(cfg, "device", "cuda"))
    elif cfg.backend == "alignscore":
        # Legacy path; model_path provided via env on a compatible (torch<2) box.
        import os

        scorer = RealAlignScorer(os.environ.get("ALIGNSCORE_CKPT", "AlignScore-large.ckpt"))
    else:
        scorer = LexicalAlignScorer()
    _SCORER_CACHE[key] = scorer
    return scorer
