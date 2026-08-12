"""The full franq_ext pipeline — all modules in order.

extract -> graph (Pillar 1) -> score + retrieve + UQ -> route (Pillar 2)
        -> targeted correction (Pillar 3) -> re-verify + correction-regret.

The same object runs offline (MockLLM) and on a real model (LocalHFLLM); only the config
changes. Ablations are just different `Config`s (fixed vs learned router, correction
on/off), which is what the experiment scripts sweep over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from franq_ext.config import Config
from franq_ext.llm import build_llm
from franq_ext.llm.client import LLMClient
from franq_ext.extraction import FactExtractor
from franq_ext.graph import EntityGraph
from franq_ext.scoring import build_scorer
from franq_ext.scoring.align_scorer import AlignScorer
from franq_ext.uncertainty import UncertaintyEstimator
from franq_ext.retrieval import build_retriever
from franq_ext.router import build_router
from franq_ext.correction import Corrector, CorrectionRecord
from franq_ext.verification import ReVerifier, ReverifyResult
from franq_ext.schema import Example, Fact, PipelineResult


@dataclass
class RunOutput:
    result: PipelineResult
    facts: list[Fact]
    graph: EntityGraph
    records: list[CorrectionRecord] = field(default_factory=list)
    reverify: ReverifyResult | None = None


class Pipeline:
    def __init__(self, cfg: Config, llm: LLMClient | None = None,
                 scorer: AlignScorer | None = None, router=None,
                 fact_source_factory=None) -> None:
        self.cfg = cfg
        self.llm = llm or build_llm(cfg.llm)
        self.scorer = scorer or build_scorer(cfg.scoring)
        self.router = router or build_router(cfg)
        self.extractor = FactExtractor(self.llm)
        self.uq = UncertaintyEstimator(self.llm, cfg.uq)
        # Optional pluggable fact source (e.g. RAG generation for real entity-attribute
        # datasets). Default None -> free-text extraction from the answer.
        self.fact_source = fact_source_factory(self.llm) if fact_source_factory else None
        # Build the retriever ONCE (its embedding model loads once), then re-index per
        # example. Rebuilding per example reloaded the model thousands of times.
        self.retriever = build_retriever(cfg.retriever)
        # Cache the expensive per-fact signals so the 4 conditions don't recompute them.
        from franq_ext.signal_cache import SignalCache
        self.signal_cache = SignalCache(signature=self._signal_signature())

    def _signal_signature(self) -> str:
        c = self.cfg
        return "|".join(str(x) for x in [
            c.llm.backend, c.llm.model_name, c.uq.n_samples,
            c.scoring.backend, c.scoring.nli_model, c.retriever.backend, c.seed,
        ])

    # ---------------------------------------------------------------- signals
    def compute_signals(self, example: Example):
        """Everything up to (but not including) routing. Reused for router training."""
        self.retriever.index(example.contexts)  # needed by the corrector regardless

        cached = self.signal_cache.get_facts(example)
        if cached is not None:
            graph = EntityGraph().build(cached)  # dependency_strength recomputed (cheap)
            return cached, graph, self.retriever

        facts = self.fact_source(example) if self.fact_source else self.extractor.extract(example)
        graph = EntityGraph().build(facts)
        for f in facts:
            f.faithfulness = self.scorer.score_fact(f, example.contexts)
            hits = self.retriever.retrieve(f"{f.entity} {f.attribute}", top_k=1)
            f.retrieval_confidence = hits[0][1] if hits else 0.0
            self.uq.annotate(f, example.question)
        self.signal_cache.put_facts(example, facts)
        return facts, graph, self.retriever

    # ---------------------------------------------------------------- full run
    def run(self, example: Example) -> RunOutput:
        facts, graph, retriever = self.compute_signals(example)

        # Pillar 2: route each fact (calibrated learned router, or FRANQ fixed rule).
        self.router.route_facts(
            facts, graph, flag_threshold=self.cfg.correction.flag_threshold
        )

        # Snapshot correctness state before touching anything (for correction-regret).
        before_values = {f.key(): f.value for f in facts}

        # Pillar 3: budget-bounded targeted correction of flagged facts.
        records: list[CorrectionRecord] = []
        if self.cfg.correction.enabled:
            corrector = Corrector(self.llm, retriever, self.scorer, self.cfg.correction)
            records = corrector.correct_flagged(facts)

        # Pillar 3: re-verify and compute correction-regret.
        reverifier = ReVerifier(self.scorer)
        reverifier.recompute_faithfulness(facts, example.contexts)
        ev = reverifier.evaluate(facts, example.gold_facts, before_values)

        result = PipelineResult(
            example=example,
            facts=facts,
            correction_regret=ev.correction_regret,
            n_correct_before=ev.n_correct_before,
            n_correct_after=ev.n_correct_after,
            n_flagged=sum(1 for f in facts if f.flagged),
            n_corrected=sum(1 for f in facts if f.corrected),
        )
        return RunOutput(result=result, facts=facts, graph=graph, records=records, reverify=ev)
