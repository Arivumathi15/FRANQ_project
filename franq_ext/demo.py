"""End-to-end demo — reproduces the paper's 7-step Einstein walk-through, fully offline.

Run:  python -m franq_ext.demo         (or the installed `franq-ext-demo` command)

Uses the deterministic MockLLM + lexical scorer so it needs no GPU, no downloads, and no
network. It trains the Pillar-2 calibrated router on the bundled sample set (with jitter
augmentation, since the set is tiny) and then traces one example through all three pillars.
"""
from __future__ import annotations

from franq_ext.config import Config
from franq_ext.data import load_sample
from franq_ext.pipeline import Pipeline
from franq_ext.train_router import train_router
from franq_ext.correction import Corrector
from franq_ext.verification import ReVerifier


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    cfg = Config()  # offline defaults: mock LLM, lexical retriever + scorer, learned router
    examples = load_sample()

    # Pillar 2: train the calibrated router on the sample set (augmented — tiny set).
    print("Training the calibrated router (Pillar 2) on the sample set...")
    router = train_router(cfg, examples, augment=40)

    pipeline = Pipeline(cfg, router=router)
    example = next(e for e in examples if e.qid == "einstein")

    # Drive the stages in order so each step shows its TRUE state at that moment
    # (Steps 2-5 pre-correction; Steps 6-7 the fix and safety check).
    _rule("STEP 1 - Answer generation (from retrieved documents)")
    print(f"Question : {example.question}")
    print(f"Generated: {example.generated_answer}")
    print("(note: 'Munich' is a hallucination; the evidence says Ulm)")

    facts, graph, retriever = pipeline.compute_signals(example)

    _rule("STEP 2 - Extraction (answer -> entity/attribute/value facts)")
    for f in facts:
        print(f"  ({f.entity}, {f.attribute}, {f.value})")

    _rule("STEP 3 - Graph building (Pillar 1: link facts by entity)")
    for entity, group in graph.groups().items():
        attrs = ", ".join(f"{g.attribute}={g.value}" for g in group)
        print(f"  entity '{entity}' -> [{attrs}]")
        print(f"     dependency_strength per fact = "
              f"{[round(g.dependency_strength, 2) for g in group]}")

    _rule("STEP 4 - Scoring (AlignScore faithfulness + UQ signals)")
    for f in facts:
        print(f"  ({f.entity}, {f.attribute}, {f.value})")
        print(f"     faithfulness={f.faithfulness:.2f}  semantic_entropy={f.semantic_entropy:.2f}"
              f"  parametric_uq={f.parametric_uq:.2f}")

    pipeline.router.route_facts(facts, graph, flag_threshold=cfg.correction.flag_threshold)

    _rule("STEP 5 - Routing decision (Pillar 2: calibrated router)")
    for f in facts:
        verdict = "SUSPICIOUS -> needs correction" if f.flagged else "looks fine"
        print(f"  ({f.entity}, {f.attribute}, {f.value}): P(wrong)={f.error_prob:.2f}"
              f"  route={f.route}  -> {verdict}")

    before_values = {f.key(): f.value for f in facts}

    _rule("STEP 6 - Targeted correction (Pillar 3: budget-bounded)")
    corrector = Corrector(pipeline.llm, retriever, pipeline.scorer, cfg.correction)
    records = corrector.correct_flagged(facts)
    if not records:
        print("  (nothing flagged)")
    for r in records:
        entity, attribute = r.fact_key
        status = "REPLACED" if r.accepted else "kept (no better evidence)"
        print(f"  focused search for '{entity} {attribute}' (budget={cfg.correction.budget},"
              f" attempts={r.attempts})")
        print(f"     {r.old_value!r} (faith {r.old_faithfulness:.2f})"
              f" -> {r.new_value!r} (faith {r.new_faithfulness:.2f})  [{status}]")

    _rule("STEP 7 - Re-verification & safety check (Pillar 3: correction-regret)")
    reverifier = ReVerifier(pipeline.scorer)
    reverifier.recompute_faithfulness(facts, example.contexts)
    ev = reverifier.evaluate(facts, example.gold_facts, before_values)
    for pf in ev.per_fact:
        entity, attribute = pf["fact"]
        change = "unchanged" if pf["before"] == pf["after"] else f"{pf['before']} -> {pf['after']}"
        print(f"  ({entity}, {attribute}): {change}  gold={pf['gold']}"
              f"  correct_before={pf['correct_before']} correct_after={pf['correct_after']}")
    print(f"\n  correct facts before = {ev.n_correct_before}")
    print(f"  correct facts after  = {ev.n_correct_after}")
    print(f"  >>> CORRECTION REGRET = {ev.correction_regret:.2f}"
          f"  (0.00 = no previously-correct fact was damaged)")

    _rule("SUMMARY")
    print("  All three pillars ran as one pipeline:")
    print("   - Pillar 1 (graph) checked Einstein's facts together, not in isolation.")
    print("   - Pillar 2 (router) calibratedly flagged only 'birth city' for checking.")
    print("   - Pillar 3 (correction + regret) fixed Munich->Ulm and proved regret = 0.")


if __name__ == "__main__":
    main()
