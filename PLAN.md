# FRANQ-EXT — 4-Day Implementation Plan

**Project:** Extending FRANQ with Dependency-Aware Entity–Attribute Verification and
Targeted Self-Correction for Faithful RAG.

**Your role:** Implementation only. The professor writes the journal paper from your
results. Therefore the implementation must produce **real, reproducible, journal-grade
quantitative results** — not just a demo.

**Locked decisions**
- Dev locally (CPU) with a deterministic `MockLLM`; run **real experiments on free
  Colab/Kaggle GPU (T4)**.
- Real models: **`Llama-3.2-3B-Instruct`** (same family as FRANQ's Llama 3, fits a free T4) + **real AlignScore**.
- Datasets: **PopQA** (primary — entity-attribute triples, ideal for Pillar 1) **+ TriviaQA** (from the FRANQ base paper, for comparability). Natural Questions is the drop-in alternative to TriviaQA.
- **Metrics aligned to the base paper:** detection quality via **AUROC + PRR** (as in FRANQ); plus our extension metrics: **correction success rate**, **correction-regret**, and router **calibration (ECE)**.

### Base paper facts (arXiv 2505.21072, "Faithfulness-Aware Uncertainty Quantification")
- FRANQ's fixed IF-ELSE switches between **semantic entropy** and **token-probability UQ**, gated by an **AlignScore** faithfulness threshold. That switch is exactly what the learned calibrated router (Pillar 2) replaces.
- Map our signals accordingly: "parametric knowledge UQ" == token/sequence-probability confidence; faithfulness == AlignScore.
- Deliverable: `franq_ext` package + unit tests + runnable demo + **experiment results
  (CSV tables, figures, logs) + README/results doc** the professor can lift into the paper.

---

## Publishability checklist (what the paper needs from your code)
A reviewer at IEEE Access (or similar) will look for these. Every one must be produced by code:
- [ ] All 3 pillars implemented and each maps 1:1 to a code module (Section 6 of the explainer).
- [ ] **Reproduced baselines**: FRANQ fixed-rule router (no correction); a RAC-style flat corrector (no graph, no regret).
- [ ] **Ablation**: FRANQ → +graph → +router → +correction (full), on both datasets.
- [ ] Headline metrics: faithfulness, factual accuracy (EM/F1), correction success rate,
      **correction-regret**, router **calibration (ECE)**.
- [ ] Reproducibility: fixed seeds, saved config per run, one-command runner, logged raw outputs.
- [ ] Results as CSV tables + matplotlib figures, ready to paste into the paper.

## Day 0 (pre-flight, ~1 hr — do before Day 1)
- Read the **FRANQ base paper** and note its **exact datasets, metrics, and baseline setup**.
  Matching the base paper's protocol is what makes the extension credible to reviewers.
- Confirm PopQA + second dataset are downloadable; skim RAC (2025) for the flat-correction baseline.
- Create a Colab/Kaggle notebook, verify GPU + `pip install` works.

---

## Package layout
```
franq_ext/
  __init__.py
  config.py                 # dataclass configs, seeds, budgets, model names
  llm/
    client.py               # LLMClient interface
    mock_llm.py             # deterministic backend (dev + CI + demo)
    local_hf.py             # real backend: Qwen2.5-3B / Llama-3.2-3B
  data/
    loaders.py              # PopQA + second dataset loaders -> unified schema
    sample/                 # tiny bundled set for offline demo/tests
  retrieval/
    dense_retriever.py      # sentence-transformers + FAISS
  extraction/
    fact_extractor.py       # answer -> (entity, attribute, value) facts   [Pillar 0]
  graph/
    entity_graph.py         # link facts by entity into dependency graph    [Pillar 1]
  scoring/
    align_scorer.py         # AlignScore wrapper (+ lightweight fallback)
  uncertainty/
    semantic_entropy.py     # k-sample + NLI clustering
    parametric_uq.py        # closed-book agreement
  router/
    features.py             # assemble feature vector per fact
    calibrated_router.py    # sklearn + CalibratedClassifierCV            [Pillar 2]
    fixed_rule_router.py    # FRANQ baseline (IF-ELSE)
  correction/
    corrector.py            # budget-bounded targeted correction           [Pillar 3]
    knowledge_base.py       # gold KB for targeted lookups (demo) / retriever (real)
  verification/
    reverify.py             # re-check + correction-regret metric          [Pillar 3]
  pipeline.py               # ties all modules into one run()
  demo.py                   # end-to-end 7-step Einstein walk-through
  experiments/
    01_baseline_franq.py
    02_add_graph.py
    03_add_router.py
    04_add_correction.py
    05_compute_regret.py
    06_final_results.py     # aggregate -> tables + figures
    run_all.py
  metrics/
    faithfulness.py  factuality.py  regret.py  calibration.py
tests/                      # pytest, one file per module
results/                    # CSVs + figures (generated)
README.md   pyproject.toml   requirements.txt
```

---

## Day 1 — Scaffold, data, retrieval, extraction, graph (Pillars 0–1)
- Package skeleton, `pyproject.toml`, `requirements.txt`, `config.py`, seeds.
- `LLMClient` interface + `MockLLM` (deterministic) + `LocalHFLLM` stub.
- Dataset loaders → unified `{question, gold_answer, gold_facts, contexts}` schema; bundle a tiny sample set.
- `dense_retriever.py` (MiniLM/bge + FAISS) over the corpus.
- **Extraction module**: answer → `(entity, attribute, value)` facts.
- **Graph module (Pillar 1)**: link facts sharing an entity; expose per-entity fact groups.
- Unit tests for extraction + graph.
- **Milestone:** `answer → facts → entity graph` runs offline against MockLLM.

## Day 2 — Scoring, uncertainty, router (Pillar 2)
- `align_scorer.py`: real AlignScore wrapper + lightweight NLI/embedding fallback (same interface).
- `semantic_entropy.py` (k samples + bidirectional-NLI clustering) and `parametric_uq.py`.
- `router/features.py`: faithfulness score, retrieval confidence, entity type,
  attribute-dependency strength (from the graph), semantic entropy, parametric UQ.
- `calibrated_router.py` (GradientBoosting/LogReg + `CalibratedClassifierCV`, report ECE) and
  `fixed_rule_router.py` (FRANQ baseline).
- Unit tests; verify calibration curve on a dev split.
- **Milestone:** router emits a calibrated per-fact decision; baseline router works for comparison.

## Day 3 — Correction loop, re-verification, full pipeline, demo (Pillar 3)
- `corrector.py`: for flagged facts, targeted search (e.g. "Einstein birth city"), retrieve,
  replace **only** that value, within a retrieval **budget** (configurable N).
- `reverify.py`: recompute faithfulness; compute **correction-regret** =
  (facts correct-before that became wrong-after) ÷ (facts correct-before).
- `pipeline.py`: extraction → graph → scoring → UQ → router → correction → reverify.
- `demo.py`: prints the paper's **7-step Einstein walk-through** end-to-end (offline, MockLLM).
- Unit tests for corrector + regret (incl. a case where a naive fix *would* break a good fact,
  proving the metric catches it).
- **Milestone:** `python -m franq_ext.demo` reproduces the walk-through; full pipeline runs.

## Day 4 — Real experiments, metrics, tables/figures, reproducibility, docs
- Swap `LocalHFLLM` on in Colab/Kaggle; run the 6 experiment scripts over PopQA + second dataset subset.
- `06_final_results.py`: aggregate → **ablation table** (FRANQ → +graph → +router → +full),
  per-dataset, with faithfulness / factual acc / correction success / **regret** / **ECE**.
- Generate matplotlib figures (ablation bars, calibration reliability plot, regret distribution).
- Reproducibility pass: seeds, saved run configs, `run_all.py`, logged raw outputs in `results/`.
- `README.md`: install, run, and a table mapping **each module → paper pillar → result**.
- Final unit-test sweep; ensure CI-style `pytest` passes offline via MockLLM.
- **Milestone:** reproducible results + figures + docs handed off in a state the professor can cite directly.

---

## Baselines & experiment protocol
- **B0 FRANQ (base):** fixed-rule router, no correction.
- **B1 RAC-style:** flat per-fact correction, no entity graph, no regret metric.
- **Ours (ablated):** +graph, +router, +correction — added one at a time.
- Same retriever, same LLM, same subset, same seeds across all conditions (only the component under test changes).
- Report mean ± std over ≥3 seeds for the headline metrics.

## Risks & mitigations
- **Slow LLM sampling (semantic entropy):** cache all generations to disk; small k (e.g. 5); subset size 500–1000.
- **AlignScore checkpoint size/VRAM:** T4 is enough; keep batch small; fallback scorer for CPU dev.
- **Colab timeouts:** checkpoint results per script; `run_all.py` resumes from saved partials.
- **Time overrun:** PopQA-only + full ablation is the acceptable-minimum fallback; second dataset is the stretch.

## Definition of done
`pytest` green offline; `demo.py` shows the 7-step trace; `run_all.py` regenerates every CSV/figure
in `results/`; README maps modules→pillars→results; the professor can write the paper from `results/`
without re-running anything.
