# franq_ext

**Dependency-Aware Entity–Attribute Verification and Targeted Self-Correction for Faithful RAG** — an extension of **FRANQ** ([arXiv:2505.21072](https://arxiv.org/abs/2505.21072)).

This package implements three integrated contributions on top of FRANQ:

| Pillar | Contribution | Answers FRANQ weakness |
|---|---|---|
| **1** | Dependency-aware **entity–attribute graph** — facts about one entity are checked together | flat, one-fact-at-a-time checking |
| **2** | **Learned, calibrated router** — replaces FRANQ's fixed IF-ELSE switch | rigid decision rule |
| **3** | **Budget-bounded targeted correction** + **correction-regret** safety metric | detection without correction / unsafe correction |

The **correction-regret** metric — *did fixing one fact break a fact that was already correct?* — is the distinctive, easily-verifiable novelty; no compared baseline (RAC, QuCo-RAG, HalluEntity, RHD) measures it.

---

## Code → paper map (every module maps to one part of the paper)

| Module | File | Paper role |
|---|---|---|
| Extraction | `franq_ext/extraction/fact_extractor.py` | answer → (entity, attribute, value) facts |
| Graph (Pillar 1) | `franq_ext/graph/entity_graph.py` | link facts by entity; `dependency_strength`, `entity_suspicion` |
| Retrieval | `franq_ext/retrieval/dense_retriever.py` | dense (FAISS) / lexical (TF-IDF) evidence |
| Faithfulness | `franq_ext/scoring/align_scorer.py` | AlignScore (real) / lexical (offline) |
| Uncertainty | `franq_ext/uncertainty/` | semantic entropy + parametric (token-prob) UQ |
| Router (Pillar 2) | `franq_ext/router/calibrated_router.py` | calibrated learned router |
| FRANQ baseline router | `franq_ext/router/fixed_rule_router.py` | the fixed IF-ELSE it replaces |
| Correction (Pillar 3) | `franq_ext/correction/corrector.py` | budget-bounded targeted correction |
| Re-verification (Pillar 3) | `franq_ext/verification/reverify.py` | recheck + correction-regret |
| Metrics | `franq_ext/metrics/` | AUROC, PRR, ECE, correction-regret |
| Pipeline | `franq_ext/pipeline.py` | all modules in order |
| Experiments | `franq_ext/experiments/` | 6 sequential scripts + `run_all` |

---

## Install

```bash
pip install -r requirements.txt      # core: numpy, scikit-learn, networkx
pip install -e .                     # install the franq_ext package
```

Heavy deps (torch, transformers, sentence-transformers, faiss, datasets, matplotlib) are
**only needed for the real GPU runs**:

```bash
pip install -e ".[full]"
```

> Note: matplotlib is optional — without it, all CSV tables are still produced; only PNG
> figures are skipped.

---

## Run the demo (offline, no GPU, no downloads)

```bash
python -m franq_ext.demo
```

Reproduces the paper's 7-step Einstein walk-through with the deterministic `MockLLM`:
extract → graph → score → route → **flag `birth city`** → **correct Munich→Ulm** →
**correction-regret = 0.00**.

## Run the experiments (offline sample)

```bash
python -m franq_ext.experiments.run_all
```

Produces:
- `results/tables/ablation.csv` — the ablation ladder (below)
- `results/tables/regret_analysis.csv` — safety-checked vs reckless correction
- `results/figures/*.png` — AUROC, PRR, ECE, regret, factual-accuracy figures

### The ablation ladder (each rung isolates one pillar)

| Condition | What changes | Isolates |
|---|---|---|
| `B0_franq` | fixed rule, no correction | FRANQ baseline |
| `A1_router` | + learned calibrated router | **Pillar 2** (vs B0) |
| `A2_graph` | + graph features | **Pillar 1** (vs A1) |
| `A3_full` | + targeted correction + regret | **Pillar 3** (vs A2) |

---

## Run the REAL experiments (free Colab / Kaggle GPU) — for the paper's numbers

The offline sample is a plumbing check. Journal-grade numbers come from real models on a
recognized benchmark. The **same scripts** switch to real backends via environment
variables (see `notebooks/franq_ext_colab.ipynb`):

```bash
export FRANQ_MODE=structured          # RAG generate-then-check (real hallucinations)
export FRANQ_DATASET=popqa_structured # PopQA grouped by subject -> multi-attribute entities
export FRANQ_N=150                    # number of ENTITIES (scale up after a sanity run)
export FRANQ_LLM_BACKEND=hf
export FRANQ_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct   # or meta-llama/Llama-3.2-3B-Instruct
export FRANQ_SCORER_BACKEND=nli                    # modern NLI faithfulness (torch-2 OK)
export FRANQ_NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
export FRANQ_RETRIEVER_BACKEND=dense
export FRANQ_DEVICE=cuda
python -m franq_ext.experiments.run_all
```

- **Why `structured` mode:** the raw QA loaders alone produce no hallucinations (facts read
  from the gold answer are always correct → AUROC undefined, accuracy 1.0→1.0). Structured
  mode adds the missing RAG step: the model **generates** each attribute's value from
  retrieved Wikipedia evidence, so wrong answers are real, labelled hallucinations.
- **Dataset:** `popqa_structured` groups PopQA by subject so each entity has several
  attributes — giving Pillar 1's graph real structure. Contexts are the subject's Wikipedia
  summary (fetched once, cached to `.cache/`).
- **Metrics:** AUROC + PRR (FRANQ detection metrics), ECE (calibration), plus correction
  success and correction-regret (this work).
- **TriviaQA / NQ** (FRANQ base-paper datasets) are single-answer, so they exercise the
  router + correction but not the entity graph; use them as a secondary detection-only run.
- **Faithfulness backend:** use `nli` (an NLI entailment model — the principle AlignScore
  is built on) because the original **AlignScore package is abandoned**: it pins
  `torch<2` / Python `<3.11` and will not install on current Colab (Python 3.12, torch 2.x).
  The `alignscore` backend remains available only on a legacy torch<2 environment.

---

## Reproducibility

- All randomness is seeded (`Config.seed`, default 13; override with `FRANQ_SEED`).
- `MockLLM` uses a **stable hash** (not Python's salted `hash()`), so offline results are
  identical across runs and machines.
- Each experiment writes its exact `Config` to `results/raw/NN_*.config.json`.
- `python -m pytest` runs the full offline test suite (no GPU, no network).

## The mock vs. real split (important, and honest)

- **Offline path** (`MockLLM` + lexical scorer + a small curated world KB in
  `franq_ext/knowledge.py`) exists for development, tests, and the demo. Its correction
  step reads a curated KB, so offline regret is trivially 0 — it is a *plumbing check*, not
  a measurement.
- **Real path** (Colab GPU: Llama-3.2-3B + AlignScore + dense retrieval over the corpus)
  produces the numbers reported in the paper. Correction there uses only the retriever +
  LLM over evidence; gold answers are never leaked into correction.

## Tests

```bash
python -m pytest -q
```
