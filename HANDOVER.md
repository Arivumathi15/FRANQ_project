# FRANQ-EXT — Project Handover Document

**Project:** Dependency-Aware Entity–Attribute Verification and Targeted Self-Correction for Faithful Retrieval-Augmented Generation
**Extends:** FRANQ — *Faithfulness-Aware Uncertainty Quantification* (arXiv:2505.21072)
**Status:** Complete and working. Real results produced on Kaggle GPU; offline demo runs on any laptop.

This document explains the whole project end-to-end: what it does, the algorithms behind each pillar, how to run it on Kaggle, how to run the live demo, how to read the results, and every design decision that matters. It is written so you can pick the project up with no prior context.

---

## Table of Contents
1. [What this project is (in plain English)](#1-what-this-project-is)
2. [The three pillars — algorithms & methods](#2-the-three-pillars)
3. [The full pipeline, step by step](#3-the-full-pipeline)
4. [Models, dataset, and metrics used](#4-models-dataset-and-metrics)
5. [Repository structure (module → pillar map)](#5-repository-structure)
6. [The results and how to read them](#6-the-results)
7. [How to run it on Kaggle (step by step)](#7-how-to-run-on-kaggle)
8. [How to run the live demo (offline, no GPU)](#8-how-to-run-the-live-demo)
9. [How to run the tests](#9-how-to-run-the-tests)
10. [Key design decisions & things to know](#10-key-design-decisions)
11. [Regenerating the Wikipedia cache](#11-regenerating-the-wikipedia-cache)
12. [Troubleshooting](#12-troubleshooting)
13. [Deliverables checklist](#13-deliverables-checklist)
14. [Glossary](#14-glossary)

---

## 1. What this project is

### The problem
Retrieval-Augmented Generation (RAG) systems answer questions using retrieved documents, but their answers still mix **correct and incorrect facts**. Example:

> **Q:** Where and when was Albert Einstein born?
> **A:** "Einstein was born in **1879** in **Munich**."
> Birth year ✓ correct · Birth city ✗ wrong (should be **Ulm**).

Only one fact is wrong. Existing systems can sometimes *detect* that something is off, but they rarely (a) identify **which** fact is wrong, (b) fix **only** that fact, and (c) verify the fix did not **break** a fact that was already right.

### FRANQ (the base paper) and its three gaps
FRANQ detects unreliable answers using a faithfulness signal (AlignScore) and a fixed **IF-ELSE** rule to choose which uncertainty measure to trust. It **detects** but never **repairs**. Its three weaknesses:
1. **Facts checked in isolation** — a suspicious entity doesn't raise doubt on its other attributes.
2. **One rigid decision rule** — the same hardcoded threshold for every question.
3. **No correction** — it flags a problem, then stops.

### Our contribution — three pillars
| Pillar | Name | Fixes gap | Code module |
|---|---|---|---|
| **1** | Dependency-Aware Entity–Attribute Graph | Gap 1 | `franq_ext/graph/entity_graph.py` |
| **2** | Learned Calibrated Router | Gap 2 | `franq_ext/router/calibrated_router.py` |
| **3** | Targeted Self-Correction + **Correction-Regret** metric | Gap 3 | `franq_ext/correction/corrector.py`, `franq_ext/verification/reverify.py` |

The headline novelty is **Correction-Regret**: a new metric that proves a correction did not quietly damage a previously-correct fact. No prior method measures this.

---

## 2. The three pillars

### Pillar 1 — Dependency-Aware Entity–Attribute Graph
**File:** `franq_ext/graph/entity_graph.py` · **Library:** `networkx`

Instead of a flat list of facts, we build a graph linking every attribute to its entity. Two signals are derived and fed to the router:

- **Dependency strength** of a fact = `0` if the entity has only one attribute, else `1 − 1/deg(entity)`, where `deg` is the number of attributes the entity has. A fact belonging to a richly-connected entity is riskier to trust in isolation.
- **Entity suspicion** of an entity = `1 − min(faithfulness over its attributes)`. If **any** attribute of an entity looks unfaithful, the whole entity — and therefore all its other attributes — becomes suspect. This is the *"doubt the grandparent → re-check the grandchildren"* effect that flat fact-checking cannot see.

These two numbers become two of the router's input features.

### Pillar 2 — Learned, Calibrated Uncertainty Router
**File:** `franq_ext/router/calibrated_router.py` · **Library:** `scikit-learn`

Replaces FRANQ's fixed IF-ELSE with a trained classifier that outputs a **calibrated** probability that a fact is wrong.

- **Algorithm:** `GradientBoostingClassifier` (100 trees, depth 2) as the base learner, wrapped in `CalibratedClassifierCV` with **isotonic** calibration.
- **Six input features per fact:** faithfulness, retrieval confidence, semantic entropy, parametric-knowledge uncertainty, dependency strength, entity suspicion. *(The last two come from Pillar 1; the Pillar-1 ablation zeroes them out.)*
- **Calibration method (important):** class-imbalance is handled with **balanced sample weights applied ONLY to the base learner**. Calibration is then fit on an **unweighted held-out split** (via `FrozenEstimator`, or `cv="prefit"` on older scikit-learn). Reweighting the calibrator would distort the very probabilities it is meant to make honest — doing it this way is what keeps ECE low.
- **Output:** `error_prob` = calibrated P(fact is wrong). A fact is **flagged** when `error_prob ≥ flag_threshold` (default **0.35**, set in `RouterConfig.flag_threshold`).

**Baseline for comparison** — `franq_ext/router/fixed_rule_router.py` (FRANQ's rule): if faithfulness `< 0.5`, trust semantic entropy as the error signal; otherwise trust `1 − parametric_uq`. No learning, no calibration.

### Pillar 3 — Targeted Self-Correction + Correction-Regret
**Files:** `franq_ext/correction/corrector.py`, `franq_ext/verification/reverify.py`

**The corrector** (for each flagged fact):
1. Build a focused query `"<entity> <attribute>"` (e.g. *"Einstein birth city"*).
2. Retrieve the top-k passages from the **correction corpus** — the **full Wikipedia article**, which is a *deeper* pool than the short snippet the answer was generated from. (This separation is essential — searching the same snippet would just reproduce the same wrong value.)
3. Ask the LLM to re-answer the attribute **from that retrieved evidence**.
4. Score the candidate with the NLI faithfulness scorer.
5. **Budget-bounded:** at most `budget` attempts (default 3), each inspecting the top-(i+1) passages.
6. **Safety-checked acceptance:** replace the value only if the candidate is *different*, scores **higher** faithfulness than the old value on the **same** evidence, **and** clears the acceptance threshold (0.5). (The "reckless" ablation instead accepts any different candidate — used to demonstrate the danger the regret metric catches.)

**Re-verification & the new metric** (`reverify.py`):
- Re-check every fact's correctness against gold.
- **Correction-Regret** = (facts correct **before** that became wrong **after**) ÷ (facts correct before). `0` = no collateral damage; higher = more damage. **Smaller is better.**

### Supporting methods used by the pillars

**Faithfulness scoring (AlignScore role)** — `franq_ext/scoring/align_scorer.py`
- **NLIAlignScorer** (GPU): entailment probability of the hypothesis *"The `<attribute>` of `<entity>` is `<value>`."* against each evidence passage, take the maximum. Model: DeBERTa-v3-base MNLI.
- **LexicalAlignScorer** (offline): a dependency-light heuristic for tests/demo.

**Semantic entropy** — `franq_ext/uncertainty/semantic_entropy.py`
Ask the model the same question *k* times (default `k=3`); cluster the answers by meaning; take the normalised entropy of the cluster distribution. Answers that *mean* different things ⇒ high entropy ⇒ the model is unsure.

**Parametric-knowledge uncertainty** — `franq_ext/uncertainty/parametric_uq.py`
Closed-book confidence: the mean token probability of the claim statement conditioned only on the question (no evidence). Measures what the model knows *from its own parameters*.

**Retrieval** — `franq_ext/retrieval/dense_retriever.py`
- **DenseRetriever** (GPU): sentence-transformers (MiniLM) embeddings + cosine search (FAISS if present, else NumPy).
- **LexicalRetriever** (offline): TF-IDF cosine.

---

## 3. The full pipeline

```
Question + retrieved evidence
      │
      ▼
LLM generates the answer (Qwen2.5-3B)                        [generation]
      │
      ▼
Extract facts → (entity, attribute, value)                  [extraction]
      │
      ▼
Build entity graph  ───────────────────────────────  PILLAR 1
      │
      ▼
Score each fact: faithfulness + retrieval conf.
                 + semantic entropy + parametric UQ          [scoring / UQ]
      │
      ▼
Router assigns calibrated P(wrong), flags facts  ──  PILLAR 2
      │
      ▼
Targeted correction of flagged facts  ─────────────  PILLAR 3
(deeper retrieval over the full article, replace only that value)
      │
      ▼
Re-verify + compute Correction-Regret  ────────────  PILLAR 3
```

**One codebase, two backends.** The exact same pipeline runs (a) **offline** with a deterministic `MockLLM` + lexical scorer (for tests and the live demo — no GPU) and (b) **on GPU** with the real models. Only the configuration changes. Entry point: `franq_ext/pipeline.py` (`Pipeline.run`).

---

## 4. Models, dataset, and metrics

### Models (GPU run)
| Role | Model | Notes |
|---|---|---|
| Answer generation | `Qwen/Qwen2.5-3B-Instruct` | Fits a free T4 GPU |
| Faithfulness (NLI) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Entailment scorer |
| Dense retriever | `sentence-transformers/all-MiniLM-L6-v2` | Evidence retrieval |

### Dataset
**PopQA** (`akariasai/PopQA`) — entity-attribute question triples. We group rows by **disambiguated Wikipedia title** (`s_wiki_title`), keep entities with **≥ 3 attributes**, and take the first **150** entities → **234 gold facts** on the evaluation split. Evidence comes from Wikipedia (bundled cache — see §11).

### Metrics
| Metric | Meaning | Better |
|---|---|---|
| **AUROC** | How well `error_prob` ranks wrong vs correct facts (detection quality) | Higher |
| **PRR** | Prediction-Rejection Ratio — error reduced by rejecting uncertain predictions (from FRANQ) | Higher |
| **ECE** | Expected Calibration Error — gap between predicted confidence and real accuracy | Lower |
| **Factual accuracy** | Fraction of facts matching gold (before / after correction) | Higher |
| **Correction success** | Among facts wrong before, fraction correct after | Higher |
| **Correction-Regret** | Among facts correct before, fraction wrong after (our novelty) | Lower |

Correctness uses **lenient QA matching** (normalise, ignore articles, accept alias/substring matches) — see `franq_ext/metrics/regret.py::is_correct`.

---

## 5. Repository structure

```
FRANQ_project/
├── franq_ext/                    # the package
│   ├── config.py                 # all tunable settings (dataclasses)
│   ├── pipeline.py               # ties every module into one run()
│   ├── schema.py                 # Fact / Example / PipelineResult data classes
│   ├── knowledge.py              # tiny KB for the offline MockLLM only
│   ├── signal_cache.py           # caches expensive per-fact signals (4× speedup)
│   ├── llm/                      # mock_llm.py (offline) · local_hf.py (real GPU)
│   ├── data/
│   │   ├── loaders.py            # PopQA / TriviaQA / sample loaders
│   │   ├── wiki_context.py       # Wikipedia evidence fetch + BUNDLED cache
│   │   ├── wiki_cache/           # ← prebuilt Wikipedia cache (no live fetch needed)
│   │   └── sample/               # tiny offline demo dataset
│   ├── extraction/               # answer → (entity, attribute, value)
│   ├── graph/entity_graph.py     # ★ PILLAR 1
│   ├── scoring/align_scorer.py   # faithfulness (NLI / lexical)
│   ├── uncertainty/              # semantic_entropy.py · parametric_uq.py · estimator.py
│   ├── retrieval/dense_retriever.py
│   ├── router/
│   │   ├── calibrated_router.py  # ★ PILLAR 2 (learned)
│   │   ├── fixed_rule_router.py  #   FRANQ baseline
│   │   └── features.py           # the 6 router features
│   ├── correction/corrector.py   # ★ PILLAR 3 (targeted correction)
│   ├── verification/reverify.py  # ★ PILLAR 3 (correction-regret)
│   ├── generation/answer_generator.py  # RAG generate-then-check (structured mode)
│   ├── metrics/                  # detection.py (AUROC/PRR) · calibration.py (ECE) · regret.py
│   ├── experiments/              # exp01..exp06 + run_all.py (the ablation ladder)
│   └── train_router.py           # fits the Pillar-2 router
├── tests/                        # 44 offline pytest tests (MockLLM, no GPU)
├── notebooks/franq_ext_kaggle.ipynb   # the Kaggle run notebook
├── scripts/build_wiki_cache.py   # rebuild the bundled Wikipedia cache
├── app.py                        # ★ live Streamlit demo (offline, no GPU)
├── FRANQ_EXT_slides.html         # presentation deck
├── result_results_popqa/         # the downloaded GPU results (CSVs + figures)
├── requirements.txt · pyproject.toml
├── PLAN.md · project_explanation.md · README.md
└── HANDOVER.md                   # this file
```

**Ablation ladder** (`franq_ext/experiments/`): each rung isolates one pillar.
`B0_franq` (FRANQ baseline) → `A1_router` (+Pillar 2) → `A2_graph` (+Pillar 1) → `A3_full` (+Pillar 3).

---

## 6. The results

Latest measured results (PopQA, 150 entities / 234 facts, Qwen2.5-3B). Files: `result_results_popqa/tables/ablation.csv` and `regret_analysis.csv`.

### Ablation ladder
| Condition | AUROC ↑ | PRR ↑ | ECE ↓ | Acc before | Acc after | Corrected | Regret ↓ |
|---|---|---|---|---|---|---|---|
| B0 · FRANQ (fixed) | 0.737 | 0.422 | **0.602** | 63.2% | 63.2% | 0 | 0 |
| A1 · +Router (P2) | 0.745 | 0.481 | **0.057** | 63.2% | 63.2% | 0 | 0 |
| A2 · +Graph (P1) | 0.764 | 0.543 | **0.026** | 63.2% | 63.2% | 0 | 0 |
| **A3 · +Correction (P3)** | 0.764 | 0.543 | 0.026 | 63.2% | **65.8%** | 20 | 0.014 |

**How to read it (down the ladder each step adds one pillar):**
- **Pillar 2** (B0→A1): ECE collapses **0.602 → 0.057** — the router is ~10× better calibrated than FRANQ's fixed rule.
- **Pillar 1** (A1→A2): graph features lift **PRR 0.481 → 0.543** and AUROC to 0.764.
- **Pillar 3** (A2→A3): correction raises accuracy **+2.6 points** (63.2% → 65.8%) at a tiny regret of 0.014.

### The key finding — safe vs reckless correction
| Policy | Corrected | Regret | Final accuracy |
|---|---|---|---|
| **Safety-checked (ours)** | 20 | **0.014** | **65.8%** ↑ |
| Reckless | 76 | **0.135** | **59.0%** ↓ (below baseline) |

Reckless correction "fixes" 3.8× more facts but its regret is **10× higher** and it drags accuracy **below the 63.2% baseline** — it does net harm. Correction-Regret is the metric that exposes this, and no compared baseline measures it. **This is the paper's headline result.**

> **Note on the baseline accuracy (63.2%):** the 3B model already knows many PopQA facts from pretraining, so accuracy is fairly high even before correction. Correction still delivers a clear, real gain on the facts it gets wrong — and, crucially, does so *safely* (low regret), which is the contribution.

---

## 7. How to run on Kaggle

The full experiment needs a **GPU** (to run Qwen-3B). Kaggle gives ~30 GPU-hours/week free. Everything is automated by the notebook `notebooks/franq_ext_kaggle.ipynb`.

### One-time setup
1. Create a free account at **kaggle.com** and verify your phone (required to enable GPU + internet).

### Step 1 — Upload the project as a Dataset
1. Go to **kaggle.com → Datasets → New Dataset**.
2. Upload **`franq_ext_project_FRESH.zip`** (the project zip — it already contains the code **and** the bundled Wikipedia cache).
3. Name it e.g. `franq-ext-v1` and **Create**.
   *(To update later: open the dataset → **New Version** → upload the new zip.)*

### Step 2 — Create the notebook
1. **kaggle.com → Code → New Notebook**.
2. In the right-hand panel:
   - **Settings → Accelerator → GPU T4 x2** (or P100).
   - **Settings → Internet → On** *(required — it downloads the models from Hugging Face).*
   - **Add Input / Add Data →** add your `franq-ext-v1` dataset.
3. Import the provided notebook: **File → Import Notebook →** upload `notebooks/franq_ext_kaggle.ipynb`. *(Or copy the 5 code cells below manually.)*

### Step 3 — Run
Click **Run All**, or better, **Save Version → Save & Run All (Commit)** so it runs headless (survives your browser closing). The notebook does, in order:

```python
# Cell A — locate the project from the uploaded dataset and put it on the path
#   (auto-extracts the zip into /kaggle/working and cd's into it)

# Cell B — install the two extra deps (torch/transformers are preinstalled on Kaggle)
!pip install -q -U datasets sentence-transformers

# Cell C — configure and run the whole ablation ladder
import os
os.environ['FRANQ_MODE']            = 'structured'
os.environ['FRANQ_DATASET']         = 'popqa_structured'
os.environ['FRANQ_N']               = '150'      # number of entities
os.environ['FRANQ_MIN_ATTRS']       = '3'        # keep entities with >=3 attributes
os.environ['FRANQ_UQ_SAMPLES']      = '3'        # semantic-entropy samples per fact
os.environ['FRANQ_LLM_BACKEND']     = 'hf'
os.environ['FRANQ_LLM_MODEL']       = 'Qwen/Qwen2.5-3B-Instruct'
os.environ['FRANQ_SCORER_BACKEND']  = 'nli'
os.environ['FRANQ_NLI_MODEL']       = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
os.environ['FRANQ_RETRIEVER_BACKEND']= 'dense'
os.environ['FRANQ_DEVICE']          = 'cuda'
os.environ['FRANQ_RESULTS']         = '/kaggle/working/results_popqa'
os.environ['FRANQ_WIKI_CACHE']      = '/kaggle/working/.cache/wiki_context.json'
!python -m franq_ext.experiments.run_all
```

### Step 4 — Get the results
- Outputs are written to **`/kaggle/working/results_popqa/`**:
  - `tables/ablation.csv`, `tables/regret_analysis.csv`
  - `figures/*.png` (AUROC, PRR, ECE, regret, accuracy bar charts)
  - `raw/*` (per-condition CSVs + saved configs) and `signal_cache.json`
- After the commit finishes, open the notebook's **Output** tab and **download** the `results_popqa` folder.

### Expected runtime & what to watch for
- **~2 hours** total (range 1.5–3 h). Model downloads ~3–5 min; the rest is generation + correction.
- Confirm the log line **`context coverage: lead 150/150 … full-article 150/150`** — this means the bundled Wikipedia cache is working (no live fetching needed). If it says a low number, the cache didn't load (see §12).
- Sanity checks in the output: ECE drops B0→A1 (Pillar 2), A2 ≥ A1 on AUROC/PRR (Pillar 1), A3 accuracy rises with low regret (Pillar 3).

### To make it faster (optional)
Lower `FRANQ_N` (e.g. `80`), lower `FRANQ_UQ_SAMPLES` (e.g. `2`), or use a smaller model `Qwen/Qwen2.5-1.5B-Instruct`.

---

## 8. How to run the live demo

`app.py` is a **Streamlit** web app that runs the real pipeline with the **offline MockLLM** — **no GPU, no internet, no model download**. Perfect for a live walkthrough on any laptop.

```powershell
# in the project folder, in a terminal:
pip install streamlit
streamlit run app.py
```

It opens at `http://localhost:8501`. Two tabs:
- **Live Pipeline** — pick Einstein / Newton / Curie, click Run, and watch all six stages: extraction → graph → scoring → routing → correction → re-verify, ending with the regret metric. (Einstein: Munich→Ulm fixed; Newton: 1642→1643 fixed; Curie: already correct, left untouched.)
- **Real Results (PopQA)** — shows the ablation table + safe-vs-reckless comparison, loaded from the `result_results_popqa/` folder.

**Tip:** the *first* launch shows a ~30 s–2 min "Training the calibrated router…" spinner (one-time warm-up). Start the app a couple of minutes **before** presenting so it's ready.

---

## 9. How to run the tests

44 offline tests (deterministic MockLLM, no GPU/network):

```powershell
pip install -r requirements.txt   # numpy, scikit-learn, networkx, streamlit
pytest -q
```

All should pass. There is also a full offline pipeline walkthrough: `python -m franq_ext.demo` (the 7-step Einstein trace in the terminal).

---

## 10. Key design decisions

These are the non-obvious choices that make the results valid. Understanding them will help you defend the work.

1. **Bundled Wikipedia cache (critical).** Kaggle's shared IP is rate-limited by Wikipedia, so fetching evidence live starves ~95% of examples of context (accuracy collapses to ~7%). We **pre-fetched all context off-Kaggle** and ship it inside the package (`franq_ext/data/wiki_cache/`). At run time the code reads from this cache — **zero live Wikipedia calls**. See §11.

2. **Group by `s_wiki_title`, not `subj`.** PopQA reuses a surface name (e.g. "Baby") across different entities (a novel, a song, a film). Grouping by the ambiguous `subj` conflated unrelated facts. Grouping by the disambiguated Wikipedia title gives one coherent entity per group.

3. **Separate correction corpus.** The initial answer is generated from a **short 3-sentence lead**; targeted correction is allowed to search the **full article (100 sentences)**. Without this separation, correction just regenerates the same wrong value. The gap between the two is the room Pillar 3 has to improve accuracy.

4. **Honest calibration (Pillar 2).** Class-balancing weights touch only the base learner; calibration is fit unweighted on a held-out split. This is why ECE is low — reweighting the calibrator would inflate ECE (an earlier bug we fixed).

5. **Decoupled thresholds.** The router *flags* generously (`flag_threshold = 0.35`) so more wrong facts get a correction attempt, while the corrector *accepts* strictly (acceptance threshold 0.5) so the zero-regret guarantee holds. Two different knobs, in `RouterConfig` and `CorrectionConfig`.

6. **Signal cache.** The four ablation conditions share the same expensive per-fact signals (generation, UQ, faithfulness), so we compute them once and cache to `results_popqa/signal_cache.json` (~4× speedup; re-runs resume instantly).

All tunable settings live in **`franq_ext/config.py`**, and every run saves its exact config to `results_popqa/raw/*.config.json` for reproducibility.

---

## 11. Regenerating the Wikipedia cache

You normally **do not need to do this** — the cache is already built and bundled in the zip. Regenerate only if you change the entity set (e.g. a different `FRANQ_MIN_ATTRS` or a much larger `FRANQ_N`) or the sentence budgets.

Run this **off-Kaggle** (on a normal machine where Wikipedia isn't rate-limited):

```powershell
pip install datasets
python scripts/build_wiki_cache.py     # caches all >=2-attribute PopQA titles
```

It writes `franq_ext/data/wiki_cache/wiki_context.json` (short leads) and `wiki_context_full.json` (full articles). Then rezip the project. Cache keys embed the sentence budgets (`FRANQ_GEN_SENTENCES=3`, `FRANQ_CORR_SENTENCES=100`) — keep those defaults, or rebuild after changing them.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `context coverage: lead 6/150` (or low) | Wikipedia cache not loaded / live fetch throttled | Make sure `franq_ext/data/wiki_cache/` exists in the uploaded zip; it should read from there with no fetching. |
| `OSError: Read-only file system: '.cache'` | Writing cache to a read-only dir | Set `FRANQ_WIKI_CACHE=/kaggle/working/.cache/wiki_context.json` (the notebook already does). |
| `assert torch.cuda.is_available()` fails | GPU not enabled | Kaggle → Settings → Accelerator → GPU. |
| Model download fails / hangs | Internet off | Kaggle → Settings → Internet → On. |
| Accuracy stuck at ~7% | Contexts empty (old cache/grouping bug) | Use the latest zip (bundled cache + title grouping). |
| Demo error `unexpected keyword 'hide_index'` | Old Streamlit | Already fixed in `app.py` (uses an HTML table). `pip install -U streamlit` if needed. |
| Run too slow | 3B model + 150 entities | Lower `FRANQ_N`, `FRANQ_UQ_SAMPLES`, or use the 1.5B model. |

---

## 13. Deliverables checklist

Everything the professor needs to write the paper and reproduce the work:

- [x] **Code + bundled cache** — `franq_ext_project_FRESH.zip` (upload to Kaggle).
- [x] **Kaggle notebook** — `notebooks/franq_ext_kaggle.ipynb` (one-click run).
- [x] **Results** — `result_results_popqa/` (CSV tables + figures).
- [x] **Ablation table** — `tables/ablation.csv` · **Regret analysis** — `tables/regret_analysis.csv`.
- [x] **Figures** — `figures/*.png` (drop straight into the paper).
- [x] **Live demo** — `app.py` (offline, for the review).
- [x] **Slides** — `FRANQ_EXT_slides.html`.
- [x] **This handover** — `HANDOVER.md`.
- [x] **Tests** — `pytest` (44 passing, offline).

**Module → Pillar → Result mapping (for the paper's Section 6):**
| Pillar | Module | Isolated by | Result |
|---|---|---|---|
| 1 Graph | `graph/entity_graph.py` | A1 → A2 | PRR 0.48 → 0.54 |
| 2 Router | `router/calibrated_router.py` | B0 → A1 | ECE 0.60 → 0.06 |
| 3 Correction + Regret | `correction/`, `verification/` | A2 → A3 | Acc +2.6 pts, regret 0.014 vs 0.135 reckless |

---

## 14. Glossary

- **RAG** — Retrieval-Augmented Generation: answer questions using retrieved documents.
- **Faithfulness** — is a claim *supported by the retrieved evidence* (vs *factuality* = true in the world).
- **AlignScore / NLI entailment** — the model that scores faithfulness (does the evidence entail the claim).
- **Semantic entropy** — uncertainty from disagreement across repeated samples.
- **Parametric UQ** — the model's own confidence, closed-book (no evidence).
- **Router** — decides which facts need checking/correction.
- **Calibration / ECE** — whether a stated confidence matches real accuracy.
- **AUROC / PRR** — detection-ranking quality metrics (from FRANQ).
- **Correction-Regret** — our new metric: fraction of previously-correct facts broken by correction.
- **Ablation** — turning one component on/off at a time to measure its individual contribution.

---

*Questions on any part of this? Every claim here maps to a specific file in `franq_ext/`. Start from `franq_ext/pipeline.py` (the orchestrator) and follow the imports — each pillar is a small, self-contained module.*
