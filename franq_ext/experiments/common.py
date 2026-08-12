"""Shared plumbing for the experiment scripts: data, splits, running a condition,
and reading/writing result rows. Kept dependency-light (csv, not pandas) so the scripts
run on the laptop; the Colab notebook installs the heavy deps for real datasets.
"""
from __future__ import annotations

import csv
import os
import random

from franq_ext.config import Config
from franq_ext.schema import Example
from franq_ext.evaluation import evaluate_config, EvalResult
from franq_ext.train_router import train_router

RESULTS_DIR = os.environ.get("FRANQ_RESULTS", "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def ensure_dirs() -> None:
    for d in (RAW_DIR, TABLES_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)


def get_dataset(name: str = "sample", n: int | None = None, seed: int = 13) -> list[Example]:
    """Offline default = bundled sample. Real runs pass name='popqa' or 'triviaqa'."""
    if name == "sample":
        from franq_ext.data import load_sample
        return load_sample()
    if name == "popqa":
        from franq_ext.data import load_popqa
        return load_popqa(n=n or 800, seed=seed)
    if name == "popqa_structured":
        from franq_ext.data import load_popqa_structured
        min_attrs = int(os.environ.get("FRANQ_MIN_ATTRS", "2"))
        return load_popqa_structured(n=n or 300, seed=seed, min_attributes=min_attrs)
    if name == "triviaqa":
        from franq_ext.data import load_triviaqa
        return load_triviaqa(n=n or 800, seed=seed)
    raise ValueError(f"Unknown dataset: {name!r}")


def train_eval_split(examples: list[Example], train_frac: float = 0.5, seed: int = 13):
    """Split for fitting vs evaluating the learned router. Tiny sets (<8) are not split —
    train==eval — so the offline sample still runs (with a noted leakage caveat)."""
    if len(examples) < 8:
        return examples, examples
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    k = max(1, int(len(shuffled) * train_frac))
    return shuffled[:k], shuffled[k:]


def apply_backend_env(cfg: Config) -> Config:
    """Override model backends from environment variables so the SAME experiment scripts
    run offline (mock/lexical) on a laptop and with real models on Colab/Kaggle GPU.

    FRANQ_LLM_BACKEND=hf FRANQ_LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct
    FRANQ_SCORER_BACKEND=alignscore FRANQ_RETRIEVER_BACKEND=dense
    """
    cfg.llm.backend = os.environ.get("FRANQ_LLM_BACKEND", cfg.llm.backend)
    cfg.llm.model_name = os.environ.get("FRANQ_LLM_MODEL", cfg.llm.model_name)
    cfg.scoring.backend = os.environ.get("FRANQ_SCORER_BACKEND", cfg.scoring.backend)
    cfg.scoring.nli_model = os.environ.get("FRANQ_NLI_MODEL", cfg.scoring.nli_model)
    cfg.retriever.backend = os.environ.get("FRANQ_RETRIEVER_BACKEND", cfg.retriever.backend)
    device = os.environ.get("FRANQ_DEVICE")
    if device:
        cfg.llm.device = device
        cfg.scoring.device = device
    # Semantic entropy samples the LLM n_samples times PER FACT — the dominant cost on GPU.
    # Lower it (e.g. 2-3) for a faster first run.
    uq_samples = os.environ.get("FRANQ_UQ_SAMPLES")
    if uq_samples:
        cfg.uq.n_samples = int(uq_samples)
    return cfg


def _fact_source_factory():
    """Structured (RAG generate-then-check) mode when FRANQ_MODE=structured."""
    if os.environ.get("FRANQ_MODE") == "structured":
        from franq_ext.generation import structured_fact_source_factory
        return structured_fact_source_factory
    return None


def run_condition(name: str, label: str, cfg: Config, examples: list[Example],
                  seed: int = 13) -> EvalResult:
    from franq_ext.pipeline import Pipeline
    from franq_ext.evaluation.harness import evaluate_with_pipeline

    apply_backend_env(cfg)
    fsf = _fact_source_factory()
    train_ex, eval_ex = train_eval_split(examples, seed=seed)

    # One pipeline per condition => models (LLM, scorer, retriever) load once, not per split.
    pipeline = Pipeline(cfg, fact_source_factory=fsf)
    if cfg.router.kind == "learned":
        augment = 40 if len(train_ex) < 50 else 0  # jitter only tiny (offline) sets
        pipeline.router = train_router(cfg, train_ex, pipeline=pipeline, augment=augment)
    return evaluate_with_pipeline(pipeline, eval_ex, condition=label)


FIELDNAMES = [
    "short_name", "condition", "n_examples", "n_facts",
    "auroc", "prr", "ece",
    "factual_acc_before", "factual_acc_after",
    "correction_success", "correction_regret", "n_flagged", "n_corrected",
]


def save_row(short_name: str, res: EvalResult, path: str) -> None:
    ensure_dirs()
    row = {"short_name": short_name, **res.as_row()}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def append_master(short_name: str, res: EvalResult) -> None:
    ensure_dirs()
    path = os.path.join(RAW_DIR, "all_conditions.csv")
    exists = os.path.exists(path)
    row = {"short_name": short_name, **res.as_row()}
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def dataset_from_env() -> tuple[str, list[Example]]:
    name = os.environ.get("FRANQ_DATASET", "sample")
    n = os.environ.get("FRANQ_N")
    examples = get_dataset(name, n=int(n) if n else None)
    return name, examples
