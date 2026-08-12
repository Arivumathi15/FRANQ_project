"""Figures for the paper. matplotlib is optional — if it is not installed (pure offline
laptop), the scripts still produce all CSV tables and just skip the PNGs.
"""
from __future__ import annotations

import os


def _have_mpl() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def plot_ablation(rows: list[dict], figures_dir: str) -> list[str]:
    """rows: ordered ladder rows (dicts). Returns list of written figure paths."""
    if not _have_mpl():
        print("     (matplotlib not installed -> skipping figures; CSVs still written)")
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)
    labels = [r["short_name"] for r in rows]
    written = []

    def _bar(metric: str, title: str, fname: str, pct: bool = False):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[metric]))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals, color="#3b6ea5")
        ax.set_title(title)
        ax.set_ylabel(metric)
        if pct:
            ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals):
            if v == v:  # not nan
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        path = os.path.join(figures_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    _bar("auroc", "Detection quality (AUROC) across ablation", "fig_auroc.png", pct=True)
    _bar("prr", "Prediction Rejection Ratio (PRR) across ablation", "fig_prr.png")
    _bar("ece", "Calibration error (ECE, lower is better)", "fig_ece.png", pct=True)
    _bar("correction_regret", "Correction regret (lower is better)", "fig_regret.png", pct=True)

    # Factual accuracy before vs after (grouped bars).
    import numpy as np
    before = [float(r["factual_acc_before"]) for r in rows]
    after = [float(r["factual_acc_after"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, before, 0.4, label="before", color="#b5651d")
    ax.bar(x + 0.2, after, 0.4, label="after", color="#3b6ea5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Factual accuracy before vs after correction")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(figures_dir, "fig_factual_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written
