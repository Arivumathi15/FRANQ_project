"""Run the whole experiment ladder in order and regenerate every table + figure.

    python -m franq_ext.experiments.run_all

Offline default uses the bundled sample dataset. For the real (GPU) runs set:
    FRANQ_DATASET=popqa   FRANQ_N=800   (and install the [full] extras + a real LLM cfg)
"""
from __future__ import annotations

import os

from franq_ext.experiments import common
from franq_ext.experiments import (
    exp01_baseline_franq,
    exp02_add_router,
    exp03_add_graph,
    exp04_add_correction,
    exp05_compute_regret,
    exp06_final_results,
)


def main() -> None:
    common.ensure_dirs()
    # Fresh master file so re-runs don't accumulate duplicates.
    master = os.path.join(common.RAW_DIR, "all_conditions.csv")
    if os.path.exists(master):
        os.remove(master)

    for step in (exp01_baseline_franq, exp02_add_router, exp03_add_graph,
                 exp04_add_correction, exp05_compute_regret, exp06_final_results):
        step.main()


if __name__ == "__main__":
    main()
