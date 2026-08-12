# Local CPU run of the full ablation ladder (no GPU needed).
# Usage:  powershell -ExecutionPolicy Bypass -File run_local.ps1
# Edit the values below to trade speed vs quality.

$env:FRANQ_MODE          = "structured"
$env:FRANQ_DATASET       = "popqa_structured"
$env:FRANQ_N             = "15"        # entities. Start small (15-20) to sanity-check; raise later.
$env:FRANQ_MIN_ATTRS     = "3"         # keep only entities with >=3 attributes (so the graph matters)
$env:FRANQ_UQ_SAMPLES    = "2"         # semantic-entropy samples/fact (fewer = faster)

$env:FRANQ_LLM_BACKEND   = "hf"
# 0.5B is fastest on CPU (~17s/fact); 1.5B is better quality but ~2-3x slower.
$env:FRANQ_LLM_MODEL     = "Qwen/Qwen2.5-0.5B-Instruct"
$env:FRANQ_SCORER_BACKEND= "nli"
$env:FRANQ_RETRIEVER_BACKEND = "dense" # dense works without faiss (numpy fallback)
$env:FRANQ_DEVICE        = "cpu"

$env:FRANQ_PROGRESS_EVERY= "3"
$env:FRANQ_RESULTS       = "results_popqa"

Write-Host "Running franq_ext ablation locally on CPU (this is slow; first condition does the heavy work, the rest reuse the cache)..."
python -m franq_ext.experiments.run_all

Write-Host ""
Write-Host "Done. Results in $($env:FRANQ_RESULTS)\tables\ablation.csv and \figures\*.png"
