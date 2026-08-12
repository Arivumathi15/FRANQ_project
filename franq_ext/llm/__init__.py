"""LLM backends behind a single interface (mock for offline dev, HF for real runs)."""
from franq_ext.llm.client import LLMClient
from franq_ext.llm.mock_llm import MockLLM


_LLM_CACHE: dict = {}


def build_llm(cfg) -> LLMClient:
    """Factory: pick a backend from the LLM config.

    Cached by (backend, model, device) so a heavy HF model loads ONCE per process instead
    of once per condition (4 conditions x reload was a major slowdown on the GPU runs).
    """
    key = (cfg.backend, cfg.model_name, cfg.device)
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    if cfg.backend == "mock":
        llm: LLMClient = MockLLM()
    elif cfg.backend == "hf":
        from franq_ext.llm.local_hf import LocalHFLLM  # lazy: torch only for real runs
        llm = LocalHFLLM(cfg)
    else:
        raise ValueError(f"Unknown LLM backend: {cfg.backend!r}")
    _LLM_CACHE[key] = llm
    return llm


__all__ = ["LLMClient", "MockLLM", "build_llm"]
