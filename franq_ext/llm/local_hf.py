"""Real Hugging Face backend for the GPU experiment runs (Colab/Kaggle T4).

Not exercised by the offline test suite (it needs torch + transformers + a download).
Kept deliberately small and readable so it is easy to audit against the paper.
"""
from __future__ import annotations

import hashlib
import json
import re

from franq_ext.llm.client import LLMClient
from franq_ext.schema import Fact


_EXTRACT_PROMPT = (
    "Extract factual claims from the answer as JSON: a list of objects with keys "
    '"entity", "attribute", "value". Only output JSON.\n'
    "Question: {question}\nAnswer: {answer}\nJSON:"
)


def _resolve_device(requested: str, torch) -> str:
    """Fall back to CPU if a GPU was requested but none is available (no crash)."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[franq_ext] CUDA not available -> running on CPU (slow). "
              "Enable a GPU runtime for real experiments.", flush=True)
        return "cpu"
    return requested


def _stable_seed(prompt: str, i: int) -> int:
    return int(hashlib.md5(f"{i}:{prompt}".encode("utf-8")).hexdigest()[:8], 16)


class LocalHFLLM(LLMClient):
    def __init__(self, cfg) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        # Resolve the device: if 'cuda' was requested but no GPU is present, fall back to CPU
        # instead of crashing (fp16 only on GPU; CPU needs fp32).
        self.device = _resolve_device(cfg.device, torch)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ helpers
    def _chat(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return text

    def _generate(self, prompt: str, do_sample: bool, temperature: float) -> str:
        inputs = self.tokenizer(self._chat(prompt), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ interface
    def complete(self, prompt: str, temperature: float | None = None) -> str:
        return self._generate(prompt, do_sample=False, temperature=temperature or 0.0)

    def sample(self, prompt: str, n: int) -> list[str]:
        # Seed each sample so semantic entropy is REPRODUCIBLE across runs and conditions
        # (still diverse: different i -> different seed -> different sample). Without this,
        # the ablation differences were contaminated by run-to-run sampling noise.
        outs = []
        for i in range(n):
            seed = _stable_seed(prompt, i)
            self.torch.manual_seed(seed)
            if self.device == "cuda":
                self.torch.cuda.manual_seed_all(seed)
            outs.append(self._generate(prompt, True, self.cfg.temperature))
        return outs

    def sequence_confidence(self, context: str, statement: str) -> float:
        """Mean token probability of `statement` conditioned on `context`, in [0,1]."""
        import math

        full = self.tokenizer(context + " " + statement, return_tensors="pt").to(self.device)
        ctx = self.tokenizer(context, return_tensors="pt")
        ctx_len = ctx["input_ids"].shape[1]
        with self.torch.no_grad():
            logits = self.model(**full).logits
        logprobs = self.torch.log_softmax(logits[0, :-1], dim=-1)
        target = full["input_ids"][0, 1:]
        stmt_lp = logprobs[ctx_len - 1:, :].gather(
            1, target[ctx_len - 1:].unsqueeze(1)
        ).squeeze(1)
        if stmt_lp.numel() == 0:
            return 0.5
        return float(math.exp(stmt_lp.mean().item()))

    def extract_facts(self, answer: str, question: str = "") -> list[Fact]:
        raw = self.complete(_EXTRACT_PROMPT.format(question=question, answer=answer))
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        facts = []
        for it in items:
            if all(k in it for k in ("entity", "attribute", "value")):
                facts.append(Fact(str(it["entity"]), str(it["attribute"]), str(it["value"])))
        return facts

    def answer_attribute(self, entity: str, attribute: str, evidence: str) -> str:
        prompt = (
            f"Answer with ONLY the {attribute} of {entity} — a name or short phrase, no "
            f"sentence, no explanation. If the evidence does not say, give your best guess.\n"
            f"Evidence: {evidence}\n"
            f"{attribute} of {entity}:"
        )
        raw = self.complete(prompt).strip()
        return self._short_value(raw, entity, attribute)

    @staticmethod
    def _short_value(raw: str, entity: str, attribute: str) -> str:
        """Reduce a possibly-verbose completion to a clean short value."""
        import re

        value = raw.splitlines()[0].strip() if raw else ""
        # Strip a leading "The <attribute> of <entity> is ..." preamble if present.
        m = re.search(r"\bis\b[:\s]+(.*)", value, flags=re.IGNORECASE)
        if m and value.lower().startswith(("the ", attribute.lower(), entity.lower())):
            value = m.group(1)
        return value.strip().strip(".").strip('"').strip()
