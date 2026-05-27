"""GSM8K pass@k harness.

Verifier-grounded math benchmark. Extracts the final numerical answer from
both the ground truth (which ends with `#### N`) and the model's output
(prefer `#### N` pattern, fall back to the last number in the text), compares
with exact float equality (tolerance 1e-6 for safety).

GSM8K headroom for Qwen3-30B-A3B-Instruct at T=1.0 is genuine: a single sample
typically has 0.5-0.9 correctness, with within-condition variance that's
exactly the diversity signal we want to measure.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass

from rng_bias.backends.tinker_backend import TinkerBackend
from rng_bias.v0_4_2.humaneval import ConditionSpec, pass_at_k_unbiased


GSM8K_SYSTEM = (
    "You are a careful mathematician. Solve the problem step by step. "
    "After your reasoning, output the final numerical answer on a new line "
    "prefixed with `####`. Example: `#### 42`."
)


_HASH_ANS_RE = re.compile(r"####\s*([+-]?\d+(?:\.\d+)?)")
_NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def extract_answer(text: str) -> float | None:
    """Pull the final numerical answer. Prefer `#### N`, fall back to last number."""
    if not text:
        return None
    m = _HASH_ANS_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    nums = _NUM_RE.findall(text.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def check_correctness(model_text: str, gt_answer_text: str) -> tuple[bool, str | None]:
    gt = extract_answer(gt_answer_text)
    pred = extract_answer(model_text)
    if gt is None:
        return False, "gt_unparseable"
    if pred is None:
        return False, "pred_unparseable"
    return abs(gt - pred) < 1e-6, None if abs(gt - pred) < 1e-6 else "wrong_value"


async def _sample_one(sampling_client, tinker_mod, *, prompt_tokens, cond, max_new_tokens, seed):
    mi = tinker_mod.ModelInput.from_ints(prompt_tokens)
    params = tinker_mod.SamplingParams(
        max_tokens=int(max_new_tokens),
        temperature=float(cond.temperature),
        top_p=float(cond.top_p),
        top_k=int(cond.top_k) if int(cond.top_k) > 0 else 0,
        seed=int(seed),
    )
    fut = sampling_client.sample(prompt=mi, num_samples=1, sampling_params=params)
    r = fut.result() if hasattr(fut, "result") else fut
    return sampling_client.get_tokenizer().decode(r.sequences[0].tokens, skip_special_tokens=True)


async def sample_completions(
    backend: TinkerBackend,
    question: str,
    *,
    cond: ConditionSpec,
    k: int,
    max_new_tokens: int = 512,
    seed_base: int = 0,
    concurrency: int = 8,
) -> list[str]:
    import tinker

    rendered = backend.apply_chat_template(system=GSM8K_SYSTEM, user=question)
    prompt_tokens = backend._encode(rendered)
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int) -> str:
        async with sem:
            return await _sample_one(
                backend._sampling,
                tinker,
                prompt_tokens=prompt_tokens,
                cond=cond,
                max_new_tokens=max_new_tokens,
                seed=seed_base + i,
            )

    return await asyncio.gather(*[_one(i) for i in range(k)])


def score_completions(
    *, completions: list[str], gt_answer: str
) -> list[tuple[bool, str | None]]:
    return [check_correctness(c, gt_answer) for c in completions]


__all__ = [
    "GSM8K_SYSTEM",
    "extract_answer",
    "check_correctness",
    "sample_completions",
    "score_completions",
    "pass_at_k_unbiased",
]
