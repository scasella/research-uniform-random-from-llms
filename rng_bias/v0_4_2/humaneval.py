"""HumanEval pass@k harness.

Minimal verifier-grounded benchmark. Loads `openai_humaneval` from HF, samples k
completions per problem, executes each completion's function against the
problem's test cases in a subprocess with a timeout, computes pass@k via the
unbiased Codex estimator.

The Tinker prompt format wraps the bare function signature in a chat template:
System asks for the body in a fenced Python block; we extract and verify.
"""
from __future__ import annotations

import ast
import asyncio
import math
import multiprocessing
import re
import textwrap
import time
from dataclasses import dataclass

import numpy as np

from rng_bias.backends.tinker_backend import TinkerBackend


HUMANEVAL_SYSTEM = textwrap.dedent(
    """\
    You are an expert Python programmer. Complete the following function.
    Output ONLY the function definition (the signature and body), wrapped in a
    single ```python ... ``` fence. Do not include explanations, imports, tests,
    or any other code.
    """
).strip()


_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


def extract_python(text: str) -> str:
    """Extract the Python code block from the model's response.

    Falls back to returning the raw text if no fence is found.
    """
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[0].strip()
    return text.strip()


def _runner(prompt: str, completion: str, test: str, entry_point: str, q):
    """Subprocess runner: exec the program and report pass/fail via Queue."""
    try:
        program = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
        # Sandbox: minimal globals; no builtins denylist for simplicity at P0
        exec_globals: dict = {}
        exec(program, exec_globals)
        q.put(("ok", None))
    except BaseException as e:  # noqa: BLE001
        q.put(("err", f"{type(e).__name__}: {e!s}"[:200]))


def check_correctness(
    prompt: str, completion: str, test: str, entry_point: str, *, timeout: float = 6.0
) -> tuple[bool, str | None]:
    """Run the completion against the test in a fresh process with a hard timeout.

    Returns (passed, reason). On timeout or any exception, returns False with a tag.
    """
    if not completion.strip():
        return False, "empty"
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_runner, args=(prompt, completion, test, entry_point, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
        return False, "timeout"
    if q.empty():
        return False, "no_result"
    status, reason = q.get()
    return status == "ok", reason


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Codex-paper unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    model_path: str | None  # None = vanilla baseline
    temperature: float
    top_p: float = 1.0
    top_k: int = 0


async def _sample_one(
    sampling_client,
    tinker_mod,
    *,
    prompt_tokens: list[int],
    cond: ConditionSpec,
    max_new_tokens: int,
    seed: int,
) -> str:
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
    he_prompt: str,
    *,
    cond: ConditionSpec,
    k: int,
    max_new_tokens: int = 512,
    seed_base: int = 0,
    concurrency: int = 8,
) -> list[str]:
    """Sample k completions of a HumanEval prompt under condition cond."""
    import tinker

    rendered = backend.apply_chat_template(system=HUMANEVAL_SYSTEM, user=he_prompt)
    prompt_tokens = backend._encode(rendered)
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int) -> str:
        async with sem:
            text = await _sample_one(
                backend._sampling,
                tinker,
                prompt_tokens=prompt_tokens,
                cond=cond,
                max_new_tokens=max_new_tokens,
                seed=seed_base + i,
            )
            return extract_python(text)

    completions = await asyncio.gather(*[_one(i) for i in range(k)])
    return completions


def score_completions(
    *, prompt: str, completions: list[str], test: str, entry_point: str
) -> list[tuple[bool, str | None]]:
    """Sequentially verify each completion. (Subprocess spawn + exec; not parallelized.)"""
    return [check_correctness(prompt, c, test, entry_point) for c in completions]


__all__ = [
    "HUMANEVAL_SYSTEM",
    "ConditionSpec",
    "extract_python",
    "check_correctness",
    "pass_at_k_unbiased",
    "sample_completions",
    "score_completions",
]
