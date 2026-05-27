"""Sampling-based empirical-TV evaluation for v0.4 distribution tasks.

Lane A in v0.4 measures TV-to-uniform from candidate logprobs — temperature-
independent because temperature scales logits uniformly. This module measures
TV-to-uniform from **actual samples** at a chosen temperature, which is what a
deployment-style measurement would see.

Used by the temperature ablation driver to answer: does the v0.4 trained
checkpoint's Lane A improvement also show up under sampling, and does vanilla
at higher temperature reach the same empirical TV?
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rng_bias.backends.tinker_backend import TinkerBackend
from rng_bias.v0_4.distribution_tasks import DistributionTask


@dataclass(frozen=True)
class SampleEvalConfig:
    n_samples: int = 150
    paraphrase_count: int = 3
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 16
    batch_size: int = 16
    concurrency: int = 4


_PARAPHRASES: tuple[str, ...] = (
    "Pick a random {item_kind}.",
    "Produce one random {item_kind}.",
    "Choose a {item_kind} at random and output only it.",
)


def _render_system(task: DistributionTask) -> str:
    return (
        f"You are a uniform random sampler. Output ONLY one {task.item_kind} "
        f"from the list below, exactly as written. No punctuation, no extra words.\n"
        f"Valid {task.item_kind}s: {task.candidate_csv}"
    )


def _render_user(task: DistributionTask, paraphrase_idx: int) -> str:
    return _PARAPHRASES[paraphrase_idx % len(_PARAPHRASES)].format(item_kind=task.item_kind)


async def _sample_one(
    sampling_client: Any,
    tinker_mod: Any,
    *,
    prompt_tokens: list[int],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
) -> str:
    mi = tinker_mod.ModelInput.from_ints(prompt_tokens)
    params = tinker_mod.SamplingParams(
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k) if int(top_k) > 0 else 0,
        seed=int(seed),
    )
    future = sampling_client.sample(prompt=mi, num_samples=1, sampling_params=params)
    response = future.result() if hasattr(future, "result") else future
    seq = response.sequences[0]
    return sampling_client.get_tokenizer().decode(seq.tokens, skip_special_tokens=True)


async def empirical_distribution_for_task(
    backend: TinkerBackend,
    task: DistributionTask,
    *,
    config: SampleEvalConfig = SampleEvalConfig(),
    seed_base: int = 0,
) -> dict[str, object]:
    """Sample n_samples times, parse to candidates, return empirical distribution.

    Splits samples across `paraphrase_count` paraphrases of the user instruction.
    Concurrency `config.concurrency` futures in flight via asyncio.gather.
    """
    import tinker

    sampling_client = backend._sampling
    counts: dict[str, int] = {c: 0 for c in task.candidates}
    invalid = 0
    raw_texts: list[str] = []

    per_para = max(1, config.n_samples // max(1, config.paraphrase_count))
    samples_used = 0
    start = time.time()
    semaphore = asyncio.Semaphore(config.concurrency)

    async def _one(prompt_tokens: list[int], seed: int) -> str:
        async with semaphore:
            return await _sample_one(
                sampling_client,
                tinker,
                prompt_tokens=prompt_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_new_tokens,
                seed=seed,
            )

    for paraphrase_idx in range(max(1, config.paraphrase_count)):
        system = _render_system(task)
        user = _render_user(task, paraphrase_idx)
        prompt = backend.apply_chat_template(system=system, user=user)
        prompt_tokens = backend._encode(prompt)
        seeds = [seed_base + paraphrase_idx * 100_000 + i for i in range(per_para)]
        texts = await asyncio.gather(*[_one(prompt_tokens, s) for s in seeds])
        raw_texts.extend(texts)
        for t in texts:
            parsed = task.parser(t)
            if parsed is None or parsed not in counts:
                invalid += 1
            else:
                counts[parsed] += 1
            samples_used += 1

    elapsed = time.time() - start

    n = len(task.candidates)
    total_valid = sum(counts.values())
    if total_valid > 0:
        uniform = 1.0 / n
        empirical = {c: counts[c] / total_valid for c in task.candidates}
        tv_uniform = 0.5 * sum(abs(empirical[c] - uniform) for c in task.candidates)
        max_prob = max(empirical.values())
        # normalized entropy
        probs = np.array(list(empirical.values()), dtype=np.float64)
        safe = np.maximum(probs, 1e-12)
        entropy = -float((probs * np.log(safe)).sum())
        max_entropy = math.log(n) if n > 1 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        # top-5 candidates by frequency
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top5_values = ", ".join(label for label, _ in sorted_items[:5])
        top5_probs = ", ".join(f"{counts[label] / total_valid:.4f}" for label, _ in sorted_items[:5])
    else:
        tv_uniform = float("nan")
        max_prob = float("nan")
        norm_entropy = float("nan")
        top5_values = ""
        top5_probs = ""

    return {
        "task_id": task.task_id,
        "split": task.split,
        "n_candidates": n,
        "n_samples_total": samples_used,
        "n_valid": total_valid,
        "n_invalid": invalid,
        "valid_rate": total_valid / samples_used if samples_used else float("nan"),
        "tv_uniform": tv_uniform,
        "max_probability": max_prob,
        "normalized_entropy": norm_entropy,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "paraphrase_count": config.paraphrase_count,
        "top5_values": top5_values,
        "top5_probabilities": top5_probs,
        "elapsed_seconds": elapsed,
    }


__all__ = ["SampleEvalConfig", "empirical_distribution_for_task"]
