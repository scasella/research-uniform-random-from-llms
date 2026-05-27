"""Lane-A candidate-scoring evaluation helpers for v0.4.

Thin wrapper over the v0.3 candidate-scoring kernel and bee_v0_3 metric helpers,
parameterized over the v0.4 distribution-task suite. Used by:

- diagnostic.py  (Phase 0 cross-task base/Instruct gap)
- stage CLIs     (post-training TV measurement against trained checkpoints)
- baselines.py   (logit-correction baseline scoring)
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from rng_bias.backends import Backend
from rng_bias.v0_4._metrics import (
    _exact_counts_from_probs,
    normalize_logprobs,
    number_distribution_metrics,
)
from rng_bias.v0_4.distribution_tasks import DistributionTask, TASKS


HEADLINE_SURFACE = "leading_space"
SURFACE_TEMPLATES: dict[str, str] = {
    "plain": "{candidate}",
    "leading_space": " {candidate}",
    "leading_newline": "\n{candidate}",
    "trailing_newline": "{candidate}\n",
}


@dataclass(frozen=True)
class TaskEvalConfig:
    paraphrase_count: int = 3
    surfaces: tuple[str, ...] = (HEADLINE_SURFACE,)
    include_logsumexp: bool = False


def _generic_distribution_metrics(probs: np.ndarray, candidate_size: int) -> dict[str, object]:
    """TV / KL / JS / entropy for an arbitrary candidate distribution."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    n = max(candidate_size, 1)
    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    tv = float(0.5 * np.abs(probs - uniform).sum())
    safe = np.maximum(probs, 1e-12)
    kl = float((probs * (np.log(safe) - np.log(uniform))).sum())
    mid = 0.5 * (probs + uniform)
    mid_safe = np.maximum(mid, 1e-12)
    js = 0.5 * float((probs * (np.log(safe) - np.log(mid_safe))).sum())
    js += 0.5 * float((uniform * (np.log(uniform) - np.log(mid_safe))).sum())
    entropy = -float((probs * np.log(safe)).sum())
    max_entropy = math.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return {
        "n_candidates": int(candidate_size),
        "tv_uniform": tv,
        "kl_uniform": kl,
        "js_uniform": js,
        "normalized_entropy": normalized_entropy,
        "max_probability": float(probs.max()) if probs.size else float("nan"),
    }


def _task_metrics_from_probs(probs: np.ndarray, task: DistributionTask) -> dict[str, object]:
    """Use v0.3 dense schema for integer-1-100 (compat), generic for others."""
    if task.task_id == "random_int_1_100":
        counts = _exact_counts_from_probs(probs, total=10_000)
        return number_distribution_metrics(counts, invalid_count=0)
    return _generic_distribution_metrics(probs, len(task.candidates))


_PARAPHRASES: tuple[str, ...] = (
    "Pick a random {item_kind}.",
    "Produce one random {item_kind}.",
    "Sample uniformly at random — output one {item_kind}.",
    "Choose a {item_kind} at random and output only it.",
    "Random {item_kind}, no commentary.",
    "Output one {item_kind} chosen uniformly.",
    "Generate a random {item_kind}.",
    "Roll once. Output the {item_kind}.",
)


def _render_prompt(task: DistributionTask, paraphrase_idx: int) -> str:
    user_phrase = _PARAPHRASES[paraphrase_idx % len(_PARAPHRASES)].format(item_kind=task.item_kind)
    return (
        f"System: You are a uniform random sampler. Output ONLY one {task.item_kind} "
        f"from the list below, exactly as written. No punctuation, no extra words.\n\n"
        f"Valid {task.item_kind}s: {task.candidate_csv}\n"
        f"User: {user_phrase}\n"
        f"Assistant:"
    )


def lane_a_distribution_for_task(
    backend: Backend,
    task: DistributionTask,
    *,
    config: TaskEvalConfig = TaskEvalConfig(),
) -> dict[str, np.ndarray]:
    """Return {surface_mode: probs_over_candidates_array} averaged across paraphrases."""
    surface_names = list(config.surfaces)
    candidate_labels = list(task.candidates)
    candidate_surfaces = [
        [SURFACE_TEMPLATES[name].format(candidate=label) for name in surface_names]
        for label in candidate_labels
    ]
    flat_candidates = [s for row in candidate_surfaces for s in row]
    paraphrase_probs: dict[str, list[np.ndarray]] = {name: [] for name in surface_names}
    if config.include_logsumexp and len(surface_names) > 1:
        paraphrase_probs["logsumexp"] = []
    for paraphrase_idx in range(max(1, config.paraphrase_count)):
        prompt = _render_prompt(task, paraphrase_idx)
        flat_logprobs = backend.candidate_logprobs(prompt=prompt, candidates=flat_candidates)
        reshaped = np.asarray(flat_logprobs, dtype=np.float64).reshape(len(candidate_labels), len(surface_names))
        for surface_index, surface_name in enumerate(surface_names):
            logprobs = reshaped[:, surface_index]
            paraphrase_probs[surface_name].append(normalize_logprobs(logprobs))
        if "logsumexp" in paraphrase_probs:
            agg = logsumexp(reshaped, axis=1)
            paraphrase_probs["logsumexp"].append(normalize_logprobs(agg))
    return {name: np.mean(np.stack(arrs, axis=0), axis=0) for name, arrs in paraphrase_probs.items()}


def lane_a_metrics_row(
    backend: Backend,
    task: DistributionTask,
    *,
    rendering: str = "exmergo_flat",
    config: TaskEvalConfig = TaskEvalConfig(),
) -> dict[str, object]:
    """Convenience: compute metrics for a single (backend, task) on the headline surface."""
    surface_probs = lane_a_distribution_for_task(backend, task, config=config)
    probs = surface_probs[HEADLINE_SURFACE]
    metrics = _task_metrics_from_probs(probs, task)
    counter = Counter()
    for label, prob in zip(task.candidates, probs):
        counter[label] = float(prob)
    top_pairs = counter.most_common(5)
    return {
        "model_id": backend.model_id,
        "pair_id": backend.pair_id,
        "model_status": backend.model_status,
        "backend_id": backend.backend_id,
        "task_id": task.task_id,
        "split": task.split,
        "rendering": rendering,
        "surface_mode": HEADLINE_SURFACE,
        "paraphrase_count": int(config.paraphrase_count),
        **metrics,
        "top5_values": ", ".join(label for label, _ in top_pairs),
        "top5_probabilities": ", ".join(f"{prob:.4f}" for _, prob in top_pairs),
    }


def lane_a_all_tasks(
    backend: Backend,
    *,
    tasks: tuple[DistributionTask, ...] = TASKS,
    config: TaskEvalConfig = TaskEvalConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in tasks:
        rows.append(lane_a_metrics_row(backend, task, config=config))
    return pd.DataFrame(rows)


__all__ = [
    "HEADLINE_SURFACE",
    "SURFACE_TEMPLATES",
    "TaskEvalConfig",
    "lane_a_distribution_for_task",
    "lane_a_metrics_row",
    "lane_a_all_tasks",
]
