"""Stage-3 baselines for BEE v0.4.

Two embarrassing-ceiling controls against which the GRPO result is judged:

1. **Logit-correction** — post-hoc, no training. For each task, take the
   Instruct model's candidate distribution P_instruct and apply a per-candidate
   bias subtraction so the corrected distribution is closer to uniform. The
   bias is log P_instruct(c) - log(1/|S|). This is parameter-free and serves
   as the lower-effort upper-bound on what's achievable without RL.

2. **KL-distill-from-Base** — minimal SFT-style training that distills the
   Base model's candidate distribution onto Instruct on the training tasks
   only. Maximally surgical, minimal expected transfer. If GRPO doesn't beat
   this on held-out transfer tasks, GRPO didn't find a shared subspace.
   Implemented as a Stage-3 deferred piece — see `kl_distill_from_base`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rng_bias.backends import Backend
from rng_bias.v0_4.distribution_tasks import DistributionTask, TASKS
from rng_bias.v0_4.eval_lane_a import (
    HEADLINE_SURFACE,
    TaskEvalConfig,
    _task_metrics_from_probs,
    lane_a_distribution_for_task,
)


@dataclass(frozen=True)
class LogitCorrectionResult:
    task_id: str
    original_probs: np.ndarray
    corrected_probs: np.ndarray
    bias_vector: np.ndarray
    original_metrics: dict[str, object]
    corrected_metrics: dict[str, object]


def logit_correction_for_task(
    backend: Backend,
    task: DistributionTask,
    *,
    eval_config: TaskEvalConfig = TaskEvalConfig(),
) -> LogitCorrectionResult:
    """Compute the logit-correction baseline for one task using a single backend.

    The bias vector is computed from the backend's own candidate-scored
    distribution and subtracted from the same distribution. This produces the
    "perfect post-hoc flattening" — the upper bound that no parameter-update
    method can beat on the same data without overfitting.
    """
    surface_probs = lane_a_distribution_for_task(backend, task, config=eval_config)
    probs = np.asarray(surface_probs[HEADLINE_SURFACE], dtype=np.float64)
    if probs.sum() <= 0:
        probs = np.full_like(probs, 1.0 / max(len(probs), 1))
    n = len(task.candidates)
    uniform_prob = 1.0 / max(n, 1)
    safe = np.maximum(probs, 1e-12)
    bias_vector = np.log(safe) - np.log(uniform_prob)
    # Corrected logprobs = original - bias; softmax to get a distribution.
    corrected_logits = np.log(safe) - bias_vector  # collapses to log(uniform_prob)
    corrected = np.exp(corrected_logits - corrected_logits.max())
    corrected = corrected / corrected.sum()
    return LogitCorrectionResult(
        task_id=task.task_id,
        original_probs=probs,
        corrected_probs=corrected,
        bias_vector=bias_vector,
        original_metrics=_task_metrics_from_probs(probs, task),
        corrected_metrics=_task_metrics_from_probs(corrected, task),
    )


def logit_correction_panel(
    backend: Backend,
    *,
    tasks: tuple[DistributionTask, ...] = TASKS,
    eval_config: TaskEvalConfig = TaskEvalConfig(),
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run logit-correction across a panel of tasks and persist a CSV row each."""
    rows: list[dict[str, object]] = []
    for task in tasks:
        result = logit_correction_for_task(backend, task, eval_config=eval_config)
        rows.append(
            {
                "task_id": task.task_id,
                "split": task.split,
                "model_id": backend.model_id,
                "pair_id": backend.pair_id,
                "model_status": backend.model_status,
                "backend_id": backend.backend_id,
                "original_tv_uniform": float(result.original_metrics.get("tv_uniform", float("nan"))),
                "corrected_tv_uniform": float(result.corrected_metrics.get("tv_uniform", float("nan"))),
                "original_kl_uniform": float(result.original_metrics.get("kl_uniform", float("nan"))),
                "corrected_kl_uniform": float(result.corrected_metrics.get("kl_uniform", float("nan"))),
                "n_candidates": int(len(task.candidates)),
            }
        )
    df = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        metrics_dir = output_dir / "data/bee_v0_4/metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(metrics_dir / "bee_v0_4_logit_correction.csv", index=False)
    return df


def kl_distill_from_base(*args: object, **kwargs: object) -> dict[str, object]:
    """Stage-3 deferred: distill Base candidate distribution onto Instruct via SFT.

    Implementation strategy when this is wired up:
    - For each training task, gather Base's candidate-scored distribution.
    - Pre-tokenize the prompt + first-token candidate sets.
    - Use `tinker.TrainingClient.forward_backward_custom` with a custom KL loss
      against the Base distribution as soft target. The Tinker SDK doesn't
      ship a built-in KL-to-distribution loss; this requires writing the loss
      in Python and shipping it via the custom-loss interface.
    - Save the resulting checkpoint and evaluate via the same v0.4 Lane-A
      pipeline as the GRPO checkpoint.
    """
    raise NotImplementedError(
        "kl_distill_from_base is a Stage-3 deferred piece. "
        "Implement after GRPO baseline lands, alongside the Stage-3 driver."
    )


__all__ = [
    "LogitCorrectionResult",
    "logit_correction_for_task",
    "logit_correction_panel",
    "kl_distill_from_base",
]
