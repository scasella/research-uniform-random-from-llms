"""Self-contained metric helpers for the v0.4 lane.

Originally these lived in `rng_bias/metrics.py`, `rng_bias/bee.py`,
`rng_bias/bee_v0_2.py`, `rng_bias/bee_v0_3.py`, and `rng_bias/scoring.py`.
For the public release the v0.4 stack is the only supported lane, so the
math is consolidated here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chisquare


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.sqrt(n))
    for divisor in range(3, limit + 1, 2):
        if n % divisor == 0:
            return False
    return True


def _safe_probs(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-300, 1.0)
    return clipped / clipped.sum()


def tv_to_uniform(probs: np.ndarray) -> float:
    p = _safe_probs(probs)
    u = np.full_like(p, 1.0 / len(p))
    return float(0.5 * np.abs(p - u).sum())


def kl_to_uniform(probs: np.ndarray) -> float:
    p = _safe_probs(probs)
    u = np.full_like(p, 1.0 / len(p))
    return float(np.sum(p * (np.log(p) - np.log(u))))


def js_to_uniform(probs: np.ndarray) -> float:
    p = _safe_probs(probs)
    u = np.full_like(p, 1.0 / len(p))
    m = 0.5 * (p + u)
    return float(
        0.5 * np.sum(p * (np.log(p) - np.log(m)))
        + 0.5 * np.sum(u * (np.log(u) - np.log(m)))
    )


def normalized_entropy(probs: np.ndarray) -> float:
    p = _safe_probs(probs)
    entropy = -float(np.sum(p * np.log(p)))
    return entropy / math.log(len(p))


def min_entropy_from_probs(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    return -math.log2(max(float(probs.max()), 1e-300))


def normalize_logprobs(logprobs: np.ndarray) -> np.ndarray:
    max_logprob = np.max(logprobs)
    shifted = np.exp(logprobs - max_logprob)
    return shifted / shifted.sum()


def _safe_probs_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() <= 0:
        return np.full_like(counts, 1.0 / len(counts), dtype=np.float64)
    return counts / counts.sum()


def distribution_metrics_from_counts(
    counts: np.ndarray,
    *,
    n_outcomes: int,
    invalid_count: int = 0,
) -> dict[str, float | int]:
    counts = np.asarray(counts, dtype=np.float64)
    valid_count = int(counts.sum())
    total_count = valid_count + int(invalid_count)
    if valid_count > 0:
        probs = _safe_probs_from_counts(counts)
        expected = np.full(n_outcomes, valid_count / n_outcomes, dtype=np.float64)
        chi_stat, chi_p = chisquare(counts, expected)
        tv = tv_to_uniform(probs)
        kl = kl_to_uniform(probs)
        js = js_to_uniform(probs)
        entropy = normalized_entropy(probs)
        min_entropy = min_entropy_from_probs(probs)
        max_probability = float(probs.max())
    else:
        chi_stat, chi_p = np.nan, np.nan
        tv = np.nan
        kl = np.nan
        js = np.nan
        entropy = np.nan
        min_entropy = np.nan
        max_probability = np.nan
    return {
        "n": total_count,
        "valid_n": valid_count,
        "invalid_n": int(invalid_count),
        "invalid_rate": invalid_count / total_count if total_count else np.nan,
        "tv_uniform": tv,
        "kl_uniform": kl,
        "js_uniform": js,
        "chi_square": float(chi_stat),
        "chi_square_pvalue": float(chi_p),
        "normalized_entropy": entropy,
        "min_entropy_bits": min_entropy,
        "max_probability": max_probability,
    }


def _exact_counts_from_probs(probs: np.ndarray, *, total: int = 100_000) -> np.ndarray:
    """Convert a probability vector to integer counts that sum exactly to ``total``.

    ``number_distribution_metrics`` calls ``scipy.stats.chisquare``, which requires
    the observed and expected masses to agree to within ~1e-8 relative tolerance.
    Float rescaling alone leaves a sub-ULP residual that, after ``int(counts.sum())``
    truncation downstream, can produce a one-unit gap. Rounding to int and
    redistributing the rounding residual to the highest-probability cells gives
    a sum that is exactly ``total`` as a float.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if total <= 0 or probs.size == 0:
        return np.zeros_like(probs, dtype=np.float64)
    raw = probs * float(total)
    counts = np.rint(raw).astype(np.int64)
    diff = int(total) - int(counts.sum())
    if diff != 0:
        order = np.argsort(-raw)
        step = 1 if diff > 0 else -1
        for i in range(abs(diff)):
            counts[order[i % counts.size]] += step
    return counts.astype(np.float64)


def number_distribution_metrics(
    counts: np.ndarray,
    *,
    invalid_count: int = 0,
) -> dict[str, object]:
    """Rich integer-1-100 distribution metrics: TV/KL/JS plus per-decade and per-modulo masses."""
    import pandas as pd

    counts = np.asarray(counts, dtype=np.float64)
    base = distribution_metrics_from_counts(counts, n_outcomes=100, invalid_count=invalid_count)
    valid_n = max(float(counts.sum()), 1.0)
    probs = counts / valid_n
    expected = valid_n / 100.0
    ordered = pd.Series(counts, index=range(1, 101))
    top = ordered.sort_values(ascending=False).head(5)
    bottom = ordered.sort_values(ascending=True).head(5)
    metrics: dict[str, object] = {
        **base,
        "top5_numbers": ", ".join(str(int(value)) for value in top.index),
        "bottom5_numbers": ", ".join(str(int(value)) for value in bottom.index),
        "mass_ending_7": float(probs[[number % 10 == 7 for number in range(1, 101)]].sum()),
        "mass_multiples_10": float(probs[[number % 10 == 0 for number in range(1, 101)]].sum()),
        "mass_multiples_5": float(probs[[number % 5 == 0 for number in range(1, 101)]].sum()),
        "mass_odd": float(probs[[number % 2 == 1 for number in range(1, 101)]].sum()),
        "mass_even": float(probs[[number % 2 == 0 for number in range(1, 101)]].sum()),
        "mass_prime": float(probs[[is_prime(number) for number in range(1, 101)]].sum()),
        "mass_repeated_digits": float(
            probs[[number in {11, 22, 33, 44, 55, 66, 77, 88, 99} for number in range(1, 101)]].sum()
        ),
    }
    for number in [37, 42, 73, 69, 10, 50, 90]:
        metrics[f"obs_vs_expected_{number}"] = float(counts[number - 1] / max(expected, 1e-12))
    for start in range(1, 101, 10):
        end = min(start + 9, 100)
        metrics[f"mass_decade_{start}_{end}"] = float(probs[start - 1 : end].sum())
    return metrics


__all__ = [
    "_exact_counts_from_probs",
    "_safe_probs",
    "_safe_probs_from_counts",
    "distribution_metrics_from_counts",
    "is_prime",
    "js_to_uniform",
    "kl_to_uniform",
    "min_entropy_from_probs",
    "normalize_logprobs",
    "normalized_entropy",
    "number_distribution_metrics",
    "tv_to_uniform",
]
