"""Reward shaping for v0.4 GRPO training.

Pure functions, no Tinker dependency — so unit tests can pin the math without
provisioning a remote service. The training driver imports `compute_group_rewards`
and `compute_group_advantages` and assembles them into per-rollout loss weights.

Reward formula per rollout in a group of K rollouts:
    invalid (format failure) -> -invalid_penalty   (negative)
    valid value v            -> -log(hist[v] / K)  (range [0, log K], always >= 0)

Within-group advantage = reward minus group mean. Standard GRPO normalization
applies on top.

Design rationale: an earlier formulation subtracted `log(|S|)` from the valid
reward, which made valid rewards uniformly negative and *lower* than the
invalid penalty for any group with |S| > K. Training then learned to emit
unparseable garbage because invalid responses won the within-group reward
contest. Dropping the `-log(|S|)` term keeps valid rewards non-negative and
strictly better than invalid, so format compliance is preserved while the
within-group "favor underrepresented values" signal still drives uniformity.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from rng_bias.v0_4.distribution_tasks import DistributionTask


EPS = 1e-9


@dataclass(frozen=True)
class RewardConfig:
    # Must be > 0; valid rewards are in [0, log(group_size)] so any positive
    # penalty makes invalid strictly worse than any valid response.
    invalid_penalty: float = 2.0
    # Optional clip on the per-rollout reward magnitude before centering.
    reward_clip: float | None = 10.0
    # Standardize advantages (subtract mean, divide by std+EPS) within group.
    standardize_advantages: bool = True


def parse_group(texts: list[str], task: DistributionTask) -> list[str | None]:
    """Parse each rollout's text into its emitted value (or None if invalid)."""
    return [task.parser(text) for text in texts]


def group_histogram(parsed: list[str | None]) -> Counter:
    """Empirical histogram of parsed values across the group, ignoring None."""
    return Counter(value for value in parsed if value is not None)


def per_rollout_rewards(
    parsed: list[str | None],
    task: DistributionTask,
    config: RewardConfig = RewardConfig(),
) -> list[float]:
    """Compute the per-rollout reward array using the within-group histogram.

    Reward = -log(hist[v] / group_size) for valid v (in [0, log(group_size)]),
    -invalid_penalty for invalid. Valid is always >= 0; invalid is always < 0
    when invalid_penalty > 0. Within-group centering then preserves the strict
    valid > invalid ordering as the signed advantage.
    """
    hist = group_histogram(parsed)
    group_size = max(len(parsed), 1)
    rewards: list[float] = []
    for value in parsed:
        if value is None:
            reward = -float(config.invalid_penalty)
        else:
            count = max(hist.get(value, 0), 1)
            freq = count / group_size
            reward = -math.log(freq)
        if config.reward_clip is not None:
            clip = float(config.reward_clip)
            reward = max(-clip, min(clip, reward))
        rewards.append(float(reward))
    return rewards


def compute_group_rewards(
    texts: list[str],
    task: DistributionTask,
    config: RewardConfig = RewardConfig(),
) -> tuple[list[str | None], list[float]]:
    """Convenience wrapper: parse texts + compute rewards in one call."""
    parsed = parse_group(texts, task)
    return parsed, per_rollout_rewards(parsed, task, config)


def compute_group_advantages(
    rewards: list[float],
    config: RewardConfig = RewardConfig(),
) -> list[float]:
    """Center rewards within the group, optionally standardize by std."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    centered = [r - mean for r in rewards]
    if not config.standardize_advantages:
        return centered
    var = sum(c * c for c in centered) / max(len(centered), 1)
    std = math.sqrt(var + EPS)
    if std < EPS:
        return [0.0 for _ in centered]
    return [c / std for c in centered]


def group_uniformity_summary(
    parsed: list[str | None],
    task: DistributionTask,
) -> dict[str, float]:
    """Snapshot stats for a group (for logging during training)."""
    hist = group_histogram(parsed)
    group_size = max(len(parsed), 1)
    valid_n = sum(hist.values())
    invalid_rate = 1.0 - (valid_n / group_size)
    unique_values = len(hist)
    coverage = unique_values / max(len(task.candidates), 1)
    if valid_n > 0:
        empirical_uniform_mass = valid_n / len(task.candidates)
        empirical_tv = 0.5 * sum(
            abs(hist.get(c, 0) / valid_n - 1.0 / len(task.candidates))
            for c in task.candidates
        )
    else:
        empirical_tv = float("nan")
    return {
        "group_size": float(group_size),
        "valid_n": float(valid_n),
        "invalid_rate": float(invalid_rate),
        "unique_values": float(unique_values),
        "coverage": float(coverage),
        "empirical_tv_uniform": float(empirical_tv),
    }


__all__ = [
    "EPS",
    "RewardConfig",
    "parse_group",
    "group_histogram",
    "per_rollout_rewards",
    "compute_group_rewards",
    "compute_group_advantages",
    "group_uniformity_summary",
]
