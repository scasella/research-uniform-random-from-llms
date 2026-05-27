"""Combine diversity + validity + judge into per-cell summaries.

The headline metric is

    useful_diversity = (# DBSCAN clusters with mean per-cluster judge-score >= 3) / n_items.

A secondary metric `quality_weighted_distinct` is computed for risk R2 — if
the trained model produces high cluster count but low cluster quality, this
exposes the trade-off without burying it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rng_bias.v0_4_1.clustering import ClusterResult
from rng_bias.v0_4_1.diversity_metrics import (
    distinct_n,
    exact_duplicate_rate,
    mean_pairwise_cosine_distance,
    near_duplicate_rate,
    self_bleu_proxy,
)


@dataclass(frozen=True)
class CellSummary:
    task_id: str
    scenario_id: str
    condition: str
    n_items: int
    n_valid: int
    valid_rate: float
    mean_pairwise_distance: float
    near_dup_rate: float
    distinct_2: float
    distinct_3: float
    exact_dup_rate: float
    self_bleu_proxy: float
    n_clusters: int
    n_noise: int
    eps_used: float
    mean_judge_score: float
    cluster_quality_pass_count: int
    useful_diversity: float
    quality_weighted_distinct: float
    joint_score: float
    parse_ok: bool
    notes: tuple[str, ...]


def _per_cluster_mean_quality(
    *, labels: np.ndarray, item_ids: list[str], judge_scores: dict[str, int]
) -> dict[int, float]:
    """Mean judge score per DBSCAN cluster (noise label -1 excluded)."""
    out: dict[int, list[int]] = {}
    for lab, iid in zip(labels.tolist(), item_ids, strict=True):
        if lab == -1:
            continue
        score = judge_scores.get(iid)
        if score is None:
            continue
        out.setdefault(int(lab), []).append(int(score))
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def summarize_cell(
    *,
    task_id: str,
    scenario_id: str,
    condition: str,
    items: list[str],
    item_ids: list[str],
    valid_flags: list[bool],
    embeddings: np.ndarray,
    cluster_result: ClusterResult,
    judge_scores: dict[str, int],
    quality_threshold: float = 3.0,
    parse_ok: bool = True,
) -> CellSummary:
    n_items = len(items)
    n_valid = sum(1 for f in valid_flags if f)
    valid_rate = n_valid / n_items if n_items else float("nan")

    mean_dist = mean_pairwise_cosine_distance(embeddings) if n_items else float("nan")
    near_dup = near_duplicate_rate(embeddings) if n_items else float("nan")
    d2 = distinct_n(items, 2) if n_items else float("nan")
    d3 = distinct_n(items, 3) if n_items else float("nan")
    exact_dup = exact_duplicate_rate(items) if n_items >= 2 else float("nan")
    bleu = self_bleu_proxy(items) if n_items >= 2 else float("nan")

    judge_score_values = [s for s in judge_scores.values()]
    mean_judge = float(np.mean(judge_score_values)) if judge_score_values else float("nan")

    cluster_mean_quality = _per_cluster_mean_quality(
        labels=cluster_result.labels,
        item_ids=item_ids,
        judge_scores=judge_scores,
    )
    cluster_pass_count = sum(1 for q in cluster_mean_quality.values() if q >= quality_threshold)
    useful_div = cluster_pass_count / n_items if n_items else float("nan")

    if judge_score_values and cluster_mean_quality:
        weighted = sum(min(q, 5.0) / 5.0 for q in cluster_mean_quality.values())
        quality_weighted = weighted / n_items if n_items else float("nan")
    else:
        quality_weighted = float("nan")

    # Joint score: smoother diversity x quality signal that doesn't depend on
    # DBSCAN's binary cluster/noise decision. Robust to embedding saturation
    # on tasks where every item sits far from every other in mpnet space.
    if not np.isnan(mean_dist) and judge_score_values:
        joint = float(mean_dist * (mean_judge / 5.0))
    else:
        joint = float("nan")

    return CellSummary(
        task_id=task_id,
        scenario_id=scenario_id,
        condition=condition,
        n_items=n_items,
        n_valid=n_valid,
        valid_rate=valid_rate,
        mean_pairwise_distance=mean_dist,
        near_dup_rate=near_dup,
        distinct_2=d2,
        distinct_3=d3,
        exact_dup_rate=exact_dup,
        self_bleu_proxy=bleu,
        n_clusters=cluster_result.n_clusters,
        n_noise=cluster_result.n_noise,
        eps_used=cluster_result.eps_used,
        mean_judge_score=mean_judge,
        cluster_quality_pass_count=cluster_pass_count,
        useful_diversity=useful_div,
        quality_weighted_distinct=quality_weighted,
        joint_score=joint,
        parse_ok=parse_ok,
        notes=cluster_result.notes,
    )


__all__ = ["CellSummary", "summarize_cell"]
