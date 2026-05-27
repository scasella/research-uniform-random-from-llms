"""DBSCAN clustering with per-task eps auto-fit + three documented fallbacks.

The auto-fit procedure: take the 4-NN cosine-distance curve, find the knee
(largest second-derivative point), round to 2 decimal places. Run only on
the baseline condition's combined outputs across all scenarios; the chosen
eps is then frozen for the comparison so trained/cad_rerank are scored
against the same partitioning rule.

Fallbacks (all logged in ClusterResult.notes):
- All-in-one cluster -> halve eps and retry once.
- All-noise -> double eps and retry once.
- DBSCAN produces 20 singleton clusters -> accept, but cross-check via
  _self_bleu_proxy in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    eps_used: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def auto_eps(embeddings: np.ndarray, *, k: int = 4, max_eps: float = 0.40) -> float:
    """Pick DBSCAN eps from the median k-NN cosine distance.

    Median is more stable than knee-finding on small (~20-100 point) samples
    where the 4-NN distance curve has no clear inflection. Cap at `max_eps`
    (default 0.40) so we don't end up with one all-encompassing cluster on
    topic-constrained tasks where every item sits in the same broad neighborhood.

    Returns a value rounded to two decimal places, clamped to [0.05, max_eps].
    """
    from sklearn.neighbors import NearestNeighbors

    n = embeddings.shape[0]
    if n <= k:
        return min(0.25, max_eps)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    kth = distances[:, k]
    eps = float(np.clip(round(float(np.median(kth)), 2), 0.05, max_eps))
    return eps


def cluster(embeddings: np.ndarray, *, eps: float, min_samples: int = 2) -> ClusterResult:
    """DBSCAN on cosine distances with three documented fallbacks."""
    from sklearn.cluster import DBSCAN

    notes: list[str] = []

    def _run(_eps: float) -> tuple[np.ndarray, int, int]:
        model = DBSCAN(eps=_eps, min_samples=min_samples, metric="cosine")
        lbl = model.fit_predict(embeddings)
        n_clust = int(len({int(x) for x in lbl if x != -1}))
        n_noise = int(np.sum(lbl == -1))
        return lbl, n_clust, n_noise

    labels, n_clusters, n_noise = _run(eps)
    eps_used = eps

    n_points = embeddings.shape[0]
    if n_clusters == 1 and n_noise == 0:
        new_eps = max(0.02, round(eps * 0.5, 3))
        notes.append(f"all-in-one-cluster fallback: eps {eps:.2f} -> {new_eps:.2f}")
        labels, n_clusters, n_noise = _run(new_eps)
        eps_used = new_eps
    elif n_clusters == 0 and n_noise == n_points:
        new_eps = min(0.99, round(eps * 2.0, 3))
        notes.append(f"all-noise fallback: eps {eps:.2f} -> {new_eps:.2f}")
        labels, n_clusters, n_noise = _run(new_eps)
        eps_used = new_eps

    if n_clusters == n_points and n_points > 0:
        notes.append(
            f"all-singleton clusters (n={n_points}); cross-check via _self_bleu_proxy in report"
        )

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        eps_used=eps_used,
        notes=tuple(notes),
    )


__all__ = ["ClusterResult", "auto_eps", "cluster"]
