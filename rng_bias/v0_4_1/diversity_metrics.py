"""Diversity metrics on a 20-generation cell.

All functions take either the raw generation texts or their embeddings; we
keep the surface small so the scoring stage can call them in any order.

self_bleu_proxy is re-exported from v0_4/capability_eval.py to avoid drift.
"""
from __future__ import annotations

import re

import numpy as np

from rng_bias.v0_4.capability_eval import _ngram_set, _self_bleu_proxy

self_bleu_proxy = _self_bleu_proxy


def mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance over a (n, d) embedding matrix.

    Embeddings are expected normalized; we still defensively normalize.
    Returns nan on n < 2.
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    u = embeddings / norm
    sim = u @ u.T
    iu = np.triu_indices(n, k=1)
    distances = 1.0 - sim[iu]
    return float(np.mean(distances))


def near_duplicate_rate(embeddings: np.ndarray, *, threshold: float = 0.95) -> float:
    """Fraction of pairs with cosine similarity > threshold."""
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    u = embeddings / norm
    sim = u @ u.T
    iu = np.triu_indices(n, k=1)
    pairs = sim[iu]
    return float(np.mean(pairs > threshold))


def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-n: |unique n-grams| / |total n-grams| across all texts."""
    total = 0
    unique: set[tuple[str, ...]] = set()
    for t in texts:
        ngs = _ngram_set(t, n)
        total += len(ngs)
        unique.update(ngs)
    if total == 0:
        return float("nan")
    return len(unique) / total


def exact_duplicate_rate(texts: list[str]) -> float:
    """Fraction of texts that are byte-identical to another text in the set."""
    if len(texts) < 2:
        return float("nan")
    normed = [re.sub(r"\s+", " ", t.strip()) for t in texts]
    counts: dict[str, int] = {}
    for t in normed:
        counts[t] = counts.get(t, 0) + 1
    dupes = sum(c for c in counts.values() if c > 1)
    return dupes / len(texts)


__all__ = [
    "self_bleu_proxy",
    "mean_pairwise_cosine_distance",
    "near_duplicate_rate",
    "distinct_n",
    "exact_duplicate_rate",
]
