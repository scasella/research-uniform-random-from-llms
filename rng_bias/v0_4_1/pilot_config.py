"""Frozen pilot configuration for BEE v0.4.1.

Single task (bug_hypotheses), two scenarios (one narrow + one open-ended),
two conditions (baseline + trained). CAD-rerank is excluded from the pilot
to keep the failure surface small; it enters at the full sweep.

This module is the single source of truth — the pilot driver imports
PILOT and never overrides individual fields.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PilotConfig:
    cells: tuple[tuple[str, str], ...]  # (task_id, scenario_id) pairs
    conditions: tuple[str, ...]
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    n_per_cell: int
    judge_shuffle_seed: int


PILOT = PilotConfig(
    cells=(
        ("bug_hypotheses", "bug_h_01"),
        ("bug_hypotheses", "bug_h_03"),
        ("product_names", "name_01"),
        ("product_names", "name_04"),
    ),
    conditions=("baseline", "trained"),
    seed=20260526,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=4000,
    n_per_cell=20,
    judge_shuffle_seed=7,
)


# Gate thresholds (the pilot writes pilot_gate.json against these)
TINKER_RETRY_FRACTION_MAX = 0.05
VALIDITY_FLOOR_MIN_VALID = 18
EMBEDDING_DISTANCE_MIN = 0.05
EMBEDDING_DISTANCE_MAX = 0.85
DBSCAN_MIN_CLUSTERS = 2
DBSCAN_MAX_CLUSTERS = 18
JUDGE_PARSE_RATE_MIN = 0.95
HEADLINE_GAP_MIN = 0.05
JOINT_SCORE_GAP_MIN = 0.03


__all__ = [
    "PilotConfig",
    "PILOT",
    "TINKER_RETRY_FRACTION_MAX",
    "VALIDITY_FLOOR_MIN_VALID",
    "EMBEDDING_DISTANCE_MIN",
    "EMBEDDING_DISTANCE_MAX",
    "DBSCAN_MIN_CLUSTERS",
    "DBSCAN_MAX_CLUSTERS",
    "JUDGE_PARSE_RATE_MIN",
    "HEADLINE_GAP_MIN",
    "JOINT_SCORE_GAP_MIN",
]
