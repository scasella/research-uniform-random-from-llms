"""Shared helpers for v0.4.1 — paths, backend re-exports.

Backend factories come from v0.4 unchanged: same model IDs, same Tinker config.
Adds UsefulDiversityPaths, a StagePaths variant that carries the additional
generations/, embeddings/, and judge/ subdirectories this sub-experiment writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rng_bias.v0_4._shared import (
    V04_BASE_MODEL_ID,
    V04_FAMILY,
    V04_PAIR_ID,
    V04_TARGET_MODEL_ID,
    build_base_backend,
    build_instruct_backend,
    write_json,
    write_markdown,
)

V041_TARGET_MODEL_ID = V04_TARGET_MODEL_ID
V041_BASE_MODEL_ID = V04_BASE_MODEL_ID
V041_PAIR_ID = V04_PAIR_ID
V041_FAMILY = V04_FAMILY


@dataclass(frozen=True)
class UsefulDiversityPaths:
    """Directory layout for a v0.4.1 sub-experiment.

    Mirrors v0.4 StagePaths but adds generations/, embeddings/, and a
    metrics/ + reports/ split scoped under data/bee_v0_4_1/ so v0.4.1
    artifacts never collide with v0.4 ones.
    """

    output_dir: Path
    metrics_dir: Path
    reports_dir: Path
    generations_dir: Path
    embeddings_dir: Path
    judge_dir: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "UsefulDiversityPaths":
        output_dir = Path(output_dir).resolve()
        return cls(
            output_dir=output_dir,
            metrics_dir=output_dir / "data/bee_v0_4_1/metrics",
            reports_dir=output_dir / "reports",
            generations_dir=output_dir / "data/bee_v0_4_1/generations",
            embeddings_dir=output_dir / "data/bee_v0_4_1/embeddings",
            judge_dir=output_dir / "data/bee_v0_4_1/judge",
        )

    def ensure(self) -> None:
        for path in [
            self.metrics_dir,
            self.reports_dir,
            self.generations_dir,
            self.embeddings_dir,
            self.judge_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "V041_TARGET_MODEL_ID",
    "V041_BASE_MODEL_ID",
    "V041_PAIR_ID",
    "V041_FAMILY",
    "build_instruct_backend",
    "build_base_backend",
    "write_json",
    "write_markdown",
    "UsefulDiversityPaths",
]
