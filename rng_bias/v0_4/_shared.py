"""Shared helpers for the v0.4 stage CLIs.

Building blocks reused across Stage 1 / 2 / 3 drivers:

- Backend factories for the v0.4 target on Tinker, parameterized by a
  `ModelFamilyConfig` (default: Qwen3-30B-A3B, the published pair) with
  optional `model_path` to point at a trained checkpoint.
- Baseline-vs-trained Lane-A measurement loops for a panel of tasks.
- Report writers that produce per-stage markdown summaries.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from rng_bias.backends import BackendModelSpec
from rng_bias.backends.tinker_backend import TinkerBackend
from rng_bias.v0_4.distribution_tasks import DistributionTask, TASK_BY_ID, get_task
from rng_bias.v0_4.eval_lane_a import TaskEvalConfig, lane_a_all_tasks, lane_a_metrics_row
from rng_bias.v0_4.model_registry import (
    ModelFamilyConfig,
    default_model_config,
    resolve_model_config,
)


# Default pair (Qwen3-30B-A3B). Kept as module constants for backward
# compatibility with callers/reports that reference them directly.
_DEFAULT_CONFIG = default_model_config()
V04_TARGET_MODEL_ID = _DEFAULT_CONFIG.instruct_model_id
V04_BASE_MODEL_ID = _DEFAULT_CONFIG.base_model_id
V04_PAIR_ID = _DEFAULT_CONFIG.pair_id
V04_FAMILY = _DEFAULT_CONFIG.family


def instruct_spec(
    config: ModelFamilyConfig, *, status: str = "post_trained"
) -> BackendModelSpec:
    return BackendModelSpec(
        model_id=config.instruct_model_id,
        pair_id=config.pair_id,
        status=status,
        family=config.family,
        backend_id="tinker",
    )


def base_spec(config: ModelFamilyConfig) -> BackendModelSpec:
    return BackendModelSpec(
        model_id=config.base_model_id,
        pair_id=config.pair_id,
        status="base",
        family=config.family,
        backend_id="tinker",
    )


def build_instruct_backend(
    *, model_path: str | None = None, config: ModelFamilyConfig | None = None
) -> TinkerBackend:
    config = config or default_model_config()
    return TinkerBackend(instruct_spec(config), model_path=model_path)


def build_base_backend(*, config: ModelFamilyConfig | None = None) -> TinkerBackend:
    config = config or default_model_config()
    return TinkerBackend(base_spec(config))


def evaluate_panel(
    backend: TinkerBackend,
    task_ids: tuple[str, ...],
    *,
    eval_config: TaskEvalConfig = TaskEvalConfig(),
) -> pd.DataFrame:
    tasks = tuple(get_task(tid) for tid in task_ids)
    return lane_a_all_tasks(backend, tasks=tasks, config=eval_config)


def tv_by_task(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    return {row["task_id"]: float(row["tv_uniform"]) for _, row in df.iterrows()}


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class StagePaths:
    output_dir: Path
    metrics_dir: Path
    reports_dir: Path
    checkpoints_dir: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "StagePaths":
        output_dir = Path(output_dir).resolve()
        return cls(
            output_dir=output_dir,
            metrics_dir=output_dir / "data/bee_v0_4/metrics",
            reports_dir=output_dir / "reports",
            checkpoints_dir=output_dir / "data/bee_v0_4/checkpoints",
        )

    def ensure(self) -> None:
        for path in [self.metrics_dir, self.reports_dir, self.checkpoints_dir]:
            path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "V04_TARGET_MODEL_ID",
    "V04_BASE_MODEL_ID",
    "V04_PAIR_ID",
    "V04_FAMILY",
    "ModelFamilyConfig",
    "default_model_config",
    "resolve_model_config",
    "instruct_spec",
    "base_spec",
    "build_instruct_backend",
    "build_base_backend",
    "evaluate_panel",
    "tv_by_task",
    "write_json",
    "write_markdown",
    "StagePaths",
]
