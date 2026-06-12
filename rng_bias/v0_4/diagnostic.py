"""Phase 0 diagnostic for BEE v0.4.

Before any RL training, measure base-vs-Instruct Lane-A TV across all 10
distribution tasks on the matched pairs available on Tinker. The question:

  Is the v0.3 narrowing observed on random-integer-1-100 a *general* property
  of post-training (gap correlated across distribution tasks), or a 1-100
  artifact (gap is large only on the trained task)?

If the post-training side has higher TV on a strong majority of tasks (>= 7/10)
across a majority of pairs, the v0.4 RL hypothesis is viable. Otherwise the
hypothesis is dead before any training spend.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from rng_bias.backends import BackendModelSpec, make_backend
from rng_bias.modeling import load_dotenv
from rng_bias.v0_4.distribution_tasks import TASKS
from rng_bias.v0_4.eval_lane_a import TaskEvalConfig, lane_a_all_tasks
from rng_bias.v0_4.model_registry import MODEL_FAMILIES, resolve_model_config


# Derived from the model registry so the Phase 0 cross-pair diagnostic and the
# stage backends never drift apart. (pair_id, base_id, instruct_id, family, backend).
V04_TINKER_PAIRS: tuple[tuple[str, str, str, str, str], ...] = tuple(
    (config.key, config.base_model_id, config.instruct_model_id, config.family, "tinker")
    for config in MODEL_FAMILIES.values()
)


@dataclass(frozen=True)
class DiagnosticConfig:
    output_dir: Path
    pair_ids: tuple[str, ...] = ("qwen3_30b_a3b",)
    paraphrase_count: int = 3
    min_post_above_base_per_pair: int = 7
    min_pairs_meeting_threshold: int = 1


def _pair_specs(pair_id: str) -> tuple[BackendModelSpec, BackendModelSpec]:
    config = resolve_model_config(pair_id)  # raises ValueError on unknown pair
    base_spec = BackendModelSpec(
        model_id=config.base_model_id, pair_id=config.key, status="base", family=config.family, backend_id="tinker"
    )
    post_spec = BackendModelSpec(
        model_id=config.instruct_model_id, pair_id=config.key, status="post_trained", family=config.family, backend_id="tinker"
    )
    return base_spec, post_spec


def _per_pair_summary(per_task: pd.DataFrame) -> dict[str, object]:
    """Compare base and post TV for each task within one pair."""
    by_status = per_task.set_index(["task_id", "model_status"])["tv_uniform"].unstack("model_status")
    delta = by_status["post_trained"] - by_status["base"]
    n_tasks = int(len(delta))
    post_above_base_n = int((delta > 0).sum())
    mean_delta = float(delta.mean())
    median_delta = float(delta.median())
    return {
        "n_tasks": n_tasks,
        "post_above_base_n": post_above_base_n,
        "post_above_base_fraction": post_above_base_n / max(n_tasks, 1),
        "mean_tv_delta_post_minus_base": mean_delta,
        "median_tv_delta_post_minus_base": median_delta,
        "per_task_delta": {task: float(delta[task]) for task in delta.index},
    }


def run_phase_0(config: DiagnosticConfig) -> str:
    output_dir = Path(config.output_dir).resolve()
    metrics_dir = output_dir / "data/bee_v0_4/metrics"
    reports_dir = output_dir / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    eval_config = TaskEvalConfig(paraphrase_count=config.paraphrase_count)

    all_rows: list[pd.DataFrame] = []
    per_pair_summary: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []

    for pair_id in config.pair_ids:
        try:
            base_spec, post_spec = _pair_specs(pair_id)
            base_backend = make_backend(base_spec)
            try:
                base_df = lane_a_all_tasks(base_backend, tasks=TASKS, config=eval_config)
            finally:
                try:
                    base_backend.close()
                except Exception:
                    pass
            post_backend = make_backend(post_spec)
            try:
                post_df = lane_a_all_tasks(post_backend, tasks=TASKS, config=eval_config)
            finally:
                try:
                    post_backend.close()
                except Exception:
                    pass
            pair_df = pd.concat([base_df, post_df], ignore_index=True)
            all_rows.append(pair_df)
            per_pair_summary[pair_id] = _per_pair_summary(pair_df)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "pair_id": pair_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )

    rows_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    rows_df.to_csv(metrics_dir / "bee_v0_4_phase_0_metrics.csv", index=False)
    pd.DataFrame(failures).to_csv(metrics_dir / "bee_v0_4_phase_0_failures.csv", index=False)

    qualifying_pairs = [
        pid for pid, summary in per_pair_summary.items()
        if summary["post_above_base_n"] >= config.min_post_above_base_per_pair
    ]
    hypothesis_viable = len(qualifying_pairs) >= config.min_pairs_meeting_threshold
    decision = "phase_0_viable" if hypothesis_viable else "phase_0_dead"

    manifest = {
        "config": {**asdict(config), "output_dir": str(output_dir), "pair_ids": list(config.pair_ids)},
        "per_pair_summary": per_pair_summary,
        "qualifying_pairs": qualifying_pairs,
        "decision": decision,
        "failures": failures,
    }
    (reports_dir / "bee_v0_4_phase_0_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# BEE v0.4 — Phase 0 Diagnostic",
        "",
        f"Decision: `{decision}`",
        f"Pairs evaluated: {', '.join(config.pair_ids) or '(none)'}",
        f"Qualifying pairs (post above base on ≥ {config.min_post_above_base_per_pair}/10 tasks): {', '.join(qualifying_pairs) or '(none)'}",
        "",
        "## Per-pair summary",
        "",
        "| pair_id | post_above_base_n | fraction | mean_delta | median_delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pid, summary in per_pair_summary.items():
        lines.append(
            f"| {pid} | {summary['post_above_base_n']}/{summary['n_tasks']} | "
            f"{summary['post_above_base_fraction']:.2f} | "
            f"{summary['mean_tv_delta_post_minus_base']:+.4f} | "
            f"{summary['median_tv_delta_post_minus_base']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Hypothesis viable: post-training narrowing is correlated across distribution tasks. "
                "Proceed to Stage 1."
                if hypothesis_viable
                else "Hypothesis dead: post-training narrowing is largely task-specific. "
                "Stop before training."
            ),
            "",
        ]
    )
    if failures:
        lines.extend(["## Failures", ""])
        for entry in failures:
            lines.append(f"- `{entry['pair_id']}`: `{entry['error_type']}` — {entry['error']}")
    (reports_dir / "bee_v0_4_phase_0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEE v0.4 Phase 0 diagnostic.")
    parser.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_phase_0"))
    parser.add_argument(
        "--pairs",
        default="qwen3_30b_a3b",
        help="Comma-separated pair_ids to evaluate. Default: just the v0.4 target.",
    )
    parser.add_argument("--paraphrase-count", type=int, default=3)
    parser.add_argument("--min-post-above-base-per-pair", type=int, default=7)
    parser.add_argument("--min-pairs-meeting-threshold", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    pair_ids = tuple(p.strip() for p in args.pairs.split(",") if p.strip())
    config = DiagnosticConfig(
        output_dir=args.output_dir,
        pair_ids=pair_ids,
        paraphrase_count=args.paraphrase_count,
        min_post_above_base_per_pair=args.min_post_above_base_per_pair,
        min_pairs_meeting_threshold=args.min_pairs_meeting_threshold,
    )
    decision = run_phase_0(config)
    print(decision)
    return 0 if decision == "phase_0_viable" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V04_TINKER_PAIRS",
    "DiagnosticConfig",
    "run_phase_0",
]
