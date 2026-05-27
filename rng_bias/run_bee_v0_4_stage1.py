"""BEE v0.4 — Stage 1 sanity smoke CLI.

50 GRPO steps on Qwen3-30B-A3B-Instruct-2507 with the single training task
`random_int_1_100`. Baseline + post-training Lane-A TV measurement. Gate:
on-task TV drops by >= 0.05 absolute.

Exit code: 0 if gate passes, 1 if gate fails, 2 on infrastructure error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rng_bias.modeling import load_dotenv
from rng_bias.v0_4._shared import (
    StagePaths,
    build_instruct_backend,
    evaluate_panel,
    tv_by_task,
    write_json,
    write_markdown,
    V04_TARGET_MODEL_ID,
)
from rng_bias.v0_4.decision import StageGateConfig, stage_1_gate
from rng_bias.v0_4.eval_lane_a import TaskEvalConfig
from rng_bias.v0_4.train import TrainConfig, run_training


STAGE_NAME = "stage_1"
TRAIN_TASK_IDS: tuple[str, ...] = ("random_int_1_100",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEE v0.4 Stage 1 sanity smoke.")
    parser.add_argument("--run-name", default="v0_4_stage1")
    parser.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_stage1"))
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-response-tokens", type=int, default=8)
    parser.add_argument("--paraphrase-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-baseline", action="store_true", help="Reuse a prior baseline-eval CSV if present.")
    return parser.parse_args()


def _stage_1_report_lines(
    *,
    baseline_tv: dict[str, float],
    post_tv: dict[str, float],
    gate_status: str,
    gate_evidence: dict[str, object],
    checkpoint_path: str | None,
) -> list[str]:
    lines = [
        "# BEE v0.4 — Stage 1 Sanity Smoke",
        "",
        f"Target model: `{V04_TARGET_MODEL_ID}`",
        f"Train task: `{TRAIN_TASK_IDS[0]}`",
        f"Gate result: `{gate_status}`",
        "",
        "## On-task Lane-A TV-to-uniform",
        "",
        "| task_id | baseline TV | post TV | Δ (baseline − post) |",
        "| --- | --- | --- | --- |",
    ]
    for task_id in TRAIN_TASK_IDS:
        before = baseline_tv.get(task_id, float("nan"))
        after = post_tv.get(task_id, float("nan"))
        delta = before - after
        lines.append(f"| {task_id} | {before:.4f} | {after:.4f} | {delta:+.4f} |")
    lines.extend(
        [
            "",
            "## Gate evidence",
            "",
            "```json",
            __import__("json").dumps(gate_evidence, indent=2, sort_keys=True),
            "```",
            "",
            f"Final checkpoint: `{checkpoint_path or '(none)'}`",
            "",
        ]
    )
    return lines


def main() -> int:
    load_dotenv()
    args = parse_args()
    paths = StagePaths.from_output_dir(args.output_dir)
    paths.ensure()

    eval_config = TaskEvalConfig(paraphrase_count=args.paraphrase_count)

    # 1) Baseline eval (no training)
    baseline_csv = paths.metrics_dir / f"bee_v0_4_{STAGE_NAME}_baseline.csv"
    if args.skip_baseline and baseline_csv.exists() and baseline_csv.stat().st_size > 0:
        import pandas as pd
        baseline_df = pd.read_csv(baseline_csv)
        print(f"Stage 1: reusing baseline CSV at {baseline_csv}")
    else:
        try:
            backend = build_instruct_backend()
            try:
                baseline_df = evaluate_panel(backend, TRAIN_TASK_IDS, eval_config=eval_config)
            finally:
                try:
                    backend.close()
                except Exception:
                    pass
            baseline_df.to_csv(baseline_csv, index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Stage 1 baseline eval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    baseline_tv = tv_by_task(baseline_df)
    print(f"Stage 1 baseline TV: {baseline_tv}")

    # 2) Training
    try:
        train_config = TrainConfig(
            stage=STAGE_NAME,
            output_dir=paths.output_dir,
            run_name=args.run_name,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            group_size=args.group_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            max_response_tokens=args.max_response_tokens,
            train_task_ids=TRAIN_TASK_IDS,
            seed=args.seed,
            save_every_n_steps=max(1, args.n_steps),
        )
        train_manifest = run_training(train_config)
        checkpoint_path = str(train_manifest.get("final_checkpoint_path") or "")
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 1 training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Stage 1 trained checkpoint: {checkpoint_path}")

    if not checkpoint_path:
        print("Stage 1 produced no checkpoint; cannot evaluate.", file=sys.stderr)
        return 2

    # 3) Post-training eval against the saved checkpoint
    try:
        trained_backend = build_instruct_backend(model_path=checkpoint_path)
        try:
            post_df = evaluate_panel(trained_backend, TRAIN_TASK_IDS, eval_config=eval_config)
        finally:
            try:
                trained_backend.close()
            except Exception:
                pass
        post_csv = paths.metrics_dir / f"bee_v0_4_{STAGE_NAME}_post_training.csv"
        post_df.to_csv(post_csv, index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 1 post-training eval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    post_tv = tv_by_task(post_df)
    print(f"Stage 1 post TV: {post_tv}")

    # 4) Gate
    task_id = TRAIN_TASK_IDS[0]
    gate_status, gate_evidence = stage_1_gate(
        baseline_tv.get(task_id, 1.0),
        post_tv.get(task_id, 1.0),
        StageGateConfig(),
    )
    print(f"Stage 1 gate: {gate_status}")
    write_json(paths.reports_dir / f"bee_v0_4_{STAGE_NAME}_gate.json", {
        "status": gate_status,
        "evidence": gate_evidence,
        "baseline_tv": baseline_tv,
        "post_tv": post_tv,
        "checkpoint_path": checkpoint_path,
    })
    lines = _stage_1_report_lines(
        baseline_tv=baseline_tv,
        post_tv=post_tv,
        gate_status=gate_status,
        gate_evidence=gate_evidence,
        checkpoint_path=checkpoint_path,
    )
    write_markdown(paths.reports_dir / f"bee_v0_4_{STAGE_NAME}_report.md", lines)

    return 0 if gate_status == "stage_1_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
