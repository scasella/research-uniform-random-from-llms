"""BEE v0.4 — Stage 2 first-signal multi-task CLI.

200 GRPO steps on all 3 training tasks. Three gates:
1. On-task TV drops by >= 0.10 on >= 2/3 trained tasks.
2. Held-out transfer TV drops by >= 0.05 on >= 1/3 sampled held-out tasks.
3. Sampled-generation invalid rate <= 30% on a control prompt.

Exit code: 0 if all gates pass, 1 if any gate fails, 2 on infrastructure error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from rng_bias.v0_4.decision import StageGateConfig, stage_2_gate
from rng_bias.v0_4.distribution_tasks import get_task
from rng_bias.v0_4.eval_lane_a import TaskEvalConfig
from rng_bias.v0_4.train import TrainConfig, run_training


STAGE_NAME = "stage_2"
TRAIN_TASK_IDS: tuple[str, ...] = ("random_int_1_100", "random_color", "random_fruit")
STAGE2_HELDOUT_TASK_IDS: tuple[str, ...] = ("random_animal", "random_int_1_10", "random_word")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEE v0.4 Stage 2 first-signal multi-task.")
    parser.add_argument("--run-name", default="v0_4_stage2")
    parser.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_stage2"))
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-response-tokens", type=int, default=8)
    parser.add_argument("--paraphrase-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resume-from-path",
        default=None,
        help="Optional tinker:// path of a Stage 1 checkpoint to continue training from.",
    )
    parser.add_argument(
        "--invalid-control-samples",
        type=int,
        default=128,
        help="Number of free-generation samples used to measure invalid-rate gate.",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def _measure_invalid_rate(backend, n_samples: int, seed: int) -> float:
    """Sample n_samples lenient (32-token) responses on random_int_1_100, return invalid rate."""
    task = get_task("random_int_1_100")
    prompt = task.render_flat()
    rows = backend.sample(
        prompt=prompt,
        n_samples=n_samples,
        max_new_tokens=32,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=seed,
        batch_size=min(32, n_samples),
    )
    invalid = sum(1 for r in rows if task.parser(str(r.get("text", ""))) is None)
    return invalid / max(len(rows), 1)


def _report_lines(
    *,
    baseline_train_tv: dict[str, float],
    post_train_tv: dict[str, float],
    baseline_heldout_tv: dict[str, float],
    post_heldout_tv: dict[str, float],
    invalid_rate: float,
    gate_status: str,
    gate_evidence: dict[str, object],
    checkpoint_path: str | None,
) -> list[str]:
    lines = [
        "# BEE v0.4 — Stage 2 First-Signal Multi-Task",
        "",
        f"Target model: `{V04_TARGET_MODEL_ID}`",
        f"Train tasks: {', '.join(TRAIN_TASK_IDS)}",
        f"Stage-2 held-out tasks: {', '.join(STAGE2_HELDOUT_TASK_IDS)}",
        f"Gate result: `{gate_status}`",
        "",
        "## Trained-task TV-to-uniform",
        "",
        "| task_id | baseline TV | post TV | Δ (baseline − post) |",
        "| --- | --- | --- | --- |",
    ]
    for tid in TRAIN_TASK_IDS:
        b = baseline_train_tv.get(tid, float("nan"))
        a = post_train_tv.get(tid, float("nan"))
        lines.append(f"| {tid} | {b:.4f} | {a:.4f} | {b - a:+.4f} |")
    lines.extend(
        [
            "",
            "## Held-out transfer TV-to-uniform",
            "",
            "| task_id | baseline TV | post TV | Δ (baseline − post) |",
            "| --- | --- | --- | --- |",
        ]
    )
    for tid in STAGE2_HELDOUT_TASK_IDS:
        b = baseline_heldout_tv.get(tid, float("nan"))
        a = post_heldout_tv.get(tid, float("nan"))
        lines.append(f"| {tid} | {b:.4f} | {a:.4f} | {b - a:+.4f} |")
    lines.extend(
        [
            "",
            f"Lenient sampled invalid rate (post): `{invalid_rate:.4f}`",
            "",
            "## Gate evidence",
            "",
            "```json",
            json.dumps(gate_evidence, indent=2, sort_keys=True),
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

    panel_ids = TRAIN_TASK_IDS + STAGE2_HELDOUT_TASK_IDS

    # 1) Baseline eval
    baseline_csv = paths.metrics_dir / f"bee_v0_4_{STAGE_NAME}_baseline.csv"
    if args.skip_baseline and baseline_csv.exists() and baseline_csv.stat().st_size > 0:
        baseline_df = pd.read_csv(baseline_csv)
    else:
        try:
            backend = build_instruct_backend()
            try:
                baseline_df = evaluate_panel(backend, panel_ids, eval_config=eval_config)
            finally:
                try:
                    backend.close()
                except Exception:
                    pass
            baseline_df.to_csv(baseline_csv, index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Stage 2 baseline eval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    baseline_tv_all = tv_by_task(baseline_df)
    baseline_train_tv = {tid: baseline_tv_all.get(tid, float("nan")) for tid in TRAIN_TASK_IDS}
    baseline_heldout_tv = {tid: baseline_tv_all.get(tid, float("nan")) for tid in STAGE2_HELDOUT_TASK_IDS}

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
            save_every_n_steps=max(50, args.n_steps // 4),
            resume_from_path=args.resume_from_path,
        )
        train_manifest = run_training(train_config)
        checkpoint_path = str(train_manifest.get("final_checkpoint_path") or "")
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 2 training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not checkpoint_path:
        print("Stage 2 produced no checkpoint; cannot evaluate.", file=sys.stderr)
        return 2

    # 3) Post-training eval (panel + invalid rate)
    try:
        trained = build_instruct_backend(model_path=checkpoint_path)
        try:
            post_df = evaluate_panel(trained, panel_ids, eval_config=eval_config)
            invalid_rate = _measure_invalid_rate(trained, args.invalid_control_samples, args.seed + 999_999)
        finally:
            try:
                trained.close()
            except Exception:
                pass
        post_csv = paths.metrics_dir / f"bee_v0_4_{STAGE_NAME}_post_training.csv"
        post_df.to_csv(post_csv, index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 2 post-training eval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    post_tv_all = tv_by_task(post_df)
    post_train_tv = {tid: post_tv_all.get(tid, float("nan")) for tid in TRAIN_TASK_IDS}
    post_heldout_tv = {tid: post_tv_all.get(tid, float("nan")) for tid in STAGE2_HELDOUT_TASK_IDS}

    # 4) Gate
    gate_status, gate_evidence = stage_2_gate(
        baseline_train_tv,
        post_train_tv,
        baseline_heldout_tv,
        post_heldout_tv,
        invalid_rate,
        StageGateConfig(),
    )
    write_json(paths.reports_dir / f"bee_v0_4_{STAGE_NAME}_gate.json", {
        "status": gate_status,
        "evidence": gate_evidence,
        "baseline_train_tv": baseline_train_tv,
        "post_train_tv": post_train_tv,
        "baseline_heldout_tv": baseline_heldout_tv,
        "post_heldout_tv": post_heldout_tv,
        "invalid_rate_post": invalid_rate,
        "checkpoint_path": checkpoint_path,
    })
    lines = _report_lines(
        baseline_train_tv=baseline_train_tv,
        post_train_tv=post_train_tv,
        baseline_heldout_tv=baseline_heldout_tv,
        post_heldout_tv=post_heldout_tv,
        invalid_rate=invalid_rate,
        gate_status=gate_status,
        gate_evidence=gate_evidence,
        checkpoint_path=checkpoint_path,
    )
    write_markdown(paths.reports_dir / f"bee_v0_4_{STAGE_NAME}_report.md", lines)
    return 0 if gate_status == "stage_2_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
