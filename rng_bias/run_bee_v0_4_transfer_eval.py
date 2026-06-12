"""Quick transfer-eval: take an RL'd checkpoint and measure Lane-A TV across
all 10 distribution tasks, then compare to the Phase 0 baseline.

This answers the v0.4 transfer question DIRECTLY without doing more training:
did the single-task Stage 1 RL on `random_int_1_100` transfer to the 9 other
distribution tasks?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rng_bias.modeling import load_dotenv
from rng_bias.v0_4._shared import (
    StagePaths,
    build_instruct_backend,
    resolve_model_config,
    tv_by_task,
    write_json,
    write_markdown,
)
from rng_bias.v0_4.distribution_tasks import HELDOUT_TASKS, TRAIN_TASKS
from rng_bias.v0_4.eval_lane_a import TaskEvalConfig, lane_a_all_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v0.4 transfer on a trained checkpoint.")
    parser.add_argument("--checkpoint-path", required=True, help="tinker:// path of the RL'd checkpoint.")
    parser.add_argument(
        "--model",
        default="qwen3_30b_a3b",
        help="Model-family key from the registry (qwen3_30b_a3b | llama_3_1_8b | qwen3_8b).",
    )
    parser.add_argument("--phase-0-dir", type=Path, default=Path("bee_v0_4_phase_0"))
    parser.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_transfer"))
    parser.add_argument("--paraphrase-count", type=int, default=3)
    parser.add_argument(
        "--trained-task-ids",
        default="random_int_1_100",
        help="Comma-separated task_ids that were trained (rest are held-out for transfer).",
    )
    return parser.parse_args()


def _load_phase_0_baseline(phase_0_dir: Path) -> dict[str, dict[str, float]]:
    """Returns {task_id: {'base_tv': ..., 'post_tv': ...}} from Phase 0 metrics."""
    metrics_path = phase_0_dir / "data/bee_v0_4/metrics/bee_v0_4_phase_0_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Phase 0 metrics not found at {metrics_path}. Run rng_bias.v0_4.diagnostic first."
        )
    df = pd.read_csv(metrics_path)
    result: dict[str, dict[str, float]] = {}
    for task_id in df["task_id"].unique():
        sub = df[df["task_id"] == task_id]
        base = sub[sub["model_status"] == "base"]
        post = sub[sub["model_status"] == "post_trained"]
        if base.empty or post.empty:
            continue
        result[task_id] = {
            "base_tv": float(base.iloc[0]["tv_uniform"]),
            "post_tv": float(post.iloc[0]["tv_uniform"]),
        }
    return result


def main() -> int:
    load_dotenv()
    args = parse_args()
    model_config = resolve_model_config(args.model)
    trained_task_ids = {tid.strip() for tid in args.trained_task_ids.split(",") if tid.strip()}

    paths = StagePaths.from_output_dir(args.output_dir)
    paths.ensure()

    baseline = _load_phase_0_baseline(args.phase_0_dir)

    print(f"Evaluating RL'd checkpoint on all {len(TRAIN_TASKS) + len(HELDOUT_TASKS)} distribution tasks...")
    print(f"Target model: {model_config.label}")
    print(f"Checkpoint: {args.checkpoint_path}")

    backend = build_instruct_backend(model_path=args.checkpoint_path, config=model_config)
    try:
        all_tasks = tuple(list(TRAIN_TASKS) + list(HELDOUT_TASKS))
        eval_config = TaskEvalConfig(paraphrase_count=args.paraphrase_count)
        post_df = lane_a_all_tasks(backend, tasks=all_tasks, config=eval_config)
    finally:
        try:
            backend.close()
        except Exception:
            pass
    post_df.to_csv(paths.metrics_dir / "bee_v0_4_transfer_post_training.csv", index=False)
    post_tv = tv_by_task(post_df)

    # Build the comparison table
    rows: list[dict[str, object]] = []
    for task_id in sorted(post_tv.keys()):
        base_tv = baseline.get(task_id, {}).get("base_tv", float("nan"))
        instruct_tv = baseline.get(task_id, {}).get("post_tv", float("nan"))  # untrained Instruct
        trained_tv = post_tv[task_id]
        # How much did training drop TV vs the original Instruct baseline?
        delta = instruct_tv - trained_tv
        # How does the trained Instruct compare to the matched Base?
        delta_vs_base = base_tv - trained_tv
        rows.append({
            "task_id": task_id,
            "split": "train" if task_id in trained_task_ids else "heldout",
            "base_tv": base_tv,
            "instruct_baseline_tv": instruct_tv,
            "trained_tv": trained_tv,
            "tv_drop_vs_instruct": delta,
            "tv_drop_vs_base": delta_vs_base,
        })
    df = pd.DataFrame(rows)
    df.to_csv(paths.metrics_dir / "bee_v0_4_transfer_comparison.csv", index=False)

    train_df = df[df["split"] == "train"]
    heldout_df = df[df["split"] == "heldout"]
    train_mean_drop = float(train_df["tv_drop_vs_instruct"].mean()) if not train_df.empty else float("nan")
    heldout_mean_drop = float(heldout_df["tv_drop_vs_instruct"].mean()) if not heldout_df.empty else float("nan")
    heldout_positive_n = int((heldout_df["tv_drop_vs_instruct"] > 0.05).sum())
    heldout_total = int(len(heldout_df))

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "trained_task_ids": sorted(trained_task_ids),
        "n_trained_tasks": int(len(train_df)),
        "n_heldout_tasks": heldout_total,
        "train_mean_tv_drop": train_mean_drop,
        "heldout_mean_tv_drop": heldout_mean_drop,
        "heldout_passing_threshold_0_05": heldout_positive_n,
    }
    write_json(paths.reports_dir / "bee_v0_4_transfer_summary.json", summary)

    # Markdown report
    lines = [
        "# BEE v0.4 — Transfer Evaluation (single-task RL → 10-task panel)",
        "",
        f"Target model: `{model_config.instruct_model_id}`",
        f"Checkpoint: `{args.checkpoint_path}`",
        f"Trained task(s): {', '.join(sorted(trained_task_ids))}",
        "",
        "## Per-task Lane-A TV-to-uniform",
        "",
        "| task_id | split | base TV | Instruct baseline TV | trained TV | Δ vs Instruct | Δ vs Base |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    df_sorted = df.sort_values(["split", "tv_drop_vs_instruct"], ascending=[True, False])
    for _, row in df_sorted.iterrows():
        marker = " ← trained" if row["split"] == "train" else ""
        lines.append(
            f"| `{row['task_id']}`{marker} | {row['split']} | "
            f"{row['base_tv']:.4f} | {row['instruct_baseline_tv']:.4f} | "
            f"{row['trained_tv']:.4f} | {row['tv_drop_vs_instruct']:+.4f} | "
            f"{row['tv_drop_vs_base']:+.4f} |"
        )
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Trained tasks: {len(train_df)}; mean TV drop vs Instruct = `{train_mean_drop:+.4f}`",
        f"- Held-out tasks: {heldout_total}; mean TV drop vs Instruct = `{heldout_mean_drop:+.4f}`",
        f"- Held-out tasks passing transfer threshold (TV drop ≥ 0.05): `{heldout_positive_n}/{heldout_total}`",
        "",
        "## Interpretation",
        "",
    ])
    if heldout_positive_n >= max(1, (heldout_total + 1) // 2):
        lines.append(
            f"**Transfer evidence:** the single-task RL on `{', '.join(sorted(trained_task_ids))}` reduced "
            f"TV on `{heldout_positive_n}/{heldout_total}` held-out distribution tasks "
            f"by at least 0.05. Post-training narrowing appears to share a subspace across distribution tasks."
        )
    elif heldout_positive_n >= 1:
        lines.append(
            f"**Partial transfer:** `{heldout_positive_n}/{heldout_total}` held-out tasks improved by ≥ 0.05. "
            f"Transfer is real but partial; the shared subspace may be narrower than expected."
        )
    else:
        lines.append(
            "**No transfer:** all held-out tasks moved by < 0.05. Training on a single task did not "
            "generalize. The Stage 2 multi-task arm may still find transfer."
        )
    write_markdown(paths.reports_dir / "bee_v0_4_transfer_report.md", lines)

    print("\n=== Transfer Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
