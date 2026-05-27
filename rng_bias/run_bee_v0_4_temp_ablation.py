"""BEE v0.4 — temperature ablation.

Tests the foundational v0.4 claim under sampling: does the trained checkpoint's
empirical TV-to-uniform on the 10 distribution tasks survive temperature
variation, and does vanilla Qwen3-30B-A3B-Instruct at higher temperatures
reach the same empirical TV that v0.4 reports at T=1.0?

Lane A in v0.4 measures TV from candidate logprobs (temperature-invariant by
construction). This driver measures TV from actual samples at T in {0.7, 1.0,
1.5} for both vanilla and trained, on all 10 tasks. 60 cells total.

Usage:
    python -m rng_bias.run_bee_v0_4_temp_ablation \\
        --checkpoint-path tinker://.../v0_4_stage1_step_50 \\
        --output-dir bee_v0_4_temp_ablation \\
        --n-samples 150
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

from rng_bias.modeling import load_dotenv
from rng_bias.v0_4._shared import (
    StagePaths,
    V04_TARGET_MODEL_ID,
    build_instruct_backend,
    write_json,
    write_markdown,
)
from rng_bias.v0_4.distribution_tasks import HELDOUT_TASKS, TRAIN_TASKS
from rng_bias.v0_4.sample_eval import SampleEvalConfig, empirical_distribution_for_task


DEFAULT_TEMPERATURES = (0.7, 1.0, 1.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0.4 temperature ablation under sampling.")
    p.add_argument("--checkpoint-path", required=True, help="tinker:// path of the v0.4 LoRA checkpoint.")
    p.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_temp_ablation"))
    p.add_argument("--n-samples", type=int, default=150)
    p.add_argument("--paraphrase-count", type=int, default=3)
    p.add_argument(
        "--temperatures",
        default="0.7,1.0,1.5",
        help="Comma-separated temperatures (default 0.7,1.0,1.5).",
    )
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


async def _eval_condition(
    *,
    backend,
    tasks,
    temperatures: tuple[float, ...],
    n_samples: int,
    paraphrase_count: int,
    seed: int,
    condition_label: str,
    concurrency: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for temperature in temperatures:
        cfg = SampleEvalConfig(
            n_samples=n_samples,
            paraphrase_count=paraphrase_count,
            temperature=float(temperature),
            top_p=1.0,
            top_k=0,
            max_new_tokens=16,
            concurrency=concurrency,
        )
        for task in tasks:
            print(f"  [{condition_label}] T={temperature:.2f}  {task.task_id} ...", flush=True)
            row = await empirical_distribution_for_task(
                backend, task, config=cfg, seed_base=seed + int(temperature * 1000)
            )
            row["condition"] = condition_label
            row["temperature"] = float(temperature)
            print(
                f"      tv={row['tv_uniform']:.4f}  valid={row['n_valid']}/{row['n_samples_total']}  "
                f"top5=[{row['top5_values']}]",
                flush=True,
            )
            rows.append(row)
    return rows


def _summarize(df: pd.DataFrame) -> dict[str, object]:
    """Compute means by condition x temperature, plus the load-bearing comparison:
    does vanilla-at-T-high TV reach trained-at-T=1.0 TV?"""
    out: dict[str, object] = {}
    grouped = df.groupby(["condition", "temperature"], dropna=False)
    summary = grouped.agg(
        mean_tv=("tv_uniform", "mean"),
        median_tv=("tv_uniform", "median"),
        mean_valid_rate=("valid_rate", "mean"),
        n_cells=("task_id", "count"),
    ).reset_index()
    out["per_condition_temperature"] = summary.to_dict(orient="records")

    trained_at_1 = df[(df["condition"] == "trained") & (df["temperature"] == 1.0)]
    if not trained_at_1.empty:
        out["trained_at_T_1.0_mean_tv"] = float(trained_at_1["tv_uniform"].mean())
        comparisons = []
        for t in sorted(df["temperature"].unique()):
            vanilla_at_t = df[(df["condition"] == "baseline") & (df["temperature"] == t)]
            if vanilla_at_t.empty:
                continue
            comparisons.append(
                {
                    "vanilla_temperature": float(t),
                    "vanilla_mean_tv": float(vanilla_at_t["tv_uniform"].mean()),
                    "gap_vs_trained_T_1.0": float(
                        vanilla_at_t["tv_uniform"].mean() - trained_at_1["tv_uniform"].mean()
                    ),
                }
            )
        out["vanilla_vs_trained_at_T_1.0"] = comparisons
    return out


def main() -> int:
    load_dotenv()
    args = parse_args()

    temperatures = tuple(float(t) for t in args.temperatures.split(",") if t.strip())
    paths = StagePaths.from_output_dir(args.output_dir)
    paths.ensure()

    all_tasks = tuple(list(TRAIN_TASKS) + list(HELDOUT_TASKS))
    print(f"v0.4 temperature ablation: {len(all_tasks)} tasks x 2 conditions x "
          f"{len(temperatures)} temperatures = {len(all_tasks) * 2 * len(temperatures)} cells")
    print(f"  n_samples per cell = {args.n_samples}, paraphrase_count = {args.paraphrase_count}")
    print(f"  temperatures = {temperatures}")
    print(f"  checkpoint = {args.checkpoint_path}")

    all_rows: list[dict[str, object]] = []

    print("\n--- vanilla baseline ---")
    baseline_backend = build_instruct_backend(model_path=None)
    try:
        rows = asyncio.run(_eval_condition(
            backend=baseline_backend,
            tasks=all_tasks,
            temperatures=temperatures,
            n_samples=args.n_samples,
            paraphrase_count=args.paraphrase_count,
            seed=args.seed,
            condition_label="baseline",
            concurrency=args.concurrency,
        ))
        all_rows.extend(rows)
    finally:
        try:
            baseline_backend.close()
        except Exception:
            pass

    print("\n--- v0.4 trained ---")
    trained_backend = build_instruct_backend(model_path=args.checkpoint_path)
    try:
        rows = asyncio.run(_eval_condition(
            backend=trained_backend,
            tasks=all_tasks,
            temperatures=temperatures,
            n_samples=args.n_samples,
            paraphrase_count=args.paraphrase_count,
            seed=args.seed,
            condition_label="trained",
            concurrency=args.concurrency,
        ))
        all_rows.extend(rows)
    finally:
        try:
            trained_backend.close()
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    df.to_csv(paths.metrics_dir / "bee_v0_4_temp_ablation_cells.csv", index=False)

    summary = _summarize(df)
    summary["checkpoint_path"] = args.checkpoint_path
    summary["n_samples_per_cell"] = args.n_samples
    summary["paraphrase_count"] = args.paraphrase_count
    summary["temperatures"] = list(temperatures)
    write_json(paths.reports_dir / "bee_v0_4_temp_ablation_summary.json", summary)

    # Markdown report
    lines = [
        "# BEE v0.4 — Temperature ablation",
        "",
        f"Target model: `{V04_TARGET_MODEL_ID}`",
        f"Checkpoint: `{args.checkpoint_path}`",
        f"Tasks: {len(all_tasks)}; conditions: vanilla baseline + v0.4-trained; "
        f"temperatures: {', '.join(f'{t:.2f}' for t in temperatures)}; "
        f"n_samples/cell = {args.n_samples}",
        "",
        "## Mean empirical TV-to-uniform by condition x temperature",
        "",
        "| condition | T | mean TV | median TV | mean valid_rate | n_cells |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["per_condition_temperature"]:
        lines.append(
            f"| `{row['condition']}` | {row['temperature']:.2f} | "
            f"{row['mean_tv']:.4f} | {row['median_tv']:.4f} | "
            f"{row['mean_valid_rate']:.3f} | {row['n_cells']} |"
        )

    if "vanilla_vs_trained_at_T_1.0" in summary:
        lines.extend([
            "",
            "## The load-bearing comparison",
            "",
            f"Trained at T=1.0, mean TV across {len(all_tasks)} tasks: "
            f"**{summary['trained_at_T_1.0_mean_tv']:.4f}**.",
            "",
            "Vanilla TV at each temperature, and gap vs trained-at-T=1.0:",
            "",
            "| vanilla T | vanilla mean TV | gap vs trained@T=1.0 |",
            "| --- | --- | --- |",
        ])
        for c in summary["vanilla_vs_trained_at_T_1.0"]:
            lines.append(
                f"| {c['vanilla_temperature']:.2f} | {c['vanilla_mean_tv']:.4f} | "
                f"{c['gap_vs_trained_T_1.0']:+.4f} |"
            )
        lines.extend([
            "",
            "**Interpretation:**",
            "",
            "- If vanilla @ T=1.5 mean TV is <= trained @ T=1.0 mean TV (negative gap), the v0.4 "
            "effect at T=1.0 is reachable from vanilla by raising temperature — the LoRA's "
            "contribution is 'expensive temperature.'",
            "- If vanilla @ T=1.5 mean TV >> trained @ T=1.0 mean TV (large positive gap), the "
            "v0.4 effect is robust to the temperature axis and the LoRA is doing something "
            "temperature scaling cannot replicate.",
        ])

    lines.extend([
        "",
        "## Per-cell table",
        "",
        "See `data/bee_v0_4/metrics/bee_v0_4_temp_ablation_cells.csv` for all "
        f"{len(df)} (task, condition, temperature) cells with empirical TV, valid_rate, "
        "and top-5 candidate distribution.",
    ])
    write_markdown(paths.reports_dir / "bee_v0_4_temp_ablation_report.md", lines)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
