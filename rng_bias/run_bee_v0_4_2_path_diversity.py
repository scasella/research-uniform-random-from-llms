"""BEE v0.4.2 — Solution-path diversity on GSM8K (scaled).

25 GSM8K problems x 3 conditions (vanilla T=1.0, vanilla T=1.5, trained T=1.0)
x k=10 generations per cell. Pass@k saturates at this model scale, so the
headline metric is the number of distinct calculation paths among the correct
solutions. Adds vanilla-T=1.5 as the temperature-can-do-this comparator.

Usage:
    python -m rng_bias.run_bee_v0_4_2_path_diversity \\
        --checkpoint-path tinker://.../v0_4_stage1_step_50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import datasets
import numpy as np
import pandas as pd

from rng_bias.modeling import load_dotenv
from rng_bias.v0_4._shared import build_instruct_backend, resolve_model_config, write_markdown
from rng_bias.v0_4_2.humaneval import ConditionSpec
from rng_bias.v0_4_2.gsm8k import sample_completions, score_completions
from rng_bias.v0_4_2.solution_diversity import compute_cell


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0.4.2 path-diversity sweep on GSM8K.")
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument(
        "--model",
        default="qwen3_30b_a3b",
        help="Model-family key from the registry (qwen3_30b_a3b | llama_3_1_8b | qwen3_8b).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_2_path_diversity"))
    p.add_argument("--n-problems", type=int, default=25)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--problem-offset", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


def _bootstrap_ci(values: list[float], *, B: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    boot_means = rng.choice(arr, size=(B, n), replace=True).mean(axis=1)
    return (float(np.percentile(boot_means, 100 * alpha / 2)), float(np.percentile(boot_means, 100 * (1 - alpha / 2))))


async def run() -> int:
    args = parse_args()
    model_config = resolve_model_config(args.model)
    out = args.output_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    print(f"Path-diversity target model: {model_config.label}")

    print(f"Loading GSM8K test split: {args.n_problems} problems from offset {args.problem_offset}")
    ds = datasets.load_dataset("gsm8k", "main", split="test")
    problems = list(ds.select(range(args.problem_offset, args.problem_offset + args.n_problems)))

    conditions = (
        ConditionSpec(label="baseline_T1.0", model_path=None, temperature=1.0),
        ConditionSpec(label="baseline_T1.5", model_path=None, temperature=1.5),
        ConditionSpec(label="trained_T1.0", model_path=args.checkpoint_path, temperature=1.0),
    )

    cell_rows: list[dict] = []
    all_completions: list[dict] = []

    for cond in conditions:
        print(f"\n--- {cond.label} ({'trained' if cond.model_path else 'vanilla'}) ---", flush=True)
        backend = build_instruct_backend(model_path=cond.model_path, config=model_config)
        try:
            for idx, problem in enumerate(problems):
                t0 = time.time()
                completions = await sample_completions(
                    backend,
                    problem["question"],
                    cond=cond,
                    k=args.k,
                    max_new_tokens=args.max_new_tokens,
                    seed_base=args.seed + int(cond.temperature * 1000),
                    concurrency=args.concurrency,
                )
                gen_elapsed = time.time() - t0

                results = score_completions(completions=completions, gt_answer=problem["answer"])
                passed_indices = [i for i, (ok, _) in enumerate(results) if ok]
                passed_completions = [completions[i] for i in passed_indices]

                diversity = compute_cell(
                    cond.label, args.problem_offset + idx, passed_completions
                )
                c = len(passed_indices)
                print(
                    f"  Q{args.problem_offset + idx}: {c}/{args.k} pass  "
                    f"distinct_text={diversity.n_distinct_text}/{c}  "
                    f"distinct_path={diversity.n_distinct_path}/{c}  "
                    f"gen={gen_elapsed:.1f}s",
                    flush=True,
                )

                cell_rows.append({
                    "condition": cond.label,
                    "temperature": cond.temperature,
                    "problem_idx": args.problem_offset + idx,
                    "k": args.k,
                    "n_correct": c,
                    "n_distinct_text": diversity.n_distinct_text,
                    "n_distinct_path": diversity.n_distinct_path,
                    "gen_elapsed_seconds": gen_elapsed,
                })
                for i, (ok, reason) in enumerate(results):
                    all_completions.append({
                        "condition": cond.label,
                        "problem_idx": args.problem_offset + idx,
                        "sample_idx": i,
                        "passed": bool(ok),
                        "fail_reason": reason,
                        "completion": completions[i],
                    })
        finally:
            try:
                backend.close()
            except Exception:
                pass

    df = pd.DataFrame(cell_rows)
    df.to_csv(out / "data" / "path_diversity_per_cell.csv", index=False)
    with (out / "data" / "path_diversity_completions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_completions:
            f.write(json.dumps(r) + "\n")

    # Per-condition summary with bootstrap CIs
    cond_summaries = []
    for cond in conditions:
        sub = df[df["condition"] == cond.label]
        if sub.empty:
            continue
        paths = sub["n_distinct_path"].tolist()
        correct = sub["n_correct"].tolist()
        lo_path, hi_path = _bootstrap_ci(paths, B=2000, seed=42)
        cond_summaries.append({
            "condition": cond.label,
            "n_problems": len(sub),
            "mean_n_correct": float(np.mean(correct)),
            "mean_distinct_path": float(np.mean(paths)),
            "distinct_path_95_lo": lo_path,
            "distinct_path_95_hi": hi_path,
            "median_distinct_path": float(np.median(paths)),
        })
    sumdf = pd.DataFrame(cond_summaries)
    sumdf.to_csv(out / "data" / "path_diversity_summary.csv", index=False)

    print("\n=== Per-condition summary ===")
    print(sumdf.to_string(index=False))

    # Comparison table: trained vs each vanilla condition
    pivot = df.pivot(index="problem_idx", columns="condition", values="n_distinct_path").reset_index()
    pivot.to_csv(out / "data" / "path_diversity_pivot.csv", index=False)

    if {"baseline_T1.0", "trained_T1.0"}.issubset(pivot.columns):
        gap_b10 = (pivot["trained_T1.0"] - pivot["baseline_T1.0"]).tolist()
        gap_b15 = (pivot["trained_T1.0"] - pivot["baseline_T1.5"]).tolist() if "baseline_T1.5" in pivot.columns else []
        gap_b10_lo, gap_b10_hi = _bootstrap_ci(gap_b10, seed=43)
        if gap_b15:
            gap_b15_lo, gap_b15_hi = _bootstrap_ci(gap_b15, seed=44)
        else:
            gap_b15_lo = gap_b15_hi = float("nan")
        wins_vs_b10 = sum(1 for g in gap_b10 if g > 0)
        ties_vs_b10 = sum(1 for g in gap_b10 if g == 0)
        losses_vs_b10 = sum(1 for g in gap_b10 if g < 0)
        wins_vs_b15 = sum(1 for g in gap_b15 if g > 0) if gap_b15 else None
        ties_vs_b15 = sum(1 for g in gap_b15 if g == 0) if gap_b15 else None
        losses_vs_b15 = sum(1 for g in gap_b15 if g < 0) if gap_b15 else None

        print(f"\ntrained vs baseline_T1.0: mean gap {np.mean(gap_b10):+.2f}  "
              f"95% CI [{gap_b10_lo:+.2f}, {gap_b10_hi:+.2f}]  "
              f"wins/ties/losses = {wins_vs_b10}/{ties_vs_b10}/{losses_vs_b10}")
        if gap_b15:
            print(f"trained vs baseline_T1.5: mean gap {np.mean(gap_b15):+.2f}  "
                  f"95% CI [{gap_b15_lo:+.2f}, {gap_b15_hi:+.2f}]  "
                  f"wins/ties/losses = {wins_vs_b15}/{ties_vs_b15}/{losses_vs_b15}")

    # Markdown report
    lines = [
        "# BEE v0.4.2 — Solution-path diversity (GSM8K)",
        "",
        f"Target model: `{model_config.instruct_model_id}`.",
        "",
        f"GSM8K test problems {args.problem_offset}-{args.problem_offset + args.n_problems - 1}. "
        f"Conditions: vanilla T=1.0, vanilla T=1.5, trained T=1.0. k={args.k} per cell.",
        "",
        "Headline metric: among the correct samples in each cell, how many DISTINCT "
        "calculation paths (sorted multisets of intermediate numbers) appear.",
        "",
        "## Per-condition summary (mean across problems)",
        "",
        "| condition | n_problems | mean correct/k | mean distinct_path | 95% CI | median |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in cond_summaries:
        lines.append(
            f"| `{s['condition']}` | {s['n_problems']} | {s['mean_n_correct']:.2f} | "
            f"**{s['mean_distinct_path']:.2f}** | "
            f"[{s['distinct_path_95_lo']:.2f}, {s['distinct_path_95_hi']:.2f}] | "
            f"{s['median_distinct_path']:.1f} |"
        )
    if {"baseline_T1.0", "trained_T1.0"}.issubset(pivot.columns):
        lines.extend([
            "",
            "## Paired comparison: trained vs each vanilla condition",
            "",
            "| comparison | mean gap | 95% CI | wins / ties / losses (n problems) |",
            "| --- | --- | --- | --- |",
            f"| trained_T1.0 - baseline_T1.0 | {np.mean(gap_b10):+.2f} | "
            f"[{gap_b10_lo:+.2f}, {gap_b10_hi:+.2f}] | "
            f"{wins_vs_b10} / {ties_vs_b10} / {losses_vs_b10} |",
        ])
        if gap_b15:
            lines.append(
                f"| trained_T1.0 - baseline_T1.5 | {np.mean(gap_b15):+.2f} | "
                f"[{gap_b15_lo:+.2f}, {gap_b15_hi:+.2f}] | "
                f"{wins_vs_b15} / {ties_vs_b15} / {losses_vs_b15} |"
            )
        lines.extend([
            "",
            "## Interpretation",
            "",
            "- If `trained - baseline_T1.0` 95% CI excludes 0 with positive sign: the "
            "v0.4 LoRA increases solution-path diversity beyond the vanilla T=1.0 reference.",
            "- If `trained - baseline_T1.5` 95% CI excludes 0 with positive sign: the "
            "training effect is **not** reachable by temperature scaling alone.",
            "- If both CIs include 0: the path-diversity gap observed in the P0 smoke "
            "(+1.8 on 5 problems) does not survive a 25-problem replication; report null.",
        ])
    lines.extend([
        "",
        "## Per-problem pivot",
        "",
        "See `data/path_diversity_pivot.csv` for problem-by-problem distinct-path counts under each condition.",
    ])
    write_markdown(out / "reports" / "bee_v0_4_2_path_diversity_report.md", lines)
    return 0


def main() -> int:
    load_dotenv()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
