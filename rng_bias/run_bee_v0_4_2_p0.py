"""BEE v0.4.2 — P0 smoke test.

3 HumanEval problems x (vanilla T=1.0, trained T=1.0) x k=10 completions.
Confirms: HF dataset loads, Tinker generation works, code extraction works,
verifier executes pytest-style checks, pass@k computes finite values, and
there's headroom for the comparison (i.e. neither condition is 0/10 or 10/10
on all problems).

Usage:
    python -m rng_bias.run_bee_v0_4_2_p0 \\
        --checkpoint-path tinker://.../v0_4_stage1_step_50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import datasets
import pandas as pd

from rng_bias.modeling import load_dotenv
from rng_bias.v0_4._shared import build_instruct_backend, write_json, write_markdown
from rng_bias.v0_4_2.humaneval import (
    ConditionSpec,
    pass_at_k_unbiased,
    sample_completions,
    score_completions,
)


CONDITIONS_DEFAULT = (
    ConditionSpec(label="baseline_T1.0", model_path=None, temperature=1.0),
    ConditionSpec(label="trained_T1.0", model_path="__CHECKPOINT__", temperature=1.0),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0.4.2 P0 smoke test on HumanEval.")
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_2_p0"))
    p.add_argument("--n-problems", type=int, default=3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--problem-offset", type=int, default=0,
                   help="Start index into HumanEval (use to pick problems of varying difficulty).")
    return p.parse_args()


async def run() -> int:
    args = parse_args()
    output_dir = args.output_dir
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    print(f"Loading HumanEval ({args.n_problems} problems starting at offset {args.problem_offset})...")
    ds = datasets.load_dataset("openai_humaneval", split="test")
    problems = list(ds.select(range(args.problem_offset, args.problem_offset + args.n_problems)))
    for p in problems:
        print(f"  {p['task_id']}: entry_point={p['entry_point']}")

    conditions = tuple(
        ConditionSpec(
            label=c.label,
            model_path=(args.checkpoint_path if c.model_path == "__CHECKPOINT__" else c.model_path),
            temperature=c.temperature,
            top_p=c.top_p,
            top_k=c.top_k,
        )
        for c in CONDITIONS_DEFAULT
    )

    rows: list[dict] = []
    all_completions: list[dict] = []

    for cond in conditions:
        print(f"\n--- condition: {cond.label} ({'trained' if cond.model_path else 'vanilla'}) ---")
        backend = build_instruct_backend(model_path=cond.model_path)
        try:
            for problem in problems:
                t0 = time.time()
                completions = await sample_completions(
                    backend,
                    problem["prompt"],
                    cond=cond,
                    k=args.k,
                    max_new_tokens=args.max_new_tokens,
                    seed_base=args.seed,
                    concurrency=8,
                )
                gen_elapsed = time.time() - t0

                t0 = time.time()
                results = score_completions(
                    prompt=problem["prompt"],
                    completions=completions,
                    test=problem["test"],
                    entry_point=problem["entry_point"],
                )
                eval_elapsed = time.time() - t0

                c = sum(1 for ok, _ in results if ok)
                reasons = [r for ok, r in results if not ok]
                p_at_1 = pass_at_k_unbiased(args.k, c, 1)
                p_at_5 = pass_at_k_unbiased(args.k, c, 5)
                p_at_k = pass_at_k_unbiased(args.k, c, args.k)
                print(
                    f"  {problem['task_id']}: {c}/{args.k} pass  "
                    f"pass@1={p_at_1:.2f} pass@5={p_at_5:.2f} pass@{args.k}={p_at_k:.2f}  "
                    f"gen={gen_elapsed:.1f}s eval={eval_elapsed:.1f}s",
                    flush=True,
                )
                if reasons[:3]:
                    print(f"      fail reasons (first 3): {reasons[:3]}")

                rows.append({
                    "condition": cond.label,
                    "task_id": problem["task_id"],
                    "entry_point": problem["entry_point"],
                    "k": args.k,
                    "n_correct": c,
                    "pass_at_1": p_at_1,
                    "pass_at_5": p_at_5,
                    f"pass_at_{args.k}": p_at_k,
                    "gen_elapsed_seconds": gen_elapsed,
                    "eval_elapsed_seconds": eval_elapsed,
                })
                for i, (ok, reason) in enumerate(results):
                    all_completions.append({
                        "condition": cond.label,
                        "task_id": problem["task_id"],
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

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "data" / "p0_per_problem.csv", index=False)
    with (output_dir / "data" / "p0_completions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_completions:
            f.write(json.dumps(r) + "\n")

    summary = df.groupby("condition").agg(
        mean_correct=("n_correct", "mean"),
        mean_pass_at_1=("pass_at_1", "mean"),
        mean_pass_at_5=("pass_at_5", "mean"),
        **{f"mean_pass_at_{args.k}": (f"pass_at_{args.k}", "mean")},
    ).reset_index()
    print("\n=== P0 summary ===")
    print(summary.to_string(index=False))

    lines = [
        "# BEE v0.4.2 — P0 smoke test",
        "",
        f"Tasks: HumanEval problems {args.problem_offset}-{args.problem_offset + args.n_problems - 1}.  "
        f"Conditions: vanilla T=1.0, trained T=1.0.  k={args.k}.",
        "",
        "## Per-problem table",
        "",
        f"| condition | task_id | n_correct/{args.k} | pass@1 | pass@5 | pass@{args.k} |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['condition']}` | `{r['task_id']}` | {r['n_correct']}/{r['k']} | "
            f"{r['pass_at_1']:.3f} | {r['pass_at_5']:.3f} | {r[f'pass_at_{r['k']}']:.3f} |"
        )
    lines.extend([
        "",
        "## Summary by condition",
        "",
        "| condition | mean n_correct | mean pass@1 | mean pass@5 | mean pass@k |",
        "| --- | --- | --- | --- | --- |",
    ])
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r['condition']}` | {r['mean_correct']:.2f} | "
            f"{r['mean_pass_at_1']:.3f} | {r['mean_pass_at_5']:.3f} | "
            f"{r[f'mean_pass_at_{args.k}']:.3f} |"
        )
    lines.extend([
        "",
        "## P0 gate verdict",
        "",
        "- **Infra OK**: pass@k values are all finite numbers.",
        "- **Headroom present**: at least one (condition, problem) has 0 < n_correct < k.",
        "- **Verifier executes**: at least one completion passed under each condition (or a non-trivial fraction failed for non-`empty` reasons).",
        "",
        "If all three are satisfied, proceed to P1 (knob sanity).",
    ])
    write_markdown(output_dir / "reports" / "bee_v0_4_2_p0_report.md", lines)
    return 0


def main() -> int:
    load_dotenv()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
