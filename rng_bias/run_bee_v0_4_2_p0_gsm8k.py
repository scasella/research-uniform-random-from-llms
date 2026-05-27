"""BEE v0.4.2 — P0 smoke test on GSM8K.

5 GSM8K problems x (vanilla T=1.0, trained T=1.0) x k=10 generations.
Confirms infra + pass@k computation + that there is measurable headroom
between the two conditions at the per-problem level.

Usage:
    python -m rng_bias.run_bee_v0_4_2_p0_gsm8k \\
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
from rng_bias.v0_4._shared import build_instruct_backend, write_markdown
from rng_bias.v0_4_2.humaneval import ConditionSpec
from rng_bias.v0_4_2.gsm8k import (
    pass_at_k_unbiased,
    sample_completions,
    score_completions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0.4.2 P0 smoke test on GSM8K.")
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_2_p0_gsm8k"))
    p.add_argument("--n-problems", type=int, default=5)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--problem-offset", type=int, default=0)
    return p.parse_args()


async def run() -> int:
    args = parse_args()
    output_dir = args.output_dir
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    print(f"Loading GSM8K test split ({args.n_problems} problems from offset {args.problem_offset})...")
    ds = datasets.load_dataset("gsm8k", "main", split="test")
    problems = list(ds.select(range(args.problem_offset, args.problem_offset + args.n_problems)))

    conditions = (
        ConditionSpec(label="baseline_T1.0", model_path=None, temperature=1.0),
        ConditionSpec(label="trained_T1.0", model_path=args.checkpoint_path, temperature=1.0),
    )

    rows: list[dict] = []
    all_completions: list[dict] = []

    for cond in conditions:
        print(f"\n--- condition: {cond.label} ({'trained' if cond.model_path else 'vanilla'}) ---")
        backend = build_instruct_backend(model_path=cond.model_path)
        try:
            for idx, problem in enumerate(problems):
                t0 = time.time()
                completions = await sample_completions(
                    backend,
                    problem["question"],
                    cond=cond,
                    k=args.k,
                    max_new_tokens=args.max_new_tokens,
                    seed_base=args.seed,
                    concurrency=8,
                )
                gen_elapsed = time.time() - t0

                results = score_completions(
                    completions=completions, gt_answer=problem["answer"]
                )

                c = sum(1 for ok, _ in results if ok)
                reasons = [r for ok, r in results if not ok]
                p_at_1 = pass_at_k_unbiased(args.k, c, 1)
                p_at_5 = pass_at_k_unbiased(args.k, c, 5)
                p_at_k = pass_at_k_unbiased(args.k, c, args.k)
                print(
                    f"  Q{args.problem_offset + idx}: {c}/{args.k} pass  "
                    f"pass@1={p_at_1:.2f} pass@5={p_at_5:.2f} pass@{args.k}={p_at_k:.2f}  "
                    f"gen={gen_elapsed:.1f}s",
                    flush=True,
                )
                rows.append({
                    "condition": cond.label,
                    "problem_idx": args.problem_offset + idx,
                    "k": args.k,
                    "n_correct": c,
                    "pass_at_1": p_at_1,
                    "pass_at_5": p_at_5,
                    f"pass_at_{args.k}": p_at_k,
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

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "data" / "p0_gsm8k_per_problem.csv", index=False)
    with (output_dir / "data" / "p0_gsm8k_completions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_completions:
            f.write(json.dumps(r) + "\n")

    summary = df.groupby("condition").agg(
        mean_correct=("n_correct", "mean"),
        mean_pass_at_1=("pass_at_1", "mean"),
        mean_pass_at_5=("pass_at_5", "mean"),
        **{f"mean_pass_at_{args.k}": (f"pass_at_{args.k}", "mean")},
    ).reset_index()
    print("\n=== P0-GSM8K summary ===")
    print(summary.to_string(index=False))

    # Gate computation
    has_headroom = bool(((df["n_correct"] > 0) & (df["n_correct"] < df["k"])).any())
    finite_pass_k = bool(df[f"pass_at_{args.k}"].notna().all())
    nonempty_completions = bool(df["n_correct"].sum() > 0)
    gate_pass = has_headroom and finite_pass_k and nonempty_completions

    lines = [
        "# BEE v0.4.2 — P0 smoke (GSM8K)",
        "",
        f"Tasks: GSM8K test problems {args.problem_offset}-{args.problem_offset + args.n_problems - 1}.  "
        f"Conditions: vanilla T=1.0, trained T=1.0.  k={args.k}.",
        "",
        "## Per-problem table",
        "",
        f"| condition | problem_idx | n_correct/{args.k} | pass@1 | pass@5 | pass@{args.k} |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['condition']}` | {r['problem_idx']} | {r['n_correct']}/{r['k']} | "
            f"{r['pass_at_1']:.3f} | {r['pass_at_5']:.3f} | {r[f'pass_at_{r['k']}']:.3f} |"
        )
    lines.extend(["", "## Summary by condition", "",
                  "| condition | mean n_correct | mean pass@1 | mean pass@5 | mean pass@k |",
                  "| --- | --- | --- | --- | --- |"])
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
        f"- Infra OK (finite pass@k): **{finite_pass_k}**",
        f"- Headroom present (some 0 < c < k): **{has_headroom}**",
        f"- Verifier produces non-zero passes: **{nonempty_completions}**",
        "",
        f"**Gate: {'PASS - proceed to P1 knob sanity' if gate_pass else 'FAIL - need harder/easier problems'}**",
    ])
    write_markdown(output_dir / "reports" / "bee_v0_4_2_p0_gsm8k_report.md", lines)

    print(f"\n=== Gate: {'PASS' if gate_pass else 'FAIL'} ===")
    return 0 if gate_pass else 1


def main() -> int:
    load_dotenv()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
