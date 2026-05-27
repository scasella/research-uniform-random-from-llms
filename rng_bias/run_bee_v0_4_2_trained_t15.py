"""BEE v0.4.2 — sanity check: trained @ T=1.5.

Completes the 2x2 condition matrix (training x temperature) by adding the
missing cell. 25 GSM8K problems x k=10 generations.

Question being answered: does the v0.4 training effect compose with high-
temperature sampling (trained_T1.5 > both trained_T1.0 and baseline_T1.5),
saturate (trained_T1.5 ~= trained_T1.0), or over-flatten (n_correct drops)?

Usage:
    python -m rng_bias.run_bee_v0_4_2_trained_t15 \\
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
from rng_bias.v0_4_2.gsm8k import sample_completions, score_completions
from rng_bias.v0_4_2.solution_diversity import compute_cell


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0.4.2 sanity check: trained @ T=1.5 on GSM8K.")
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_2_trained_t15"))
    p.add_argument("--n-problems", type=int, default=25)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--problem-offset", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


async def run() -> int:
    args = parse_args()
    out = args.output_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset("gsm8k", "main", split="test")
    problems = list(ds.select(range(args.problem_offset, args.problem_offset + args.n_problems)))

    cond = ConditionSpec(label="trained_T1.5", model_path=args.checkpoint_path, temperature=1.5)
    print(f"--- {cond.label} (trained @ T=1.5) on {args.n_problems} GSM8K problems ---", flush=True)

    cell_rows: list[dict] = []
    all_completions: list[dict] = []
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
                seed_base=args.seed + int(cond.temperature * 1000),
                concurrency=args.concurrency,
            )
            elapsed = time.time() - t0
            results = score_completions(completions=completions, gt_answer=problem["answer"])
            passed = [completions[i] for i, (ok, _) in enumerate(results) if ok]
            diversity = compute_cell(cond.label, args.problem_offset + idx, passed)
            c = len(passed)
            print(
                f"  Q{args.problem_offset + idx}: {c}/{args.k} pass  "
                f"distinct_path={diversity.n_distinct_path}/{c}  gen={elapsed:.1f}s",
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
                "gen_elapsed_seconds": elapsed,
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

    pd.DataFrame(cell_rows).to_csv(out / "data" / "trained_t15_per_cell.csv", index=False)
    with (out / "data" / "trained_t15_completions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_completions:
            f.write(json.dumps(r) + "\n")

    # Print summary
    df = pd.DataFrame(cell_rows)
    print(f"\n=== trained_T1.5 summary (n={len(df)} problems) ===")
    print(f"  mean n_correct       = {df['n_correct'].mean():.2f} / {args.k}")
    print(f"  mean distinct_path   = {df['n_distinct_path'].mean():.2f}")
    print(f"  median distinct_path = {df['n_distinct_path'].median():.1f}")
    return 0


def main() -> int:
    load_dotenv()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
