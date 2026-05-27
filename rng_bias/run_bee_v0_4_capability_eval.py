"""Run capability eval on a v0.4 checkpoint and compare to the baseline Instruct.

Independent of Stage 1/2/3 — answers the "did training hurt capabilities?"
question for any tinker:// checkpoint without requiring a full stage run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from rng_bias.modeling import load_dotenv
from rng_bias.v0_4._shared import (
    StagePaths,
    build_instruct_backend,
    write_json,
    write_markdown,
    V04_TARGET_MODEL_ID,
)
from rng_bias.v0_4.capability_eval import CapabilityEvalConfig, run_capability_eval


_BENCH_LABELS = ("MMLU", "IFEval", "GSM8K", "SelfBLEU")
_SCORE_KEYS = {
    "MMLU": "mmlu_score_pct",
    "IFEval": "ifeval_score_pct",
    "GSM8K": "gsm8k_score_pct",
    "SelfBLEU": "selfbleu_diversity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capability eval on a trained v0.4 checkpoint.")
    parser.add_argument("--checkpoint-path", required=True, help="tinker:// path of the RL'd checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_capability"))
    parser.add_argument("--mmlu-n", type=int, default=200)
    parser.add_argument("--ifeval-n", type=int, default=100)
    parser.add_argument("--gsm8k-n", type=int, default=50)
    parser.add_argument("--selfbleu-n-prompts", type=int, default=30)
    parser.add_argument("--selfbleu-samples-per-prompt", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the baseline pass (reuse a prior row).")
    return parser.parse_args()


def _eval_backend(backend, *, label: str, args: argparse.Namespace) -> dict[str, object]:
    """Run capability_eval against a backend, returning the result row."""
    import tinker  # local import — only needed for the actual eval

    config = CapabilityEvalConfig(
        output_dir=args.output_dir,
        backend_label=label,
        mmlu_n=args.mmlu_n,
        ifeval_n=args.ifeval_n,
        gsm8k_n=args.gsm8k_n,
        selfbleu_n_prompts=args.selfbleu_n_prompts,
        selfbleu_samples_per_prompt=args.selfbleu_samples_per_prompt,
        seed=args.seed,
    )
    return run_capability_eval(backend._sampling, tinker, config)


def main() -> int:
    load_dotenv()
    args = parse_args()
    paths = StagePaths.from_output_dir(args.output_dir)
    paths.ensure()

    # 1) Baseline Instruct capabilities
    baseline_row: dict[str, object] | None = None
    capability_csv = paths.metrics_dir / "bee_v0_4_capability_eval.csv"
    if args.skip_baseline and capability_csv.exists():
        df = pd.read_csv(capability_csv)
        sub = df[df["backend_label"] == "baseline_instruct"]
        if not sub.empty:
            baseline_row = sub.iloc[-1].to_dict()
            print("Reused prior baseline_instruct row.")
    if baseline_row is None:
        print(f"Evaluating baseline `{V04_TARGET_MODEL_ID}`...")
        backend = build_instruct_backend()
        try:
            baseline_row = _eval_backend(backend, label="baseline_instruct", args=args)
        finally:
            try:
                backend.close()
            except Exception:
                pass

    # 2) Trained checkpoint capabilities
    print(f"Evaluating trained checkpoint `{args.checkpoint_path}`...")
    backend = build_instruct_backend(model_path=args.checkpoint_path)
    try:
        trained_row = _eval_backend(backend, label="trained_stage1", args=args)
    finally:
        try:
            backend.close()
        except Exception:
            pass

    # 3) Compute regressions
    deltas: dict[str, dict[str, float | None]] = {}
    for label in _BENCH_LABELS:
        key = _SCORE_KEYS[label]
        before = baseline_row.get(key)
        after = trained_row.get(key)
        try:
            before_f = float(before) if before is not None else None
            after_f = float(after) if after is not None else None
        except (TypeError, ValueError):
            before_f = after_f = None
        regression: float | None
        if before_f is not None and after_f is not None and not (before_f != before_f) and not (after_f != after_f):
            # For SelfBLEU diversity, higher is better — so regression = -delta.
            if label == "SelfBLEU":
                regression = float(before_f - after_f)
            else:
                regression = float(before_f - after_f)
        else:
            regression = None
        deltas[label] = {"baseline": before_f, "trained": after_f, "regression_pp": regression}

    # 4) Reports
    summary = {
        "checkpoint_path": args.checkpoint_path,
        "baseline_label": "baseline_instruct",
        "trained_label": "trained_stage1",
        "deltas": deltas,
        "config": {**vars(args), "output_dir": str(args.output_dir)},
    }
    write_json(paths.reports_dir / "bee_v0_4_capability_summary.json", summary)

    lines = [
        "# BEE v0.4 — Capability Eval (Stage 1 checkpoint vs baseline Instruct)",
        "",
        f"Target model: `{V04_TARGET_MODEL_ID}`",
        f"Trained checkpoint: `{args.checkpoint_path}`",
        "",
        "## Per-benchmark scores",
        "",
        "| benchmark | baseline | trained | regression (pp) |",
        "| --- | --- | --- | --- |",
    ]
    for label in _BENCH_LABELS:
        d = deltas[label]
        baseline_str = f"{d['baseline']:.2f}" if d['baseline'] is not None else "NA"
        trained_str = f"{d['trained']:.2f}" if d['trained'] is not None else "NA"
        reg_str = f"{d['regression_pp']:+.2f}" if d['regression_pp'] is not None else "NA"
        lines.append(f"| {label} | {baseline_str} | {trained_str} | {reg_str} |")
    headline_regression = max(
        (d["regression_pp"] for label in ("MMLU", "IFEval") if (d := deltas[label])["regression_pp"] is not None),
        default=None,
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if headline_regression is None:
        lines.append("Capability eval was inconclusive (one or both passes returned NA).")
    elif headline_regression <= 2.0:
        lines.append(
            f"**Capabilities preserved.** Max regression on MMLU / IFEval is "
            f"`{headline_regression:+.2f}` pp, within the ≤ 2 pp threshold for "
            f"`transfer_with_preservation`."
        )
    elif headline_regression <= 5.0:
        lines.append(
            f"**Moderate capability cost.** Max regression on MMLU / IFEval is "
            f"`{headline_regression:+.2f}` pp — above the preservation threshold (2 pp) but "
            f"below the breach threshold (5 pp). Treat as `transfer_with_partial_regression`."
        )
    else:
        lines.append(
            f"**Capability regression breach.** Max regression on MMLU / IFEval is "
            f"`{headline_regression:+.2f}` pp, exceeding the 5 pp breach threshold. "
            f"This pushes the v0.4 decision to `transfer_with_regression`."
        )
    write_markdown(paths.reports_dir / "bee_v0_4_capability_report.md", lines)

    print("\n=== Capability Summary ===")
    print(json.dumps(deltas, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
