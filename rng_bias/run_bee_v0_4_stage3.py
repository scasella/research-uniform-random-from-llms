"""BEE v0.4 — Stage 3 full experiment CLI.

Continues to 500 total steps on all 3 training tasks. Full eval: all 10
distribution tasks + capability suite (MMLU/IFEval/GSM8K/SelfBLEU) + the
logit-correction baseline.

Optional `--kl-sweep` runs three training arms at KL coefficients
{0.01, 0.05, 0.10} to map the uniformity-vs-capabilities trade-off.

Produces the final v0.4 decision via `final_decision`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from rng_bias.v0_4.baselines import logit_correction_panel
from rng_bias.v0_4.capability_eval import CapabilityEvalConfig, run_capability_eval
from rng_bias.v0_4.decision import FinalGateConfig, final_decision
from rng_bias.v0_4.distribution_tasks import TASKS
from rng_bias.v0_4.eval_lane_a import TaskEvalConfig
from rng_bias.v0_4.train import TrainConfig, run_training


STAGE_NAME = "stage_3"
TRAIN_TASK_IDS: tuple[str, ...] = ("random_int_1_100", "random_color", "random_fruit")
HELDOUT_TASK_IDS: tuple[str, ...] = tuple(t.task_id for t in TASKS if t.split == "heldout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEE v0.4 Stage 3 full experiment.")
    parser.add_argument("--run-name", default="v0_4_stage3")
    parser.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_stage3"))
    parser.add_argument("--n-steps", type=int, default=500)
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
        help="Optional tinker:// path of a Stage 2 checkpoint to continue from.",
    )
    parser.add_argument(
        "--kl-sweep",
        action="store_true",
        help="Run three KL-coef training arms (0.01 / 0.05 / 0.10) instead of one.",
    )
    parser.add_argument(
        "--include-capability-eval",
        action="store_true",
        default=True,
        help="Run MMLU/IFEval/GSM8K/SelfBLEU on the final checkpoint.",
    )
    parser.add_argument(
        "--include-logit-correction",
        action="store_true",
        default=True,
        help="Run the logit-correction baseline panel on the original Instruct.",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def _train_one_arm(args: argparse.Namespace, paths: StagePaths, kl_label: str) -> str | None:
    train_config = TrainConfig(
        stage=STAGE_NAME,
        output_dir=paths.output_dir,
        run_name=f"{args.run_name}_kl_{kl_label}",
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_response_tokens=args.max_response_tokens,
        train_task_ids=TRAIN_TASK_IDS,
        seed=args.seed,
        save_every_n_steps=max(50, args.n_steps // 5),
        resume_from_path=args.resume_from_path,
    )
    manifest = run_training(train_config)
    return manifest.get("final_checkpoint_path")  # type: ignore[return-value]


def main() -> int:
    load_dotenv()
    args = parse_args()
    paths = StagePaths.from_output_dir(args.output_dir)
    paths.ensure()

    eval_config = TaskEvalConfig(paraphrase_count=args.paraphrase_count)
    panel_ids = TRAIN_TASK_IDS + HELDOUT_TASK_IDS

    # 1) Baseline eval (full panel)
    baseline_csv = paths.metrics_dir / f"bee_v0_4_{STAGE_NAME}_baseline.csv"
    if args.skip_baseline and baseline_csv.exists() and baseline_csv.stat().st_size > 0:
        baseline_df = pd.read_csv(baseline_csv)
    else:
        try:
            backend = build_instruct_backend()
            try:
                baseline_df = evaluate_panel(backend, panel_ids, eval_config=eval_config)
                if args.include_logit_correction:
                    logit_correction_panel(
                        backend,
                        tasks=tuple(t for t in TASKS if t.task_id in panel_ids),
                        eval_config=eval_config,
                        output_dir=paths.output_dir,
                    )
            finally:
                try:
                    backend.close()
                except Exception:
                    pass
            baseline_df.to_csv(baseline_csv, index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Stage 3 baseline eval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    baseline_tv_all = tv_by_task(baseline_df)
    baseline_train_tv = {tid: baseline_tv_all.get(tid, float("nan")) for tid in TRAIN_TASK_IDS}
    baseline_heldout_tv = {tid: baseline_tv_all.get(tid, float("nan")) for tid in HELDOUT_TASK_IDS}

    # 2) Training: single arm or KL sweep
    kl_labels = ("005",) if not args.kl_sweep else ("001", "005", "010")
    final_paths: dict[str, str] = {}
    for label in kl_labels:
        try:
            ckpt = _train_one_arm(args, paths, label)
            if ckpt:
                final_paths[label] = ckpt
        except Exception as exc:  # noqa: BLE001
            print(f"Stage 3 training (kl={label}) failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    if not final_paths:
        print("Stage 3 produced no checkpoints; cannot evaluate.", file=sys.stderr)
        return 2

    # 3) Post-training eval (each arm). Pick the arm with best on-task TV drop
    # as the headline arm for the decision.
    arm_results: dict[str, dict[str, object]] = {}
    for label, ckpt_path in final_paths.items():
        try:
            trained = build_instruct_backend(model_path=ckpt_path)
            try:
                post_df = evaluate_panel(trained, panel_ids, eval_config=eval_config)
                capability_row = None
                if args.include_capability_eval:
                    import tinker  # local import — capability eval requires Tinker
                    capability_row = run_capability_eval(
                        trained._sampling,  # type: ignore[attr-defined]
                        tinker,
                        CapabilityEvalConfig(
                            output_dir=paths.output_dir,
                            backend_label=f"trained_kl_{label}",
                            seed=args.seed + 7,
                        ),
                    )
            finally:
                try:
                    trained.close()
                except Exception:
                    pass
            post_df.to_csv(paths.metrics_dir / f"bee_v0_4_{STAGE_NAME}_post_training_kl_{label}.csv", index=False)
            arm_results[label] = {
                "checkpoint_path": ckpt_path,
                "post_tv": tv_by_task(post_df),
                "capability_row": capability_row,
            }
        except Exception as exc:  # noqa: BLE001
            print(f"Stage 3 post-training eval (kl={label}) failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            arm_results[label] = {"error": str(exc)}

    # Pick headline arm = the one with the largest on-task TV improvement on random_int_1_100
    def _on_task_drop(arm: dict[str, object]) -> float:
        post_tv = arm.get("post_tv") or {}
        before = baseline_train_tv.get("random_int_1_100", float("nan"))
        after = post_tv.get("random_int_1_100", float("nan"))
        if isinstance(before, float) and isinstance(after, float):
            return before - after
        return float("-inf")

    valid_arms = {label: arm for label, arm in arm_results.items() if "post_tv" in arm}
    if not valid_arms:
        print("Stage 3: no arm completed post-training eval", file=sys.stderr)
        return 2
    headline_label = max(valid_arms.keys(), key=lambda k: _on_task_drop(valid_arms[k]))
    headline_arm = valid_arms[headline_label]
    post_tv_all: dict[str, float] = dict(headline_arm["post_tv"])  # type: ignore[assignment]
    post_train_tv = {tid: post_tv_all.get(tid, float("nan")) for tid in TRAIN_TASK_IDS}
    post_heldout_tv = {tid: post_tv_all.get(tid, float("nan")) for tid in HELDOUT_TASK_IDS}

    # 4) Capability regressions (relative to baseline)
    baseline_capability = None
    if args.include_capability_eval:
        # Run capability eval on the baseline Instruct too so we have a reference.
        try:
            import tinker
            base_backend = build_instruct_backend()
            try:
                baseline_capability = run_capability_eval(
                    base_backend._sampling,  # type: ignore[attr-defined]
                    tinker,
                    CapabilityEvalConfig(
                        output_dir=paths.output_dir,
                        backend_label="baseline_instruct",
                        seed=args.seed + 7,
                    ),
                )
            finally:
                try:
                    base_backend.close()
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            print(f"Stage 3 baseline capability eval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            baseline_capability = None

    capability_regressions_pp: dict[str, float] = {}
    headline_capability = headline_arm.get("capability_row")
    if baseline_capability and headline_capability:
        for key in ("mmlu_score_pct", "ifeval_score_pct", "gsm8k_score_pct"):
            before = baseline_capability.get(key)
            after = headline_capability.get(key) if isinstance(headline_capability, dict) else None
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                label = {"mmlu_score_pct": "MMLU", "ifeval_score_pct": "IFEval", "gsm8k_score_pct": "GSM8K"}[key]
                capability_regressions_pp[label] = float(before) - float(after)

    # 5) Final decision
    status, evidence = final_decision(
        baseline_train_tv,
        post_train_tv,
        baseline_heldout_tv,
        post_heldout_tv,
        capability_regressions_pp,
        FinalGateConfig(),
    )

    write_json(paths.reports_dir / f"bee_v0_4_{STAGE_NAME}_decision.json", {
        "status": status,
        "evidence": evidence,
        "headline_arm": headline_label,
        "arm_results": arm_results,
        "baseline_capability": baseline_capability,
        "capability_regressions_pp": capability_regressions_pp,
    })

    lines = [
        "# BEE v0.4 — Stage 3 Decision",
        "",
        f"Final status: `{status}`.",
        f"Target model: `{V04_TARGET_MODEL_ID}`.",
        f"Headline arm (best on-task drop): `kl={headline_label}`.",
        "",
        "## Trained-task TV drops",
        "",
        "| task_id | baseline TV | post TV | drop |",
        "| --- | --- | --- | --- |",
    ]
    for tid in TRAIN_TASK_IDS:
        b = baseline_train_tv.get(tid, float("nan"))
        a = post_train_tv.get(tid, float("nan"))
        lines.append(f"| {tid} | {b:.4f} | {a:.4f} | {b - a:+.4f} |")
    lines.extend(
        [
            "",
            "## Held-out transfer TV drops",
            "",
            "| task_id | baseline TV | post TV | drop |",
            "| --- | --- | --- | --- |",
        ]
    )
    for tid in HELDOUT_TASK_IDS:
        b = baseline_heldout_tv.get(tid, float("nan"))
        a = post_heldout_tv.get(tid, float("nan"))
        lines.append(f"| {tid} | {b:.4f} | {a:.4f} | {b - a:+.4f} |")
    if capability_regressions_pp:
        lines.extend(
            [
                "",
                "## Capability regressions (percentage points)",
                "",
                "| benchmark | baseline | post (headline arm) | regression |",
                "| --- | --- | --- | --- |",
            ]
        )
        for name in ("MMLU", "IFEval", "GSM8K"):
            if baseline_capability and headline_capability and isinstance(headline_capability, dict):
                key = {"MMLU": "mmlu_score_pct", "IFEval": "ifeval_score_pct", "GSM8K": "gsm8k_score_pct"}[name]
                before = baseline_capability.get(key)
                after = headline_capability.get(key)
                regression = capability_regressions_pp.get(name, float("nan"))
                lines.append(
                    f"| {name} | {before} | {after} | {regression:+.2f} |"
                )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "```json",
            json.dumps(evidence, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    write_markdown(paths.reports_dir / f"bee_v0_4_{STAGE_NAME}_decision.md", lines)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
