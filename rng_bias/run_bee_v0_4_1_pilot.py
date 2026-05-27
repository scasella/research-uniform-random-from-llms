"""BEE v0.4.1 pilot driver.

Runs 1 task * 2 scenarios * 2 conditions (baseline + trained) = 4 cells of
20 items each (80 items). Embeds with mpnet, clusters with DBSCAN, judges
each item via gpt-4.1-mini through OpenRouter. Writes pilot_gate.json
with six gate checks; exit code 0 if all pass, 1 on gate failure, 2 on
infra error.

Run after `pip install -e .[v041]` (or just `pip install sentence-transformers
scikit-learn openai`) and `export OPENROUTER_API_KEY=sk-or-v1-...`.

Usage:
    python -m rng_bias.run_bee_v0_4_1_pilot \\
        --checkpoint-path tinker://... \\
        --output-dir bee_v0_4_1_pilot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from rng_bias.modeling import load_dotenv
from rng_bias.v0_4_1._shared import (
    UsefulDiversityPaths,
    build_instruct_backend,
    write_json,
    write_markdown,
)
from rng_bias.v0_4_1.clustering import auto_eps, cluster
from rng_bias.v0_4_1.embedding import MpnetEncoder, encode_with_cache
from rng_bias.v0_4_1.generation import GenerationRow, generate_cell
from rng_bias.v0_4_1.judge import RUBRIC_VERSION, score_cell
from rng_bias.v0_4_1.pilot_config import (
    DBSCAN_MAX_CLUSTERS,
    DBSCAN_MIN_CLUSTERS,
    EMBEDDING_DISTANCE_MAX,
    EMBEDDING_DISTANCE_MIN,
    HEADLINE_GAP_MIN,
    JOINT_SCORE_GAP_MIN,
    JUDGE_PARSE_RATE_MIN,
    PILOT,
    VALIDITY_FLOOR_MIN_VALID,
)
from rng_bias.v0_4_1.scoring import summarize_cell
from rng_bias.v0_4_1.tasks import TASK_BY_ID
from rng_bias.v0_4_1.validity_rules import RULE_VERSION, extract_items, is_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEE v0.4.1 pilot.")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="tinker:// path to the v0.4 LoRA-trained checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bee_v0_4_1_pilot"))
    return parser.parse_args()


async def _generate_all(*, paths: UsefulDiversityPaths, checkpoint_path: str) -> list[GenerationRow]:
    rows: list[GenerationRow] = []

    for condition in PILOT.conditions:
        model_path = checkpoint_path if condition == "trained" else None
        backend = build_instruct_backend(model_path=model_path)
        try:
            for task_id, scenario_id in PILOT.cells:
                task = TASK_BY_ID[task_id]
                scenario = next(s for s in task.scenarios if s.scenario_id == scenario_id)
                row = await generate_cell(
                    backend,
                    task=task,
                    scenario=scenario,
                    condition=condition,
                    seed=PILOT.seed,
                    temperature=PILOT.temperature,
                    top_p=PILOT.top_p,
                    max_new_tokens=PILOT.max_new_tokens,
                    backend_model_path=model_path,
                )
                rows.append(row)
                print(f"  generated [{condition}] {task_id}/{scenario_id} "
                      f"({row.token_count_response} tokens)")
        finally:
            try:
                backend.close()
            except Exception:
                pass
    return rows


def _score_cells(rows: list[GenerationRow], *, paths: UsefulDiversityPaths) -> list[dict]:
    encoder = MpnetEncoder()
    summaries: list[dict] = []

    # First pass: per-(task, scenario) DBSCAN eps from the baseline rows ONLY,
    # then frozen for the comparison. Per-scenario rather than per-task because
    # different scenarios within a task can have very different intra-scenario
    # spread (e.g., a narrow Java bug vs an open-ended perf regression).
    eps_by_cell: dict[str, float] = {}
    for row in rows:
        if row.condition != "baseline":
            continue
        items = extract_items(row.response_text, row.task_id)[: PILOT.n_per_cell]
        key = f"{row.task_id}/{row.scenario_id}"
        if len(items) >= 3:
            emb = encoder.encode(items)
            eps_by_cell[key] = auto_eps(emb)
        else:
            eps_by_cell[key] = 0.30
    write_json(paths.metrics_dir / "cluster_params.json", eps_by_cell)

    for row in rows:
        items = extract_items(row.response_text, row.task_id)
        items = items[: PILOT.n_per_cell]
        item_ids = [f"{row.row_id}_{i:02d}" for i in range(len(items))]
        validity = [is_valid(it, row.task_id) for it in items]
        valid_flags = [v.is_valid for v in validity]

        cache_path = paths.embeddings_dir / f"{row.task_id}_{row.scenario_id}_{row.condition}.npz"
        embeddings = encode_with_cache(items, encoder=encoder, cache_path=cache_path) if items else np.zeros((0, 768))

        clust = cluster(embeddings, eps=eps_by_cell[f"{row.task_id}/{row.scenario_id}"])

        judge_result = score_cell(
            task_id=row.task_id,
            scenario_id=row.scenario_id,
            condition=row.condition,
            items=list(zip(item_ids, items, strict=True)),
            shuffle_seed=PILOT.judge_shuffle_seed,
        )

        summary = summarize_cell(
            task_id=row.task_id,
            scenario_id=row.scenario_id,
            condition=row.condition,
            items=items,
            item_ids=item_ids,
            valid_flags=valid_flags,
            embeddings=embeddings,
            cluster_result=clust,
            judge_scores=judge_result.item_scores,
            parse_ok=judge_result.parse_ok,
        )
        summaries.append(asdict(summary))
        print(f"  scored   [{row.condition}] {row.task_id}/{row.scenario_id}: "
              f"items={summary.n_items}, valid={summary.n_valid}, "
              f"n_clusters={summary.n_clusters}, useful_diversity={summary.useful_diversity:.3f}, "
              f"mean_judge={summary.mean_judge_score:.2f}, parse_ok={summary.parse_ok}")

    return summaries


def _evaluate_gates(*, summaries: list[dict], rows: list[GenerationRow]) -> dict:
    df = pd.DataFrame(summaries)

    infra_ok = True  # Tinker retries aren't surfaced via the backend; treat as OK if generation completed
    validity_floor = bool(
        (df["n_valid"] >= VALIDITY_FLOOR_MIN_VALID).all()
    )
    baseline = df[df["condition"] == "baseline"]
    trained = df[df["condition"] == "trained"]

    embed_ok = bool(
        (baseline["mean_pairwise_distance"].between(EMBEDDING_DISTANCE_MIN, EMBEDDING_DISTANCE_MAX)).all()
    )
    dbscan_ok = bool(
        (baseline["n_clusters"].between(DBSCAN_MIN_CLUSTERS, DBSCAN_MAX_CLUSTERS)).all()
    )
    judge_pass = float(df["parse_ok"].mean())
    judge_ok = judge_pass >= JUDGE_PARSE_RATE_MIN

    dist_gap_by_scenario: dict[str, float] = {}
    joint_gap_by_scenario: dict[str, float] = {}
    judge_gap_by_scenario: dict[str, float] = {}
    for scenario_id in baseline["scenario_id"].unique():
        b = baseline[baseline["scenario_id"] == scenario_id]
        t = trained[trained["scenario_id"] == scenario_id]
        if b.empty or t.empty:
            continue
        dist_gap_by_scenario[scenario_id] = float(
            t["mean_pairwise_distance"].iloc[0] - b["mean_pairwise_distance"].iloc[0]
        )
        joint_gap_by_scenario[scenario_id] = float(
            t["joint_score"].iloc[0] - b["joint_score"].iloc[0]
        )
        judge_gap_by_scenario[scenario_id] = float(
            t["mean_judge_score"].iloc[0] - b["mean_judge_score"].iloc[0]
        )

    # Headline gate now uses joint_score (smoother, doesn't saturate like
    # mpnet pairwise distance on short-text tasks).
    headline_ok = any(gap >= JOINT_SCORE_GAP_MIN for gap in joint_gap_by_scenario.values())

    if headline_ok is False and validity_floor and embed_ok and judge_ok:
        recommendation = "stop_null_result"
        status = "null_result"
    elif validity_floor and embed_ok and judge_ok and headline_ok:
        recommendation = "proceed_full_sweep"
        status = "pass"
    else:
        recommendation = "iterate_per_failing_gate"
        status = "fail"

    return {
        "status": status,
        "infra_ok": infra_ok,
        "validity_floor_passed": validity_floor,
        "embedding_range_ok": embed_ok,
        "dbscan_converged": dbscan_ok,
        "dbscan_advisory_only": True,
        "judge_json_pass_rate": judge_pass,
        "judge_pass_ok": judge_ok,
        "pairwise_distance_gap": dist_gap_by_scenario,
        "joint_score_gap": joint_gap_by_scenario,
        "judge_score_gap": judge_gap_by_scenario,
        "headline_gap_visible": headline_ok,
        "headline_metric": "joint_score (mean_pairwise_distance * mean_judge_score / 5)",
        "recommendation": recommendation,
        "rubric_version": RUBRIC_VERSION,
        "rule_version": RULE_VERSION,
    }


def main() -> int:
    load_dotenv()
    args = parse_args()

    paths = UsefulDiversityPaths.from_output_dir(args.output_dir)
    paths.ensure()

    print(f"v0.4.1 pilot starting (output_dir={paths.output_dir})")
    print(f"  cells={PILOT.cells}")
    print(f"  conditions={PILOT.conditions}  checkpoint={args.checkpoint_path}")

    try:
        rows = asyncio.run(_generate_all(paths=paths, checkpoint_path=args.checkpoint_path))
    except Exception as e:  # noqa: BLE001
        print(f"[infra error during generation] {e!r}", file=sys.stderr)
        write_json(paths.metrics_dir / "pilot_gate.json", {"status": "infra_error", "error": repr(e)})
        return 2

    gen_jsonl = paths.generations_dir / "generations.jsonl"
    with gen_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), default=str) + "\n")
    write_json(
        paths.generations_dir / "manifest.json",
        {
            "config": asdict(PILOT),
            "checkpoint_path": args.checkpoint_path,
            "rubric_version": RUBRIC_VERSION,
            "rule_version": RULE_VERSION,
            "n_rows": len(rows),
        },
    )

    summaries = _score_cells(rows, paths=paths)
    pd.DataFrame(summaries).to_csv(paths.metrics_dir / "cell_summary.csv", index=False)

    gate = _evaluate_gates(summaries=summaries, rows=rows)
    write_json(paths.metrics_dir / "pilot_gate.json", gate)

    lines = [
        "# BEE v0.4.1 — Pilot report",
        "",
        f"Cells: {', '.join(f'{t}/{s}' for t, s in PILOT.cells)}",
        f"Conditions: {', '.join(PILOT.conditions)}",
        "",
        "## Cell summary",
        "",
        "| task | scenario | condition | n_valid | n_clusters | mean_dist | mean_judge | useful_diversity |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lines.append(
            f"| `{s['task_id']}` | `{s['scenario_id']}` | `{s['condition']}` | "
            f"{s['n_valid']}/{s['n_items']} | {s['n_clusters']} | "
            f"{s['mean_pairwise_distance']:.3f} | {s['mean_judge_score']:.2f} | "
            f"{s['useful_diversity']:.3f} |"
        )
    lines.extend(["", "## Gate verdict", "", "```json", json.dumps(gate, indent=2), "```"])
    write_markdown(paths.reports_dir / "bee_v0_4_1_pilot_report.md", lines)

    print("\n=== Pilot gate verdict ===")
    print(json.dumps(gate, indent=2))

    if gate["status"] == "pass":
        return 0
    if gate["status"] == "null_result":
        print("\nNote: gates pass on infra but no headline diversity gap detected. "
              "Stop and write up as a null result.", file=sys.stderr)
        return 0
    print("\nOne or more gates failed. Inspect cell_summary.csv and iterate.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
