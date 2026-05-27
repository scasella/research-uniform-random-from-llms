"""BEE v0.4 audit CLI. Checks artifact presence + decision-state validity."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rng_bias.v0_4.decision import ALLOWED_STAGE_GATES, ALLOWED_TERMINAL_STATES


@dataclass(frozen=True)
class Check:
    requirement: str
    status: str
    evidence: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def _data_dir(output_dir: Path) -> Path:
    return output_dir / "data/bee_v0_4"


def _check_exists(base: Path, filename: str, label: str) -> Check:
    path = base / filename
    if path.exists() and path.stat().st_size > 0:
        return Check(f"{label}: {filename}", "pass", f"{path} exists and is non-empty")
    return Check(f"{label}: {filename}", "fail", f"{path} missing or empty")


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _audit_table(rows: list[Check]) -> list[str]:
    lines = ["| requirement | status | evidence |", "| --- | --- | --- |"]
    for check in rows:
        evidence = check.evidence.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check.requirement} | {check.status} | {evidence} |")
    return lines


def audit_v0_4(output_dir: Path, *, mode: str) -> tuple[list[Check], bool]:
    output_dir = output_dir.resolve()
    reports = output_dir / "reports"
    metrics = _data_dir(output_dir) / "metrics"
    checks: list[Check] = []

    # Phase 0 may or may not be present; if it is, validate.
    phase_0_path = reports / "bee_v0_4_phase_0_manifest.json"
    if phase_0_path.exists():
        phase_0 = _load_json(phase_0_path)
        decision = str(phase_0.get("decision", ""))
        checks.append(
            Check(
                "phase 0 decision recognized",
                "pass" if decision in {"phase_0_viable", "phase_0_dead"} else "fail",
                f"decision={decision}",
            )
        )
        checks.append(_check_exists(metrics, "bee_v0_4_phase_0_metrics.csv", "phase_0_metrics"))

    # Stage 1
    stage1_gate = _load_json(reports / "bee_v0_4_stage_1_gate.json")
    if stage1_gate:
        status = str(stage1_gate.get("status", ""))
        checks.append(
            Check(
                "stage 1 gate status in allowed set",
                "pass" if status in {"stage_1_passed", "stage_1_failed"} else "fail",
                f"status={status}",
            )
        )
        checks.append(_check_exists(metrics, "bee_v0_4_stage_1_baseline.csv", "stage_1_baseline"))
        checks.append(_check_exists(metrics, "bee_v0_4_stage_1_post_training.csv", "stage_1_post"))

    # Stage 2
    stage2_gate = _load_json(reports / "bee_v0_4_stage_2_gate.json")
    if stage2_gate:
        status = str(stage2_gate.get("status", ""))
        checks.append(
            Check(
                "stage 2 gate status in allowed set",
                "pass" if status in {"stage_2_passed", "stage_2_failed"} else "fail",
                f"status={status}",
            )
        )
        checks.append(_check_exists(metrics, "bee_v0_4_stage_2_baseline.csv", "stage_2_baseline"))
        checks.append(_check_exists(metrics, "bee_v0_4_stage_2_post_training.csv", "stage_2_post"))

    # Stage 3
    stage3_decision = _load_json(reports / "bee_v0_4_stage_3_decision.json")
    if stage3_decision:
        status = str(stage3_decision.get("status", ""))
        checks.append(
            Check(
                "stage 3 status in terminal set",
                "pass" if status in ALLOWED_TERMINAL_STATES else "fail",
                f"status={status}",
            )
        )
        checks.append(_check_exists(metrics, "bee_v0_4_stage_3_baseline.csv", "stage_3_baseline"))

    if mode == "full":
        # All three stage reports must exist.
        for filename in [
            "bee_v0_4_stage_1_report.md",
            "bee_v0_4_stage_2_report.md",
            "bee_v0_4_stage_3_decision.md",
        ]:
            checks.append(_check_exists(reports, filename, "required report"))

    if not checks:
        checks.append(Check("any v0.4 artifact present", "fail", "no v0.4 reports or metrics found"))

    overall = all(c.passed for c in checks)
    return checks, overall


def write_audit(output_dir: Path, *, mode: str) -> bool:
    checks, complete = audit_v0_4(output_dir, mode=mode)
    output_dir = output_dir.resolve()
    metrics = _data_dir(output_dir) / "metrics"
    reports = output_dir / "reports"
    metrics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{**c.__dict__, "passed": c.passed} for c in checks])
    df.to_csv(metrics / "bee_v0_4_completion_audit.csv", index=False)
    failing = [c for c in checks if not c.passed]
    lines = [
        "# BEE v0.4 Completion Audit",
        "",
        f"Audit mode: `{mode}`.",
        f"Overall completion: `{'complete' if complete else 'incomplete'}`.",
        "",
        "## Failing Or Missing Items",
        "",
    ]
    if not failing:
        lines.append("No failing checks.")
    else:
        lines.extend(_audit_table(failing))
    lines.extend(["", "## All Checks", ""])
    lines.extend(_audit_table(checks))
    (reports / "bee_v0_4_completion_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BEE v0.4 artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    complete = write_audit(args.output_dir, mode=args.mode)
    print("complete" if complete else "incomplete")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v0_4", "write_audit"]
