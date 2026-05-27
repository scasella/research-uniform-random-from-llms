"""v0.4 decision state machine and stage gates."""
from __future__ import annotations

from dataclasses import dataclass, field


ALLOWED_TERMINAL_STATES = {
    "transfer_with_preservation",
    "transfer_with_regression",
    "narrow_task_only",
    "task_failure",
}

ALLOWED_STAGE_GATES = {
    "stage_1_passed",
    "stage_1_failed",
    "stage_2_passed",
    "stage_2_failed",
}


@dataclass(frozen=True)
class StageGateConfig:
    stage_1_min_on_task_tv_drop: float = 0.05
    stage_2_min_on_task_tv_drop: float = 0.10
    stage_2_min_trained_tasks_passing: int = 2
    stage_2_min_heldout_tv_drop: float = 0.05
    stage_2_min_heldout_tasks_passing: int = 1
    stage_2_max_invalid_rate: float = 0.30


@dataclass(frozen=True)
class FinalGateConfig:
    on_task_min_tv_drop: float = 0.10
    transfer_min_tv_drop: float = 0.05
    transfer_min_tasks_passing: int = 5
    transfer_max_tasks_passing_for_narrow: int = 1
    capability_max_regression_pp: float = 2.0
    capability_regression_breach_pp: float = 5.0


def _safe_drop(before: float, after: float) -> float:
    """How much TV dropped (positive = improvement)."""
    return float(before) - float(after)


def stage_1_gate(
    on_task_tv_before: float,
    on_task_tv_after: float,
    config: StageGateConfig = StageGateConfig(),
) -> tuple[str, dict[str, object]]:
    drop = _safe_drop(on_task_tv_before, on_task_tv_after)
    passed = drop >= config.stage_1_min_on_task_tv_drop
    evidence = {
        "on_task_tv_before": float(on_task_tv_before),
        "on_task_tv_after": float(on_task_tv_after),
        "on_task_tv_drop": float(drop),
        "threshold": float(config.stage_1_min_on_task_tv_drop),
    }
    return ("stage_1_passed" if passed else "stage_1_failed", evidence)


def stage_2_gate(
    trained_tv_before: dict[str, float],
    trained_tv_after: dict[str, float],
    heldout_tv_before: dict[str, float],
    heldout_tv_after: dict[str, float],
    invalid_rate_after: float,
    config: StageGateConfig = StageGateConfig(),
) -> tuple[str, dict[str, object]]:
    trained_drops: dict[str, float] = {}
    trained_pass: list[str] = []
    for task_id, before in trained_tv_before.items():
        after = trained_tv_after.get(task_id, before)
        drop = _safe_drop(before, after)
        trained_drops[task_id] = drop
        if drop >= config.stage_2_min_on_task_tv_drop:
            trained_pass.append(task_id)
    heldout_drops: dict[str, float] = {}
    heldout_pass: list[str] = []
    for task_id, before in heldout_tv_before.items():
        after = heldout_tv_after.get(task_id, before)
        drop = _safe_drop(before, after)
        heldout_drops[task_id] = drop
        if drop >= config.stage_2_min_heldout_tv_drop:
            heldout_pass.append(task_id)
    on_task_ok = len(trained_pass) >= config.stage_2_min_trained_tasks_passing
    transfer_ok = len(heldout_pass) >= config.stage_2_min_heldout_tasks_passing
    invalid_ok = invalid_rate_after <= config.stage_2_max_invalid_rate
    passed = on_task_ok and transfer_ok and invalid_ok
    evidence = {
        "trained_tv_drops": trained_drops,
        "trained_tasks_passing": sorted(trained_pass),
        "heldout_tv_drops": heldout_drops,
        "heldout_tasks_passing": sorted(heldout_pass),
        "invalid_rate_after": float(invalid_rate_after),
        "on_task_gate": bool(on_task_ok),
        "transfer_gate": bool(transfer_ok),
        "format_gate": bool(invalid_ok),
        "thresholds": {
            "min_on_task_tv_drop": config.stage_2_min_on_task_tv_drop,
            "min_trained_tasks_passing": config.stage_2_min_trained_tasks_passing,
            "min_heldout_tv_drop": config.stage_2_min_heldout_tv_drop,
            "min_heldout_tasks_passing": config.stage_2_min_heldout_tasks_passing,
            "max_invalid_rate": config.stage_2_max_invalid_rate,
        },
    }
    return ("stage_2_passed" if passed else "stage_2_failed", evidence)


def final_decision(
    trained_tv_before: dict[str, float],
    trained_tv_after: dict[str, float],
    heldout_tv_before: dict[str, float],
    heldout_tv_after: dict[str, float],
    capability_regressions_pp: dict[str, float],
    config: FinalGateConfig = FinalGateConfig(),
) -> tuple[str, dict[str, object]]:
    """Compute the terminal state for Stage 3.

    `capability_regressions_pp` is a dict {benchmark_name: regression_in_percentage_points}.
    Positive value = the trained model scored that many percentage points lower than
    the original Instruct on the benchmark. Negative = it improved.
    """
    trained_drops = {
        task_id: _safe_drop(before, trained_tv_after.get(task_id, before))
        for task_id, before in trained_tv_before.items()
    }
    heldout_drops = {
        task_id: _safe_drop(before, heldout_tv_after.get(task_id, before))
        for task_id, before in heldout_tv_before.items()
    }
    on_task_passing = [t for t, drop in trained_drops.items() if drop >= config.on_task_min_tv_drop]
    transfer_passing = [t for t, drop in heldout_drops.items() if drop >= config.transfer_min_tv_drop]

    headline_capability_regression = max(
        capability_regressions_pp.get("MMLU", 0.0),
        capability_regressions_pp.get("IFEval", 0.0),
    )
    any_breach = any(
        capability_regressions_pp.get(name, 0.0) > config.capability_regression_breach_pp
        for name in ("MMLU", "IFEval", "GSM8K")
    )

    evidence = {
        "trained_tv_drops": trained_drops,
        "on_task_tasks_passing": sorted(on_task_passing),
        "heldout_tv_drops": heldout_drops,
        "transfer_tasks_passing": sorted(transfer_passing),
        "capability_regressions_pp": dict(capability_regressions_pp),
        "headline_capability_regression_pp": float(headline_capability_regression),
        "thresholds": {
            "on_task_min_tv_drop": config.on_task_min_tv_drop,
            "transfer_min_tv_drop": config.transfer_min_tv_drop,
            "transfer_min_tasks_passing": config.transfer_min_tasks_passing,
            "transfer_max_tasks_passing_for_narrow": config.transfer_max_tasks_passing_for_narrow,
            "capability_max_regression_pp": config.capability_max_regression_pp,
            "capability_regression_breach_pp": config.capability_regression_breach_pp,
        },
    }

    if not on_task_passing:
        return "task_failure", evidence
    if any_breach:
        return "transfer_with_regression", evidence
    if (
        len(transfer_passing) >= config.transfer_min_tasks_passing
        and headline_capability_regression <= config.capability_max_regression_pp
    ):
        return "transfer_with_preservation", evidence
    if len(transfer_passing) <= config.transfer_max_tasks_passing_for_narrow:
        return "narrow_task_only", evidence
    if headline_capability_regression > config.capability_max_regression_pp:
        return "transfer_with_regression", evidence
    return "narrow_task_only", evidence


__all__ = [
    "ALLOWED_TERMINAL_STATES",
    "ALLOWED_STAGE_GATES",
    "StageGateConfig",
    "FinalGateConfig",
    "stage_1_gate",
    "stage_2_gate",
    "final_decision",
]
