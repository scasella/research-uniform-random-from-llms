from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd
import pytest

from rng_bias.v0_4.decision import (
    ALLOWED_STAGE_GATES,
    ALLOWED_TERMINAL_STATES,
    FinalGateConfig,
    StageGateConfig,
    final_decision,
    stage_1_gate,
    stage_2_gate,
)
from rng_bias.v0_4.distribution_tasks import (
    CARD_SUITS,
    COLORS,
    FRUITS,
    HELDOUT_TASKS,
    TASK_BY_ID,
    TASKS,
    TRAIN_TASKS,
    get_task,
)
from rng_bias.v0_4.grpo_env import (
    RewardConfig,
    compute_group_advantages,
    compute_group_rewards,
    group_histogram,
    group_uniformity_summary,
    parse_group,
    per_rollout_rewards,
)


# ---------------------------------------------------------------------------
# Distribution task definitions
# ---------------------------------------------------------------------------


def test_panel_has_3_train_and_7_heldout() -> None:
    assert len(TRAIN_TASKS) == 3
    assert len(HELDOUT_TASKS) == 7
    assert len(TASKS) == 10
    train_ids = {t.task_id for t in TRAIN_TASKS}
    assert train_ids == {"random_int_1_100", "random_color", "random_fruit"}


def test_every_task_has_nonempty_candidate_set() -> None:
    for task in TASKS:
        assert len(task.candidates) > 0
        assert len(set(task.candidates)) == len(task.candidates), f"duplicates in {task.task_id}"


def test_each_task_renders_and_parses_a_known_good_sample() -> None:
    samples = {
        "random_int_1_100": ("42", "42"),
        "random_color": ("blue", "blue"),
        "random_fruit": (" apple please", "apple"),
        "random_int_1_10": (" 7", "7"),
        "random_int_1_1000": ("123", "123"),
        "random_animal": ("wolf!", "wolf"),
        "random_first_name": ("Henry", "Henry"),
        "random_word": ("mountain", "mountain"),
        "random_emoji": ("🐳 ok", "🐳"),
        "random_card_suit": ("Hearts.", "Hearts"),
    }
    for task in TASKS:
        sample_text, expected = samples[task.task_id]
        parsed = task.parser(sample_text)
        assert parsed == expected, f"{task.task_id}: parsed={parsed!r} expected={expected!r}"
        prompt = task.render_flat(request_id="rid")
        assert "System:" in prompt
        assert "Assistant:" in prompt
        assert task.item_kind in prompt


def test_integer_parser_rejects_out_of_range() -> None:
    task_100 = get_task("random_int_1_100")
    assert task_100.parser("101") is None
    assert task_100.parser("0") is None
    assert task_100.parser(" 100 ") == "100"
    task_10 = get_task("random_int_1_10")
    assert task_10.parser("11") is None


def test_word_parser_is_case_insensitive_first_match() -> None:
    task = get_task("random_color")
    assert task.parser("BLUE is nice") == "blue"
    assert task.parser("Choose Red.") == "red"
    assert task.parser("foo bar baz") is None
    # First match wins
    assert task.parser("green or blue") == "green"


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


def test_uniform_group_yields_equal_rewards() -> None:
    task = get_task("random_color")
    K = len(task.candidates)
    texts = list(task.candidates)  # each value picked exactly once
    parsed, rewards = compute_group_rewards(texts, task, RewardConfig(reward_clip=None))
    assert all(v is not None for v in parsed)
    # All rewards should be equal (within fp): -log(1/K) = log(K)
    assert max(rewards) - min(rewards) < 1e-9
    assert abs(rewards[0] - math.log(K)) < 1e-9


def test_delta_group_yields_zero_reward_for_valid_rollouts() -> None:
    task = get_task("random_color")
    texts = ["blue"] * 10  # all pick the same value
    parsed, rewards = compute_group_rewards(texts, task, RewardConfig(reward_clip=None))
    assert all(v == "blue" for v in parsed)
    # Each rollout: freq = 1.0, reward = -log(1) = 0
    for r in rewards:
        assert abs(r - 0.0) < 1e-9


def test_invalid_strictly_worse_than_any_valid() -> None:
    """The reward bug from Stage 1: invalid used to beat mode-collapsed valid.
    With the corrected formula, invalid_penalty>0 guarantees invalid < valid for all groups.
    """
    task = get_task("random_int_1_100")  # |S| = 100, the original failure case
    K = 16
    texts = ["42"] * K  # worst-case valid: mode-collapsed group
    _, valid_rewards = compute_group_rewards(texts, task, RewardConfig(invalid_penalty=2.0, reward_clip=None))
    assert all(r == 0.0 for r in valid_rewards), "delta-group valid should be exactly 0"
    invalid_texts = ["zzz"] * K
    _, invalid_rewards = compute_group_rewards(invalid_texts, task, RewardConfig(invalid_penalty=2.0, reward_clip=None))
    assert all(r == -2.0 for r in invalid_rewards)
    assert all(invalid_rewards[i] < valid_rewards[i] for i in range(K))


def test_mixed_group_rewards_underrepresented_values_higher() -> None:
    task = get_task("random_color")
    # 4 rollouts: three pick "blue", one picks "red". Underrepresented (red) gets higher reward.
    texts = ["blue", "blue", "blue", "red"]
    parsed, rewards = compute_group_rewards(texts, task, RewardConfig(reward_clip=None))
    red_reward = -math.log(1 / 4)
    blue_reward = -math.log(3 / 4)
    assert abs(rewards[3] - red_reward) < 1e-9
    for i in range(3):
        assert abs(rewards[i] - blue_reward) < 1e-9
    assert rewards[3] > rewards[0]


def test_invalid_format_gets_invalid_penalty() -> None:
    task = get_task("random_color")
    texts = ["red", "this is not a color"]
    parsed, rewards = compute_group_rewards(texts, task, RewardConfig(invalid_penalty=2.5, reward_clip=None))
    assert parsed == ["red", None]
    assert rewards[1] == -2.5


def test_advantages_center_to_zero_after_group_mean_subtraction() -> None:
    task = get_task("random_color")
    texts = ["blue", "blue", "blue", "red"]
    _, rewards = compute_group_rewards(texts, task)
    advs = compute_group_advantages(rewards, RewardConfig(standardize_advantages=False))
    assert abs(sum(advs)) < 1e-9


def test_standardized_advantages_have_unit_std_when_std_positive() -> None:
    task = get_task("random_color")
    texts = ["blue", "blue", "blue", "red"]
    _, rewards = compute_group_rewards(texts, task)
    advs = compute_group_advantages(rewards, RewardConfig(standardize_advantages=True))
    arr = np.asarray(advs)
    assert abs(arr.mean()) < 1e-6
    assert abs(float(arr.std()) - 1.0) < 0.05


def test_group_uniformity_summary_reports_invalid_and_coverage() -> None:
    task = get_task("random_color")
    texts = ["blue", "red", "garbage", "garbage"]
    parsed = parse_group(texts, task)
    summary = group_uniformity_summary(parsed, task)
    assert summary["group_size"] == 4
    assert summary["valid_n"] == 2
    assert abs(summary["invalid_rate"] - 0.5) < 1e-9
    assert summary["unique_values"] == 2


# ---------------------------------------------------------------------------
# Decision states
# ---------------------------------------------------------------------------


def test_stage_1_gate_passes_on_meaningful_drop() -> None:
    status, ev = stage_1_gate(0.94, 0.85, StageGateConfig())
    assert status == "stage_1_passed"
    assert ev["on_task_tv_drop"] == pytest.approx(0.09)


def test_stage_1_gate_fails_on_small_drop() -> None:
    status, ev = stage_1_gate(0.94, 0.93, StageGateConfig())
    assert status == "stage_1_failed"
    assert ev["on_task_tv_drop"] == pytest.approx(0.01)


def test_stage_2_gate_all_three_gates_pass() -> None:
    trained_before = {"random_int_1_100": 0.94, "random_color": 0.70, "random_fruit": 0.65}
    trained_after = {"random_int_1_100": 0.70, "random_color": 0.50, "random_fruit": 0.60}  # 2/3 pass at 0.10 threshold
    heldout_before = {"random_animal": 0.60, "random_int_1_10": 0.55, "random_word": 0.62}
    heldout_after = {"random_animal": 0.50, "random_int_1_10": 0.54, "random_word": 0.61}  # 1/3 pass at 0.05
    status, ev = stage_2_gate(trained_before, trained_after, heldout_before, heldout_after, 0.10, StageGateConfig())
    assert status == "stage_2_passed", ev


def test_stage_2_gate_fails_when_invalid_rate_too_high() -> None:
    trained_before = {"random_int_1_100": 0.94, "random_color": 0.70, "random_fruit": 0.65}
    trained_after = {"random_int_1_100": 0.70, "random_color": 0.50, "random_fruit": 0.60}
    heldout_before = {"random_animal": 0.60}
    heldout_after = {"random_animal": 0.50}
    status, _ = stage_2_gate(trained_before, trained_after, heldout_before, heldout_after, 0.45, StageGateConfig())
    assert status == "stage_2_failed"


def test_stage_2_gate_fails_when_no_transfer() -> None:
    trained_before = {"random_int_1_100": 0.94, "random_color": 0.70, "random_fruit": 0.65}
    trained_after = {"random_int_1_100": 0.70, "random_color": 0.50, "random_fruit": 0.60}
    heldout_before = {"random_animal": 0.60, "random_int_1_10": 0.55, "random_word": 0.62}
    heldout_after = {"random_animal": 0.59, "random_int_1_10": 0.54, "random_word": 0.61}  # 0/3 pass at 0.05
    status, _ = stage_2_gate(trained_before, trained_after, heldout_before, heldout_after, 0.10, StageGateConfig())
    assert status == "stage_2_failed"


def test_final_decision_hits_each_terminal_state() -> None:
    # task_failure: no on-task drop
    state, _ = final_decision(
        {"random_int_1_100": 0.9}, {"random_int_1_100": 0.89},
        {"random_animal": 0.6}, {"random_animal": 0.55},
        {"MMLU": 0.0, "IFEval": 0.0},
    )
    assert state == "task_failure"

    # transfer_with_preservation: on-task drop, 5/7 heldout drop, low cap regression
    trained_before = {"random_int_1_100": 0.94, "random_color": 0.70, "random_fruit": 0.65}
    trained_after = {"random_int_1_100": 0.70, "random_color": 0.50, "random_fruit": 0.55}
    heldout_before = {f"t{i}": 0.6 for i in range(7)}
    heldout_after = {f"t{i}": (0.5 if i < 5 else 0.6) for i in range(7)}
    state, _ = final_decision(trained_before, trained_after, heldout_before, heldout_after, {"MMLU": 1.0, "IFEval": 0.5})
    assert state == "transfer_with_preservation"

    # transfer_with_regression: transfer happens, but capability breach
    state, _ = final_decision(
        trained_before, trained_after, heldout_before, heldout_after, {"MMLU": 1.0, "IFEval": 0.5, "GSM8K": 8.0},
    )
    assert state == "transfer_with_regression"

    # narrow_task_only: on-task drop, heldout doesn't transfer
    heldout_after_narrow = {f"t{i}": 0.6 for i in range(7)}  # 0/7 transfer
    state, _ = final_decision(
        trained_before, trained_after, heldout_before, heldout_after_narrow, {"MMLU": 0.5},
    )
    assert state == "narrow_task_only"


def test_final_decision_states_are_in_allowed_set() -> None:
    state, _ = final_decision(
        {"random_int_1_100": 0.5}, {"random_int_1_100": 0.5},
        {}, {}, {},
    )
    assert state in ALLOWED_TERMINAL_STATES


def test_stage_gate_statuses_are_in_allowed_set() -> None:
    s, _ = stage_1_gate(0.5, 0.5)
    assert s in ALLOWED_STAGE_GATES
    s, _ = stage_2_gate({"a": 0.5}, {"a": 0.5}, {"b": 0.5}, {"b": 0.5}, 0.0)
    assert s in ALLOWED_STAGE_GATES


# ---------------------------------------------------------------------------
# Backend protocol contract (extended TinkerBackend)
# ---------------------------------------------------------------------------


def test_tinker_backend_accepts_model_path(monkeypatch) -> None:
    """Verify TinkerBackend forwards model_path to create_sampling_client."""
    from rng_bias.backends import BackendModelSpec
    from rng_bias.backends import tinker_backend as tb

    captured: dict[str, object] = {}

    class _FakeSampling:
        def get_tokenizer(self):
            class _Tok:
                def encode(self, text, add_special_tokens=False):
                    return [1, 2, 3]

                def decode(self, ids, skip_special_tokens=True):
                    return "decoded"

            return _Tok()

    class _FakeService:
        def create_sampling_client(self, **kwargs):
            captured.update(kwargs)
            return _FakeSampling()

    class _FakeTinker:
        ModelInput = type("ModelInput", (), {"from_ints": staticmethod(lambda ids: ids)})
        SamplingParams = type("SamplingParams", (), {})

    monkeypatch.setattr(tb, "_load_tinker", lambda: _FakeTinker)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    spec = BackendModelSpec(
        model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        pair_id="qwen3_30b_a3b",
        status="post_trained",
        family="qwen3",
        backend_id="tinker",
    )

    # No model_path: should pass base_model
    captured.clear()
    _ = tb.TinkerBackend(spec, service_client=_FakeService())
    assert "base_model" in captured
    assert "model_path" not in captured

    # With model_path: should pass model_path instead
    captured.clear()
    backend = tb.TinkerBackend(spec, service_client=_FakeService(), model_path="tinker://run/weights/abc")
    assert captured.get("model_path") == "tinker://run/weights/abc"
    assert "base_model" not in captured
    meta = backend.model_meta()
    assert meta["model_path"] == "tinker://run/weights/abc"


# ---------------------------------------------------------------------------
# Logit-correction baseline
# ---------------------------------------------------------------------------


def test_logit_correction_flattens_distribution() -> None:
    """Calling logit_correction_for_task with a fake backend that returns a
    spike distribution should produce a corrected distribution close to uniform.
    """
    from rng_bias.v0_4.baselines import logit_correction_for_task
    from rng_bias.v0_4.distribution_tasks import get_task

    task = get_task("random_card_suit")  # 4 candidates

    class _SpikeBackend:
        backend_id = "fake"
        model_id = "fake/model"
        pair_id = "fake_pair"
        model_status = "post_trained"
        family = "fake"

        def candidate_logprobs(self, prompt: str, candidates: list[str]):
            # Return logprobs heavily concentrated on the first candidate of the
            # leading_space surface.
            arr = np.full(len(candidates), -10.0, dtype=np.float64)
            arr[0] = 0.0
            return arr

    result = logit_correction_for_task(_SpikeBackend(), task)
    # Corrected probs should be (approximately) uniform
    assert abs(result.corrected_probs.sum() - 1.0) < 1e-9
    assert float(result.corrected_metrics["tv_uniform"]) < 0.01
    # Original metrics should show the spike
    assert float(result.original_metrics["tv_uniform"]) > 0.5
