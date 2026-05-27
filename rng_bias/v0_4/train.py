"""GRPO-style training driver for BEE v0.4.

Implements a small custom policy-gradient loop over the Tinker SDK rather than
the cookbook. The reason: our reward is *group-level* (depends on the within-
group histogram of emitted values), which doesn't fit cleanly into
`ProblemEnv` / `MessageEnv` whose `step` returns a single per-rollout reward
at step-time.

The loop per step:

1. Get a SamplingClient for the current weights (always fresh after every save).
2. For each of `batch_size` groups, pick a training task and sample `group_size`
   rollouts at temperature 1.0.
3. Parse rollouts, compute within-group histogram rewards via
   `rng_bias.v0_4.grpo_env.compute_group_rewards`.
4. Center within-group advantages via `compute_group_advantages`.
5. Build Datums: token sequence = prompt_tokens + rollout_tokens; per-token
   weight = 0 on prompt tokens, the rollout's advantage on rollout tokens.
6. `tc.forward_backward_async(data=datums, loss_fn="cross_entropy")` —
   the cross-entropy loss with weights `w` and targets `y` reduces to
   `-sum(w * log p(y | context))`, i.e. the REINFORCE policy-gradient
   objective with `w` as the advantage signal.
7. `tc.optim_step_async(adam_params=AdamParams(learning_rate=lr))`.
8. Periodically save a checkpoint via `tc.save_weights_for_sampler(name=...)`
   and record the tinker:// path.

This is policy-gradient, not full PPO/GRPO with importance-sampling clipping.
For Stage 1's 50-step smoke and Stage 2's 200-step run with a small LR + tight
LoRA rank, the simplification is appropriate. Stage 3 may graduate to
`loss_fn="importance_sampling"` if drift becomes a concern.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rng_bias.modeling import load_dotenv
from rng_bias.v0_4.distribution_tasks import (
    DistributionTask,
    TRAIN_TASKS,
    get_task,
)
from rng_bias.v0_4.grpo_env import (
    RewardConfig,
    compute_group_advantages,
    compute_group_rewards,
    group_uniformity_summary,
)


DEFAULT_TRAIN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_RENDERER_FALLBACK_TEMPLATE = (
    "System: You are a uniform random sampler. Output ONLY one {item_kind} from the list below, "
    "exactly as written. No punctuation, no extra words.\n\nValid {item_kind}s: {candidate_csv}\n"
    "User: Pick a random {item_kind}.\n"
    "Assistant:"
)


@dataclass(frozen=True)
class TrainConfig:
    """Stage-aware GRPO training config."""

    stage: str
    output_dir: Path
    run_name: str
    model_name: str = DEFAULT_TRAIN_MODEL
    lora_rank: int = 16
    learning_rate: float = 2e-5
    n_steps: int = 50
    batch_size: int = 8
    group_size: int = 16
    max_response_tokens: int = 8
    temperature: float = 1.0
    train_task_ids: tuple[str, ...] = ("random_int_1_100",)
    save_every_n_steps: int = 25
    seed: int = 2026
    reward_invalid_penalty: float = 1.0
    reward_clip: float | None = 10.0
    standardize_advantages: bool = True
    resume_from_path: str | None = None

    @property
    def reward_config(self) -> RewardConfig:
        return RewardConfig(
            invalid_penalty=self.reward_invalid_penalty,
            reward_clip=self.reward_clip,
            standardize_advantages=self.standardize_advantages,
        )

    @property
    def tasks(self) -> list[DistributionTask]:
        if self.train_task_ids:
            return [get_task(tid) for tid in self.train_task_ids]
        return list(TRAIN_TASKS)


def _require_tinker() -> Any:
    try:
        import tinker  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Tinker SDK not installed. Run `pip install tinker` and set TINKER_API_KEY."
        ) from exc
    return tinker


def _require_tinker_api_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY is required for v0.4 training. Set it in .env or your shell."
        )


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(ids, "ids"):
        ids = ids.ids
    return [int(token) for token in ids]


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _result_of(awaitable_or_future: Any) -> Any:
    """Synchronously unwrap an APIFuture / concurrent.futures.Future-like object."""
    if hasattr(awaitable_or_future, "result") and callable(awaitable_or_future.result):
        return awaitable_or_future.result()
    return awaitable_or_future


async def _await_result(maybe_coro_or_future: Any) -> Any:
    """Resolve a Tinker SDK call return value to its underlying response.

    Tinker's `_async` methods are coroutines that return an `APIFuture` rather
    than the resolved value directly, so we may need *two* levels of awaiting:
    first the coroutine, then the APIFuture's `result_async()`. This helper
    walks that chain transparently.
    """
    obj = maybe_coro_or_future
    if asyncio.iscoroutine(obj):
        obj = await obj
    if hasattr(obj, "result_async"):
        return await obj.result_async()
    if hasattr(obj, "__await__"):
        return await obj
    if hasattr(obj, "result") and callable(obj.result):
        return obj.result()
    return obj


def _new_sampler(tc: Any) -> Any:
    """Synchronous wrapper around save_weights_and_get_sampling_client."""
    if hasattr(tc, "save_weights_and_get_sampling_client"):
        return _result_of(tc.save_weights_and_get_sampling_client())
    raise RuntimeError(
        "TrainingClient does not expose save_weights_and_get_sampling_client; "
        "Tinker SDK API may have changed."
    )


async def _new_sampler_async(tc: Any) -> Any:
    """Save current weights and return a fresh SamplingClient pointing at them.

    Prefer the `_async` variant of the API when available so we don't call sync
    methods from async context. Always creates a new SamplingClient — stale
    clients silently sample from outdated weights (the tinker-rl pitfall).
    """
    if hasattr(tc, "save_weights_and_get_sampling_client_async"):
        return await _await_result(tc.save_weights_and_get_sampling_client_async())
    if hasattr(tc, "save_weights_and_get_sampling_client"):
        return _result_of(tc.save_weights_and_get_sampling_client())
    raise RuntimeError(
        "TrainingClient does not expose save_weights_and_get_sampling_client*; "
        "Tinker SDK API may have changed."
    )


def _build_datum(
    tinker_mod: Any,
    prompt_tokens: list[int],
    response_tokens: list[int],
    advantage: float,
    pad_token_id: int = 0,
) -> Any:
    """Build a training Datum for policy-gradient on the response tokens.

    Tinker's `cross_entropy` loss requires `len(target_tokens) == len(model_input)`.
    Convention is next-token prediction: at position i, the model conditions on
    `model_input[:i+1]` and predicts `target_tokens[i]`. For policy gradient on a
    sampled response we want:
        loss = -sum_{i in response positions} advantage * log P(response_tok | context)
    which we achieve with:
        model_input    = prompt + response                (length P + R)
        target_tokens  = prompt[1:] + response + [pad]    (length P + R, next-token-shifted)
        weights[i]     = advantage  for i in [P-1, P+R-2]  (the response prediction positions)
        weights[i]     = 0          elsewhere
    Position P+R-1 is the trailing pad — weight 0 so it never contributes.
    """
    prompt_len = len(prompt_tokens)
    response_len = len(response_tokens)
    full = list(prompt_tokens) + list(response_tokens)
    # Next-token target sequence, same length as full.
    target_seq: list[int] = list(prompt_tokens[1:]) + list(response_tokens) + [int(pad_token_id)]
    assert len(target_seq) == len(full)
    weights = [0.0] * len(full)
    for i in range(prompt_len - 1, prompt_len + response_len - 1):
        if 0 <= i < len(weights):
            weights[i] = float(advantage)
    target_arr = np.asarray(target_seq, dtype=np.int64)
    weights_arr = np.asarray(weights, dtype=np.float32)
    mi = tinker_mod.ModelInput.from_ints(full)
    target_td = tinker_mod.TensorData.from_numpy(target_arr)
    weights_td = tinker_mod.TensorData.from_numpy(weights_arr)
    return tinker_mod.Datum(
        model_input=mi,
        loss_fn_inputs={"target_tokens": target_td, "weights": weights_td},
    )


async def _sample_group_async(
    sampler: Any,
    tinker_mod: Any,
    prompt_tokens: list[int],
    *,
    group_size: int,
    max_response_tokens: int,
    temperature: float,
    seed: int,
) -> list[list[int]]:
    """Sample `group_size` independent continuations from the current policy."""
    params = tinker_mod.SamplingParams(
        max_tokens=int(max_response_tokens),
        temperature=float(temperature),
        top_p=1.0,
        top_k=0,
        seed=int(seed),
    )
    mi = tinker_mod.ModelInput.from_ints(prompt_tokens)
    future = sampler.sample(prompt=mi, num_samples=group_size, sampling_params=params)
    response = _result_of(future)
    return [[int(t) for t in seq.tokens] for seq in response.sequences]


@dataclass
class StepLog:
    step: int
    elapsed_seconds: float
    mean_reward: float
    mean_advantage: float
    mean_invalid_rate: float
    empirical_tv_uniform_mean: float
    per_task_invalid_rate: dict[str, float] = field(default_factory=dict)
    per_task_empirical_tv: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None


async def _train_async(config: TrainConfig) -> dict[str, object]:
    load_dotenv()
    _require_tinker_api_key()
    tinker_mod = _require_tinker()

    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir).resolve()
    logs_dir = output_dir / "data/bee_v0_4/metrics"
    ckpt_dir = output_dir / "data/bee_v0_4/checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    svc = tinker_mod.ServiceClient()
    if config.resume_from_path is not None:
        if hasattr(svc, "create_training_client_from_state_async"):
            tc = await _await_result(svc.create_training_client_from_state_async(path=config.resume_from_path))
        else:
            tc = _result_of(svc.create_training_client_from_state(path=config.resume_from_path))
    else:
        if hasattr(svc, "create_lora_training_client_async"):
            tc = await _await_result(
                svc.create_lora_training_client_async(base_model=config.model_name, rank=config.lora_rank)
            )
        else:
            tc = _result_of(svc.create_lora_training_client(base_model=config.model_name, rank=config.lora_rank))

    sampler = await _new_sampler_async(tc)
    tokenizer = sampler.get_tokenizer()
    AdamParams = tinker_mod.AdamParams
    adam_params = AdamParams(learning_rate=float(config.learning_rate))

    tasks = config.tasks
    step_logs: list[StepLog] = []
    checkpoints: list[dict[str, object]] = []

    start = time.time()
    for step in range(int(config.n_steps)):
        # Collect rollouts for the batch.
        groups_meta: list[dict[str, object]] = []
        sample_seed_base = config.seed + step * 1_000_003
        for group_idx in range(int(config.batch_size)):
            task = rng.choice(tasks)
            prompt = task.render_flat()
            prompt_tokens = _encode_prompt(tokenizer, prompt)
            response_tokens_list = await _sample_group_async(
                sampler,
                tinker_mod,
                prompt_tokens,
                group_size=config.group_size,
                max_response_tokens=config.max_response_tokens,
                temperature=config.temperature,
                seed=sample_seed_base + group_idx,
            )
            texts = [_decode(tokenizer, toks) for toks in response_tokens_list]
            parsed, rewards = compute_group_rewards(texts, task, config.reward_config)
            advantages = compute_group_advantages(rewards, config.reward_config)
            groups_meta.append(
                {
                    "task": task,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens_list,
                    "texts": texts,
                    "parsed": parsed,
                    "rewards": rewards,
                    "advantages": advantages,
                }
            )

        # Build training datums and execute the forward/backward + optim step.
        batch_data: list[Any] = []
        all_rewards: list[float] = []
        all_advantages: list[float] = []
        per_task_invalid: dict[str, list[float]] = {}
        per_task_tv: dict[str, list[float]] = {}
        for group in groups_meta:
            task: DistributionTask = group["task"]  # type: ignore[assignment]
            prompt_tokens: list[int] = group["prompt_tokens"]  # type: ignore[assignment]
            response_tokens_list: list[list[int]] = group["response_tokens"]  # type: ignore[assignment]
            advantages: list[float] = group["advantages"]  # type: ignore[assignment]
            rewards: list[float] = group["rewards"]  # type: ignore[assignment]
            parsed_values = group["parsed"]  # type: ignore[assignment]
            for resp_tokens, advantage in zip(response_tokens_list, advantages, strict=True):
                if not resp_tokens:
                    continue
                batch_data.append(_build_datum(tinker_mod, prompt_tokens, resp_tokens, advantage))
            all_rewards.extend(rewards)
            all_advantages.extend(advantages)
            summary = group_uniformity_summary(parsed_values, task)
            per_task_invalid.setdefault(task.task_id, []).append(summary["invalid_rate"])
            tv = summary["empirical_tv_uniform"]
            if isinstance(tv, float) and not (tv != tv):  # NaN check
                per_task_tv.setdefault(task.task_id, []).append(tv)

        if not batch_data:
            elapsed = time.time() - start
            step_logs.append(
                StepLog(
                    step=step,
                    elapsed_seconds=elapsed,
                    mean_reward=float("nan"),
                    mean_advantage=float("nan"),
                    mean_invalid_rate=1.0,
                    empirical_tv_uniform_mean=float("nan"),
                )
            )
            sampler = await _new_sampler_async(tc)
            continue

        fb_future = tc.forward_backward_async(data=batch_data, loss_fn="cross_entropy")
        opt_future = tc.optim_step_async(adam_params=adam_params)
        await _await_result(fb_future)
        await _await_result(opt_future)

        elapsed = time.time() - start
        log = StepLog(
            step=step,
            elapsed_seconds=elapsed,
            mean_reward=float(np.mean(all_rewards)) if all_rewards else float("nan"),
            mean_advantage=float(np.mean(all_advantages)) if all_advantages else float("nan"),
            mean_invalid_rate=float(
                np.mean([np.mean(v) for v in per_task_invalid.values()])
            ) if per_task_invalid else 1.0,
            empirical_tv_uniform_mean=float(
                np.mean([np.mean(v) for v in per_task_tv.values()])
            ) if per_task_tv else float("nan"),
            per_task_invalid_rate={tid: float(np.mean(v)) for tid, v in per_task_invalid.items()},
            per_task_empirical_tv={tid: float(np.mean(v)) for tid, v in per_task_tv.items()},
        )

        # Save checkpoint if requested at this step.
        save_now = (
            ((step + 1) % max(1, config.save_every_n_steps) == 0)
            or (step + 1 == config.n_steps)
        )
        if save_now:
            if hasattr(tc, "save_weights_for_sampler_async"):
                save_result = await _await_result(
                    tc.save_weights_for_sampler_async(name=f"{config.run_name}_step_{step + 1}")
                )
            else:
                save_result = _result_of(tc.save_weights_for_sampler(name=f"{config.run_name}_step_{step + 1}"))
            ckpt_path = getattr(save_result, "path", None) or str(save_result)
            log.checkpoint_path = ckpt_path
            checkpoints.append({"step": step + 1, "path": ckpt_path})
            # Refresh sampler against the new weights.
            sampler = await _new_sampler_async(tc)
        else:
            sampler = await _new_sampler_async(tc)

        step_logs.append(log)

        # Append to JSONL log incrementally so partial runs are inspectable.
        with (logs_dir / f"bee_v0_4_{config.run_name}_train_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(log), sort_keys=True) + "\n")

    manifest = {
        "config": {**asdict(config), "output_dir": str(output_dir), "train_task_ids": list(config.train_task_ids)},
        "checkpoints": checkpoints,
        "final_checkpoint_path": checkpoints[-1]["path"] if checkpoints else None,
        "n_steps_completed": len(step_logs),
        "total_elapsed_seconds": time.time() - start,
    }
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / f"bee_v0_4_{config.run_name}_train_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_training(config: TrainConfig) -> dict[str, object]:
    """Synchronous wrapper around the async training loop."""
    return asyncio.run(_train_async(config))


__all__ = [
    "DEFAULT_TRAIN_MODEL",
    "StepLog",
    "TrainConfig",
    "run_training",
]
