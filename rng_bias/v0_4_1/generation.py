"""Free-form sampling kernel for v0.4.1 cells.

Each call produces ONE response per (task, scenario, condition) — the response
is expected to contain ~20 items (the prompt asks for "Generate 20 different X").
The parsing into individual items is the caller's responsibility (see
validity_rules.extract_items).

Async fan-out follows the capability_eval.py:61-80 pattern: prepare ModelInput
once, fire `sampler.sample(num_samples=1)` futures in parallel via asyncio.gather.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from rng_bias.backends.tinker_backend import TinkerBackend
from rng_bias.v0_4_1.tasks import ProductTask, ScenarioPrompt


@dataclass(frozen=True)
class GenerationRow:
    row_id: str
    task_id: str
    scenario_id: str
    condition: str
    seed: int
    prompt_rendered: str
    response_text: str
    token_count_response: int
    backend_model_id: str
    backend_model_path: str | None
    temperature: float
    top_p: float
    max_new_tokens: int
    cad_alpha: float | None
    cad_k_candidates: int | None
    score_trained_per_tok: float | None
    score_base_per_tok: float | None
    contrastive_score: float | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_row_id(task_id: str, scenario_id: str, condition: str, seed: int) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(f"{task_id}|{scenario_id}|{condition}|{seed}".encode("utf-8"))
    return h.hexdigest()[:16]


async def _sample_response(
    backend: TinkerBackend,
    *,
    prompt_rendered: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    import tinker  # local import; backend already initializes service

    tokens = backend._encode(prompt_rendered)
    mi = tinker.ModelInput.from_ints(tokens)
    params = tinker.SamplingParams(
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=0,
        seed=int(seed),
    )
    start = time.time()
    future = backend._sampling.sample(prompt=mi, num_samples=1, sampling_params=params)
    response = future.result() if hasattr(future, "result") else future
    seq = response.sequences[0]
    token_ids = [int(t) for t in seq.tokens]
    text = backend._tokenizer.decode(token_ids, skip_special_tokens=True)
    return {
        "text": text,
        "token_ids": token_ids,
        "elapsed_seconds": time.time() - start,
    }


async def generate_cell(
    backend: TinkerBackend,
    *,
    task: ProductTask,
    scenario: ScenarioPrompt,
    condition: str,
    seed: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 2500,
    backend_model_path: str | None = None,
) -> GenerationRow:
    """Generate one response for one (task, scenario, condition) cell."""
    prompt_rendered = backend.apply_chat_template(system=scenario.system, user=scenario.user)
    sample = await _sample_response(
        backend,
        prompt_rendered=prompt_rendered,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    return GenerationRow(
        row_id=_hash_row_id(task.task_id, scenario.scenario_id, condition, seed),
        task_id=task.task_id,
        scenario_id=scenario.scenario_id,
        condition=condition,
        seed=seed,
        prompt_rendered=prompt_rendered,
        response_text=sample["text"],
        token_count_response=len(sample["token_ids"]),
        backend_model_id=backend.model_id,
        backend_model_path=backend_model_path,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        cad_alpha=None,
        cad_k_candidates=None,
        score_trained_per_tok=None,
        score_base_per_tok=None,
        contrastive_score=None,
        elapsed_seconds=float(sample["elapsed_seconds"]),
    )


async def generate_cells(
    *,
    backend: TinkerBackend,
    cells: Iterable[tuple[ProductTask, ScenarioPrompt, str, int]],
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 2500,
    backend_model_path: str | None = None,
    concurrency: int = 4,
) -> list[GenerationRow]:
    """Fan out generate_cell calls with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    cells_list = list(cells)

    async def _run(task, scenario, condition, seed):
        async with semaphore:
            return await generate_cell(
                backend,
                task=task,
                scenario=scenario,
                condition=condition,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                backend_model_path=backend_model_path,
            )

    return await asyncio.gather(*[_run(*c) for c in cells_list])


__all__ = ["GenerationRow", "generate_cell", "generate_cells"]
