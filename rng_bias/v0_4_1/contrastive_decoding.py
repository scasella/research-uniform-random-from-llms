"""CAD-rerank: sequence-level contrastive rescoring.

True token-level Contrastive Decoding requires per-step logit manipulation
that Tinker's SamplingClient does not expose. As a tractable approximation
we generate K candidate responses from the trained model, score each under
both the trained and the base model via `candidate_logprobs`, length-normalize,
and keep the candidate with the highest

    (1 + alpha) * lp_trained_per_tok - alpha * lp_base_per_tok.

Labelled `cad_rerank` everywhere to keep the distinction from true CAD honest.

The key correctness invariant: the rendered prompt (with Instruct chat template
applied) is passed as raw text to BOTH backends' `candidate_logprobs`. Since
the Instruct and Base backends share the Qwen tokenizer, both tokenize that
string identically, so the per-token logprobs are comparable position-by-position.
"""
from __future__ import annotations

import asyncio
import hashlib
import time

import numpy as np

from rng_bias.backends.tinker_backend import TinkerBackend
from rng_bias.v0_4_1.generation import GenerationRow, _sample_response
from rng_bias.v0_4_1.tasks import ProductTask, ScenarioPrompt


def _row_id(task_id: str, scenario_id: str, condition: str, seed: int) -> str:
    h = hashlib.sha256()
    h.update(f"{task_id}|{scenario_id}|{condition}|{seed}".encode("utf-8"))
    return h.hexdigest()[:16]


async def generate_cad_rerank_cell(
    *,
    trained_backend: TinkerBackend,
    base_backend: TinkerBackend,
    task: ProductTask,
    scenario: ScenarioPrompt,
    seed: int,
    alpha: float = 0.5,
    candidate_multiplier: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 2500,
    backend_model_path: str | None = None,
) -> GenerationRow:
    """Generate K candidates from `trained_backend`, rerank against `base_backend`.

    Returns a GenerationRow for the winning candidate with CAD metadata filled.
    """
    rendered_prompt = trained_backend.apply_chat_template(
        system=scenario.system,
        user=scenario.user,
    )

    k = int(candidate_multiplier)
    if k < 1:
        raise ValueError("candidate_multiplier must be >= 1")

    start = time.time()
    samples = await asyncio.gather(
        *[
            _sample_response(
                trained_backend,
                prompt_rendered=rendered_prompt,
                seed=int(seed + i),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            for i in range(k)
        ]
    )

    cand_texts = [s["text"] for s in samples]
    cand_token_counts = np.array([max(1, len(s["token_ids"])) for s in samples], dtype=np.float64)

    lp_trained_sum = trained_backend.candidate_logprobs(rendered_prompt, cand_texts)
    lp_base_sum = base_backend.candidate_logprobs(rendered_prompt, cand_texts)

    lp_trained_per_tok = lp_trained_sum / cand_token_counts
    lp_base_per_tok = lp_base_sum / cand_token_counts

    contrastive = (1.0 + alpha) * lp_trained_per_tok - alpha * lp_base_per_tok
    best_idx = int(np.argmax(contrastive))
    best_text = cand_texts[best_idx]
    elapsed = time.time() - start

    return GenerationRow(
        row_id=_row_id(task.task_id, scenario.scenario_id, "cad_rerank", seed),
        task_id=task.task_id,
        scenario_id=scenario.scenario_id,
        condition="cad_rerank",
        seed=int(seed),
        prompt_rendered=rendered_prompt,
        response_text=best_text,
        token_count_response=int(cand_token_counts[best_idx]),
        backend_model_id=trained_backend.model_id,
        backend_model_path=backend_model_path,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        cad_alpha=float(alpha),
        cad_k_candidates=k,
        score_trained_per_tok=float(lp_trained_per_tok[best_idx]),
        score_base_per_tok=float(lp_base_per_tok[best_idx]),
        contrastive_score=float(contrastive[best_idx]),
        elapsed_seconds=float(elapsed),
    )


__all__ = ["generate_cad_rerank_cell"]
