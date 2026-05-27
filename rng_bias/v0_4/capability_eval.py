"""Stage-3 capability evaluation: MMLU / IFEval / GSM8K / HumanEval / Self-BLEU.

Lightweight implementations using direct Tinker SamplingClient calls. Each
benchmark loads its eval data from HuggingFace at runtime; results land in a
single CSV row per checkpoint for the Stage-3 report.

This module is intentionally minimal — full benchmark fidelity is not the
point of v0.4. The point is to detect *regressions* from the original Instruct
to the RL'd checkpoint. We need consistent measurement on both sides, not
peer-reviewed benchmark accuracy.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rng_bias.modeling import load_dotenv


@dataclass(frozen=True)
class CapabilityEvalConfig:
    output_dir: Path
    backend_label: str
    mmlu_n: int = 200
    ifeval_n: int = 100
    gsm8k_n: int = 50
    humaneval_n: int = 50
    selfbleu_n_prompts: int = 50
    selfbleu_samples_per_prompt: int = 5
    seed: int = 2026


_LETTER_RE = re.compile(r"\b([A-D])\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_letter(text: str) -> str | None:
    match = _LETTER_RE.search(text)
    return match.group(1) if match else None


def _parse_number(text: str) -> float | None:
    match = _NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


async def _sample_one(sampler: Any, tinker_mod: Any, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
    tokens = sampler.get_tokenizer().encode(prompt, add_special_tokens=False)
    if hasattr(tokens, "ids"):
        tokens = tokens.ids
    tokens = [int(t) for t in tokens]
    mi = tinker_mod.ModelInput.from_ints(tokens)
    params = tinker_mod.SamplingParams(
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=1.0,
        top_k=0,
        seed=int(seed),
    )
    future = sampler.sample(prompt=mi, num_samples=1, sampling_params=params)
    if hasattr(future, "result"):
        response = future.result()
    else:
        response = future
    seq = response.sequences[0]
    return sampler.get_tokenizer().decode(seq.tokens, skip_special_tokens=True)


def _ngram_set(text: str, n: int = 3) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _self_bleu_proxy(samples: list[str]) -> float:
    """Proxy for diversity: 1 - mean Jaccard of 3-gram sets across pairs.

    Higher = more diverse. We use this instead of corpus-BLEU because it doesn't
    need a tokenizer-specific reference and reads as a single scalar per prompt.
    """
    if len(samples) < 2:
        return float("nan")
    sets = [_ngram_set(s, 3) for s in samples]
    pair_scores: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a and not b:
                pair_scores.append(0.0)
                continue
            union = a | b
            if not union:
                pair_scores.append(0.0)
                continue
            inter = a & b
            jaccard = len(inter) / len(union)
            pair_scores.append(1.0 - jaccard)
    return float(np.mean(pair_scores)) if pair_scores else float("nan")


async def _evaluate_mmlu(sampler: Any, tinker_mod: Any, n: int, seed: int) -> dict[str, object]:
    """Evaluate MMLU subset (200 questions stratified by subject by default).

    Returns {"score_pct": float, "n": int}. If `datasets` is unavailable or the
    download fails, returns NaN with a status note.
    """
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("cais/mmlu", "all", split="test")
        subjects = sorted(set(ds["subject"]))
        rng = np.random.default_rng(seed)
        per_subject = max(1, n // len(subjects))
        indices: list[int] = []
        for subject in subjects:
            subj_idx = [i for i, s in enumerate(ds["subject"]) if s == subject]
            chosen = rng.choice(subj_idx, size=min(per_subject, len(subj_idx)), replace=False)
            indices.extend(int(i) for i in chosen)
        indices = indices[:n]
    except Exception as exc:  # noqa: BLE001
        return {"score_pct": float("nan"), "n": 0, "status": f"unavailable: {type(exc).__name__}: {exc}"}

    async def evaluate_one(idx: int) -> bool:
        row = ds[idx]
        question = row["question"]
        choices = row["choices"]
        answer_idx = int(row["answer"])
        letters = ["A", "B", "C", "D"]
        prompt = (
            f"Answer the following multiple choice question with a single letter (A, B, C, or D).\n\n"
            f"{question}\n"
        )
        for letter, choice in zip(letters, choices, strict=True):
            prompt += f"{letter}. {choice}\n"
        prompt += "\nAnswer:"
        text = await _sample_one(sampler, tinker_mod, prompt, max_tokens=4, temperature=0.0, seed=seed + idx)
        parsed = _parse_letter(text)
        return parsed == letters[answer_idx]

    results = await asyncio.gather(*[evaluate_one(i) for i in indices])
    correct = sum(1 for r in results if r)
    return {"score_pct": 100.0 * correct / max(len(results), 1), "n": int(len(results)), "status": "ok"}


async def _evaluate_gsm8k(sampler: Any, tinker_mod: Any, n: int, seed: int) -> dict[str, object]:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("openai/gsm8k", "main", split="test")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    except Exception as exc:  # noqa: BLE001
        return {"score_pct": float("nan"), "n": 0, "status": f"unavailable: {type(exc).__name__}: {exc}"}

    async def evaluate_one(idx: int) -> bool:
        row = ds[int(idx)]
        question = row["question"]
        answer_text = row["answer"]
        ref = answer_text.split("####")[-1].strip().replace(",", "")
        ref_val = _parse_number(ref)
        prompt = (
            f"Solve the following math problem. Output only the final numeric answer on the last line "
            f"after '#### '.\n\n{question}\n\n"
        )
        text = await _sample_one(sampler, tinker_mod, prompt, max_tokens=256, temperature=0.0, seed=seed + int(idx))
        marker = text.rsplit("####", 1)
        last = marker[-1] if len(marker) > 1 else text
        parsed = _parse_number(last)
        if parsed is None or ref_val is None:
            return False
        return math.isclose(parsed, ref_val, rel_tol=1e-3, abs_tol=1e-3)

    results = await asyncio.gather(*[evaluate_one(i) for i in indices])
    correct = sum(1 for r in results if r)
    return {"score_pct": 100.0 * correct / max(len(results), 1), "n": int(len(results)), "status": "ok"}


async def _evaluate_ifeval(sampler: Any, tinker_mod: Any, n: int, seed: int) -> dict[str, object]:
    """IFEval subset: a tiny hand-coded instruction-following check.

    Real IFEval requires the official verifier with ~25 distinct checkers; this
    implementation covers a small subset (count-words, contains-keyword, ends-with)
    sufficient for *regression* detection, which is all v0.4 needs.
    """
    rng = np.random.default_rng(seed)
    checks: list[tuple[str, Any]] = [
        (
            "Reply with exactly 8 words. No more, no less.",
            lambda text: len(re.findall(r"\w+", text)) == 8,
        ),
        (
            "Write your reply so that it ends with the word 'banana'. (No punctuation after.)",
            lambda text: text.strip().lower().rstrip(".,!?").endswith("banana"),
        ),
        (
            "Reply containing the keyword 'parsimony' somewhere in your answer.",
            lambda text: "parsimony" in text.lower(),
        ),
        (
            "Reply with a single sentence that is all uppercase.",
            lambda text: text.strip().split("\n")[0].isupper() if text.strip() else False,
        ),
    ]
    indices = list(range(min(n, len(checks) * 25)))

    async def evaluate_one(idx: int) -> bool:
        instr, fn = checks[idx % len(checks)]
        prompt = f"{instr}\n\nResponse:"
        text = await _sample_one(sampler, tinker_mod, prompt, max_tokens=64, temperature=0.0, seed=seed + idx)
        try:
            return bool(fn(text))
        except Exception:
            return False

    results = await asyncio.gather(*[evaluate_one(i) for i in indices])
    correct = sum(1 for r in results if r)
    return {"score_pct": 100.0 * correct / max(len(results), 1), "n": int(len(results)), "status": "ok"}


async def _evaluate_self_bleu(sampler: Any, tinker_mod: Any, n_prompts: int, samples_per_prompt: int, seed: int) -> dict[str, object]:
    prompts = [
        "Write a short opening sentence for a story about a forgotten lighthouse.",
        "Compose a single tweet describing what coffee tastes like.",
        "Describe in one sentence how it feels to walk into a library.",
        "Give a one-line description of an autumn afternoon.",
        "Write a single sentence about losing a key.",
    ]
    # Repeat prompts cyclically until we have n_prompts.
    expanded = [prompts[i % len(prompts)] for i in range(n_prompts)]

    async def evaluate_prompt(idx: int, prompt: str) -> float:
        samples = await asyncio.gather(
            *[
                _sample_one(
                    sampler,
                    tinker_mod,
                    f"{prompt}\n\nResponse:",
                    max_tokens=80,
                    temperature=1.0,
                    seed=seed + idx * 1000 + j,
                )
                for j in range(samples_per_prompt)
            ]
        )
        return _self_bleu_proxy(samples)

    diversity = await asyncio.gather(*[evaluate_prompt(i, p) for i, p in enumerate(expanded)])
    diversity = [d for d in diversity if not math.isnan(d)]
    return {
        "diversity": float(np.mean(diversity)) if diversity else float("nan"),
        "n_prompts": int(len(expanded)),
        "samples_per_prompt": int(samples_per_prompt),
    }


async def _evaluate_all_async(sampler: Any, tinker_mod: Any, config: CapabilityEvalConfig) -> dict[str, object]:
    mmlu = await _evaluate_mmlu(sampler, tinker_mod, config.mmlu_n, config.seed)
    ifeval = await _evaluate_ifeval(sampler, tinker_mod, config.ifeval_n, config.seed + 1)
    gsm8k = await _evaluate_gsm8k(sampler, tinker_mod, config.gsm8k_n, config.seed + 2)
    selfbleu = await _evaluate_self_bleu(
        sampler, tinker_mod, config.selfbleu_n_prompts, config.selfbleu_samples_per_prompt, config.seed + 3
    )
    return {
        "MMLU": mmlu,
        "IFEval": ifeval,
        "GSM8K": gsm8k,
        "SelfBLEU": selfbleu,
    }


def run_capability_eval(sampler: Any, tinker_mod: Any, config: CapabilityEvalConfig) -> dict[str, object]:
    """Synchronous entry. Persists a CSV row in `config.output_dir`."""
    load_dotenv()
    results = asyncio.run(_evaluate_all_async(sampler, tinker_mod, config))
    output_dir = Path(config.output_dir).resolve()
    metrics_dir = output_dir / "data/bee_v0_4/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "backend_label": config.backend_label,
        "mmlu_score_pct": results["MMLU"].get("score_pct"),
        "mmlu_n": results["MMLU"].get("n"),
        "mmlu_status": results["MMLU"].get("status"),
        "ifeval_score_pct": results["IFEval"].get("score_pct"),
        "ifeval_n": results["IFEval"].get("n"),
        "gsm8k_score_pct": results["GSM8K"].get("score_pct"),
        "gsm8k_n": results["GSM8K"].get("n"),
        "gsm8k_status": results["GSM8K"].get("status"),
        "selfbleu_diversity": results["SelfBLEU"].get("diversity"),
        "selfbleu_n_prompts": results["SelfBLEU"].get("n_prompts"),
    }
    out_path = metrics_dir / "bee_v0_4_capability_eval.csv"
    new_df = pd.DataFrame([row])
    if out_path.exists() and out_path.stat().st_size > 0:
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)
    return row


__all__ = [
    "CapabilityEvalConfig",
    "run_capability_eval",
]
