"""Tinker backend for BEE v0.3.

Wraps `tinker.ServiceClient().create_sampling_client(base_model=...)` and reuses
its `compute_logprobs` and `sample` primitives to drive Lane A and Lane B
without local GPU.

`tinker` is imported lazily so the rest of the v0.3 pipeline is usable on
machines that don't have the SDK installed.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from rng_bias.backends import BackendModelSpec

if TYPE_CHECKING:  # pragma: no cover
    pass


_RENDERER_FALLBACK = {
    "Qwen/Qwen3-8B": "qwen3_chat",
    "Qwen/Qwen3-8B-Base": None,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen3_instruct",
    "Qwen/Qwen3-30B-A3B-Base": None,
    "meta-llama/Llama-3.1-8B": None,
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_chat",
    "meta-llama/Llama-3.1-70B": None,
    "meta-llama/Llama-3.3-70B-Instruct": "llama3_chat",
}


def _require_tinker_api_key() -> str:
    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise RuntimeError(
            "TinkerBackend requires the TINKER_API_KEY environment variable. "
            "Set it via .env or your shell before running v0.3 against Tinker."
        )
    return key


def _load_tinker() -> Any:
    try:
        import tinker  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TinkerBackend requires the `tinker` package. Install with `pip install tinker`."
        ) from exc
    return tinker


def _safe_renderer_name(model_id: str) -> str | None:
    try:
        from tinker_cookbook import model_info  # type: ignore

        return model_info.get_recommended_renderer_name(model_id)
    except Exception:
        return _RENDERER_FALLBACK.get(model_id)


class TinkerBackend:
    backend_id = "tinker"

    def __init__(
        self,
        spec: BackendModelSpec,
        *,
        max_retries: int | None = None,
        service_client: Any = None,
        model_path: str | None = None,
    ) -> None:
        _require_tinker_api_key()
        self.spec = spec
        self.model_id = spec.model_id
        self.pair_id = spec.pair_id
        self.model_status = spec.status
        self.family = spec.family
        self.model_path = model_path
        tinker = _load_tinker()
        if service_client is None:
            kwargs: dict[str, Any] = {}
            if max_retries is not None:
                kwargs["max_retries"] = max_retries
            service_client = tinker.ServiceClient(**kwargs)
        self._service = service_client
        if model_path is not None:
            self._sampling = service_client.create_sampling_client(model_path=model_path)
        else:
            self._sampling = service_client.create_sampling_client(base_model=spec.model_id)
        self._tokenizer = self._sampling.get_tokenizer()
        self._renderer_name = _safe_renderer_name(spec.model_id)
        self._ModelInput = tinker.ModelInput
        self._SamplingParams = tinker.SamplingParams

    def _encode(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if hasattr(ids, "ids"):
            ids = ids.ids
        return [int(token) for token in ids]

    def candidate_logprobs(self, prompt: str, candidates: list[str]) -> np.ndarray:
        prompt_ids = self._encode(prompt)
        if not prompt_ids:
            raise ValueError("Prompts must tokenize to at least one token for continuation scoring.")
        futures = []
        candidate_id_lists: list[list[int]] = []
        for candidate in candidates:
            candidate_ids = self._encode(candidate)
            if not candidate_ids:
                raise ValueError(f"Candidate {candidate!r} tokenizes to empty token list.")
            candidate_id_lists.append(candidate_ids)
            mi = self._ModelInput.from_ints(prompt_ids + candidate_ids)
            futures.append(self._sampling.compute_logprobs(prompt=mi))
        scores: list[float] = []
        for future, candidate_ids in zip(futures, candidate_id_lists, strict=True):
            response = future.result() if hasattr(future, "result") else future
            logprobs = self._extract_token_logprobs(response, len(prompt_ids), len(candidate_ids))
            scores.append(float(sum(logprobs)))
        return np.asarray(scores, dtype=np.float64)

    @staticmethod
    def _extract_token_logprobs(response: Any, prompt_len: int, candidate_len: int) -> list[float]:
        if isinstance(response, list):
            per_token = response
        else:
            per_token = None
            for attr in ("prompt_logprobs", "logprobs", "token_logprobs"):
                value = getattr(response, attr, None)
                if value is not None:
                    per_token = list(value)
                    break
        if per_token is None:
            raise RuntimeError(
                "TinkerBackend: compute_logprobs response did not expose a per-token logprob list."
            )
        # Tinker returns per_token[i] = log P(token_i | tokens_0..i-1) with per_token[0] = None.
        start = prompt_len
        stop = start + candidate_len
        slice_ = per_token[start:stop]
        if len(slice_) < candidate_len:
            slice_ = per_token[-candidate_len:] if candidate_len <= len(per_token) else per_token
        out: list[float] = []
        for value in slice_:
            if value is None:
                continue
            out.append(float(value))
        if not out:
            raise RuntimeError("TinkerBackend: compute_logprobs returned no usable token logprobs.")
        return out

    def sample(
        self,
        prompt: str,
        *,
        n_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        batch_size: int,
    ) -> list[dict[str, object]]:
        prompt_ids = self._encode(prompt)
        mi = self._ModelInput.from_ints(prompt_ids)
        rows: list[dict[str, object]] = []
        offset = 0
        start = time.time()
        while offset < n_samples:
            batch_n = min(batch_size, n_samples - offset)
            params = self._SamplingParams(
                max_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k) if int(top_k) > 0 else 0,
                seed=int(seed + offset),
            )
            response = self._sampling.sample(prompt=mi, num_samples=batch_n, sampling_params=params)
            if hasattr(response, "result"):
                response = response.result()
            for local_idx, seq in enumerate(response.sequences):
                token_ids = [int(token) for token in seq.tokens]
                text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
                rows.append(
                    {
                        "sample_id": offset + local_idx,
                        "sample_seed": int(seed + offset),
                        "token_ids": token_ids,
                        "n_generated_tokens": len(token_ids),
                        "text": text,
                        "stop_reason": getattr(seq, "stop_reason", ""),
                    }
                )
            offset += batch_n
        elapsed = max(time.time() - start, 1e-9)
        for row in rows:
            row["batch_elapsed_seconds"] = elapsed
        return rows

    def apply_chat_template(self, system: str, user: str) -> str:
        if self.model_status == "base":
            return f"System: {system}\nUser: {user}\nAssistant:"
        try:
            from tinker_cookbook.renderers import get_renderer  # type: ignore

            renderer = get_renderer(self._renderer_name or self.model_id, self._tokenizer)
            example, _ = renderer.build_supervised_example(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            tokens = example.to_ints() if hasattr(example, "to_ints") else list(example)
            return self._tokenizer.decode(tokens, skip_special_tokens=False)
        except Exception:
            return f"System: {system}\nUser: {user}\nAssistant:"

    def model_meta(self) -> dict[str, object]:
        tokenizer_name = self._tokenizer.__class__.__name__
        return {
            "model_id": self.model_id,
            "pair_id": self.pair_id,
            "model_family": self.family,
            "model_status": self.model_status,
            "backend_id": self.backend_id,
            "tokenizer": tokenizer_name,
            "parameter_count": -1,
            "decoding_backend": "tinker.SamplingClient",
            "sampler_implementation": "tinker.SamplingClient.sample with temperature, top_p, top_k, seed",
            "seed_policy": "tinker SamplingParams seed per batch",
            "scoring_backend": "tinker.SamplingClient.compute_logprobs",
            "renderer_name": self._renderer_name or "",
            "model_path": self.model_path or "",
        }

    def close(self) -> None:
        self._sampling = None  # release reference; SDK manages remote teardown
        self._service = None


__all__ = ["TinkerBackend"]
