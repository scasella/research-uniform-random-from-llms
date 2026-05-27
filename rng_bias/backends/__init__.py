"""Backend protocol for BEE v0.3.

A `Backend` exposes a uniform surface for two operations used by v0.3:
1. `candidate_logprobs(prompt, candidates)` — exact `log P(candidate | prompt)`
   summed over the candidate's token sequence.
2. `sample(prompt, n_samples, max_new_tokens, ...)` — sampled continuations.

Concrete backends live in `local_backend.py` (HF transformers / Modal) and
`tinker_backend.py` (Thinking Machines Tinker SDK). The factory `make_backend`
dispatches on `spec["backend_id"]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BackendModelSpec:
    model_id: str
    pair_id: str
    status: str
    family: str
    backend_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "pair_id": self.pair_id,
            "status": self.status,
            "family": self.family,
            "backend_id": self.backend_id,
        }


@runtime_checkable
class Backend(Protocol):
    backend_id: str
    model_id: str
    pair_id: str
    model_status: str
    family: str

    def candidate_logprobs(self, prompt: str, candidates: list[str]) -> np.ndarray:
        """Return shape (len(candidates),): summed log P over the candidate's tokens."""

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
        """Return n_samples rows with keys: text, token_ids, sample_id, sample_seed."""

    def apply_chat_template(self, system: str, user: str) -> str:
        """Render system+user via the model's chat template (post-trained) or identity (base)."""

    def model_meta(self) -> dict[str, object]:
        """Return registry metadata (model_id, pair_id, family, status, tokenizer, ...)."""

    def close(self) -> None:
        """Release model resources (GPU memory, remote session)."""


def make_backend(spec: BackendModelSpec, **kwargs: object) -> Backend:
    if spec.backend_id == "local":
        from rng_bias.backends.local_backend import LocalBackend

        return LocalBackend(spec, **kwargs)
    if spec.backend_id == "tinker":
        from rng_bias.backends.tinker_backend import TinkerBackend

        return TinkerBackend(spec, **kwargs)
    raise ValueError(f"Unknown backend_id: {spec.backend_id}")


__all__ = ["Backend", "BackendModelSpec", "make_backend"]
