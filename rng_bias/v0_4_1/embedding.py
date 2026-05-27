"""mpnet sentence embedding with on-disk .npz cache.

Cache key: sha256(model_name + "\\n" + text). Once a generation is encoded we
never re-encode it, so iterating on scoring/judging doesn't re-pay the embedding
cost.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def _row_key(model_name: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\n")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class MpnetEncoder:
    def __init__(self, *, model_name: str = MPNET_MODEL_NAME, device: str | None = None):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        model = self._load()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


def encode_with_cache(
    texts: list[str],
    *,
    encoder: MpnetEncoder,
    cache_path: Path,
) -> np.ndarray:
    """Encode texts, returning a stack of shape (len(texts), dim).

    Persists to `cache_path` (.npz). On a re-call with the same texts (or a
    superset), we read what's cached and only encode the missing rows.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [_row_key(encoder.model_name, t) for t in texts]
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            stored_keys = list(data.get("keys", np.array([], dtype=object)))
            stored_emb = data.get("emb")
            if stored_emb is not None and len(stored_keys) == stored_emb.shape[0]:
                for k, row in zip(stored_keys, stored_emb, strict=True):
                    cached[str(k)] = row

    missing_idx = [i for i, k in enumerate(keys) if k not in cached]
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        new_emb = encoder.encode(missing_texts)
        for idx, k, row in zip(missing_idx, [keys[i] for i in missing_idx], new_emb, strict=True):
            cached[k] = row

    out = np.stack([cached[k] for k in keys], axis=0)

    # Persist union of old + new keys
    all_keys = sorted(set(cached.keys()))
    all_emb = np.stack([cached[k] for k in all_keys], axis=0)
    np.savez_compressed(cache_path, keys=np.array(all_keys, dtype=object), emb=all_emb)
    return out


__all__ = ["MPNET_MODEL_NAME", "MpnetEncoder", "encode_with_cache"]
