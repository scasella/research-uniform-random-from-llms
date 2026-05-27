"""Local HuggingFace backend.

Wraps `load_causal_lm` for in-process candidate scoring and sampling. The
exact candidate-scoring kernel `_candidate_logprobs_for_prompt` is defined
inline below.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from rng_bias.backends import BackendModelSpec
from rng_bias.modeling import LoadedModel, load_causal_lm


def _encode(tokenizer: object, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _candidate_logprobs_for_prompt(
    model: object,
    tokenizer: object,
    device: torch.device,
    prompt: str,
    candidates: list[str],
    batch_size: int,
) -> np.ndarray:
    prompt_ids = _encode(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("Prompts must tokenize to at least one token for continuation scoring.")

    candidate_ids = [_encode(tokenizer, candidate) for candidate in candidates]
    if any(len(ids) == 0 for ids in candidate_ids):
        raise ValueError("Every candidate must tokenize to at least one token.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    all_scores: list[float] = []
    for offset in range(0, len(candidates), batch_size):
        batch_candidate_ids = candidate_ids[offset : offset + batch_size]
        sequences = [prompt_ids + ids for ids in batch_candidate_ids]
        max_len = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, sequence in enumerate(sequences):
            input_ids[row_idx, : len(sequence)] = torch.tensor(sequence, device=device)
            attention_mask[row_idx, : len(sequence)] = 1

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        for row_idx, ids in enumerate(batch_candidate_ids):
            start = len(prompt_ids) - 1
            stop = start + len(ids)
            token_logits = logits[row_idx, start:stop, :]
            token_logprobs = F.log_softmax(token_logits.float(), dim=-1)
            target_ids = torch.tensor(ids, dtype=torch.long, device=device)
            score = token_logprobs.gather(1, target_ids[:, None]).sum().item()
            all_scores.append(score)

    return np.asarray(all_scores, dtype=np.float64)


class LocalBackend:
    backend_id = "local"

    def __init__(
        self,
        spec: BackendModelSpec,
        *,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.spec = spec
        self.model_id = spec.model_id
        self.pair_id = spec.pair_id
        self.model_status = spec.status
        self.family = spec.family
        self._loaded: LoadedModel | None = load_causal_lm(
            spec.model_id,
            device_name=device,
            dtype_name=dtype,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = self._loaded.tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    @property
    def loaded(self) -> LoadedModel:
        if self._loaded is None:
            raise RuntimeError("LocalBackend has been closed.")
        return self._loaded

    def candidate_logprobs(self, prompt: str, candidates: list[str]) -> np.ndarray:
        loaded = self.loaded
        return _candidate_logprobs_for_prompt(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            device=loaded.device,
            prompt=prompt,
            candidates=candidates,
            batch_size=min(len(candidates), 50),
        )

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
        loaded = self.loaded
        tokenizer = loaded.tokenizer
        model = loaded.model
        device = loaded.device
        tokenizer.padding_side = "left"
        rows: list[dict[str, object]] = []
        offset = 0
        start = time.time()
        while offset < n_samples:
            batch_n = min(batch_size, n_samples - offset)
            batch_seed = seed + offset
            torch.manual_seed(batch_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(batch_seed)
            prompts = [prompt] * batch_n
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_len = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k) if int(top_k) > 0 else 0,
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            for local_idx, sequence in enumerate(outputs):
                token_ids = sequence[input_len:].detach().cpu().tolist()
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                rows.append(
                    {
                        "sample_id": offset + local_idx,
                        "sample_seed": batch_seed,
                        "token_ids": [int(token_id) for token_id in token_ids],
                        "n_generated_tokens": len(token_ids),
                        "text": text,
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
        tokenizer = self.loaded.tokenizer
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"System: {system}\nUser: {user}\nAssistant:"
        return str(rendered)

    def model_meta(self) -> dict[str, object]:
        tokenizer_name = self.loaded.tokenizer.__class__.__name__
        try:
            param_count = int(self.loaded.model.num_parameters())
        except Exception:
            param_count = -1
        return {
            "model_id": self.model_id,
            "pair_id": self.pair_id,
            "model_family": self.family,
            "model_status": self.model_status,
            "backend_id": self.backend_id,
            "tokenizer": tokenizer_name,
            "parameter_count": param_count,
            "decoding_backend": "transformers.generate",
            "sampler_implementation": "torch/transformers multinomial sampling with temperature, top-p, top-k",
            "seed_policy": "torch manual seed per generation batch plus independent sample offsets",
            "scoring_backend": "rng_bias.backends.local_backend._candidate_logprobs_for_prompt",
        }

    def close(self) -> None:
        if self._loaded is None:
            return
        try:
            del self._loaded.model
        except Exception:
            pass
        self._loaded = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["LocalBackend"]
