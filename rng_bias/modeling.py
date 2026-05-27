from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from httpx import ConnectError as HttpxConnectError
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub import snapshot_download
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedModel:
    model_id: str
    tokenizer: object
    model: object
    device: torch.device


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"Unknown dtype: {requested}")
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_causal_lm(
    model_id: str,
    device_name: str = "auto",
    dtype_name: str = "auto",
    trust_remote_code: bool = False,
) -> LoadedModel:
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = choose_device(device_name)
    dtype = choose_dtype(device, dtype_name)
    tokenizer_kwargs = {
        "token": token,
        "trust_remote_code": trust_remote_code,
    }
    model_kwargs = {
        "token": token,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    except (OSError, RequestsConnectionError, HttpxConnectError, LocalEntryNotFoundError):
        local_model_path = snapshot_download(model_id, token=token, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except (OSError, RequestsConnectionError, HttpxConnectError, LocalEntryNotFoundError):
        local_model_path = locals().get("local_model_path")
        if local_model_path is None:
            local_model_path = snapshot_download(model_id, token=token, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            **model_kwargs,
        )
    model.to(device)
    model.eval()
    return LoadedModel(model_id=model_id, tokenizer=tokenizer, model=model, device=device)


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")
