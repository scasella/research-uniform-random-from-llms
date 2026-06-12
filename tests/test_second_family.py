"""Tests for the model-family parameterization (second-family support).

These prove, offline (no Tinker API), that:
  - the published Qwen default still resolves verbatim (reproducibility);
  - the Llama-3.1-8B pair resolves to Tinker-supported ids;
  - the backend builders thread a non-default config all the way to
    `create_sampling_client(base_model=...)`;
  - the Phase 0 diagnostic and the stage backends share one registry.
"""
from __future__ import annotations

import pytest

from rng_bias.v0_4 import _shared
from rng_bias.v0_4.model_registry import (
    DEFAULT_FAMILY_KEY,
    LLAMA_3_1_8B,
    MODEL_FAMILIES,
    QWEN3_30B_A3B,
    default_model_config,
    resolve_model_config,
)


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_default_family_is_qwen_for_reproducibility() -> None:
    """The published headline must reproduce: default == Qwen3-30B-A3B."""
    assert DEFAULT_FAMILY_KEY == "qwen3_30b_a3b"
    default = default_model_config()
    assert default is QWEN3_30B_A3B
    assert default.instruct_model_id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert default.base_model_id == "Qwen/Qwen3-30B-A3B-Base"
    # resolve(None) and the explicit key agree with the default.
    assert resolve_model_config(None) is default
    assert resolve_model_config("qwen3_30b_a3b") is default
    # Module-level back-compat constants still point at the Qwen default.
    assert _shared.V04_TARGET_MODEL_ID == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert _shared.V04_BASE_MODEL_ID == "Qwen/Qwen3-30B-A3B-Base"
    assert _shared.V04_PAIR_ID == "qwen3_30b_a3b"
    assert _shared.V04_FAMILY == "qwen3"


def test_llama_family_resolves_to_supported_ids() -> None:
    """Second family: Llama-3.1-8B-Instruct (dense, distinct from Qwen MoE)."""
    config = resolve_model_config("llama_3_1_8b")
    assert config is LLAMA_3_1_8B
    assert config.instruct_model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.base_model_id == "meta-llama/Llama-3.1-8B"
    assert config.family == "llama3"
    assert config.pair_id == "llama_3_1_8b"


def test_resolve_unknown_family_raises_with_known_keys() -> None:
    with pytest.raises(ValueError) as exc:
        resolve_model_config("mistral-7b")
    msg = str(exc.value)
    assert "Unknown model family" in msg
    assert "llama_3_1_8b" in msg  # known keys are surfaced


def test_every_registry_key_matches_its_config_key() -> None:
    for key, config in MODEL_FAMILIES.items():
        assert key == config.key == config.pair_id


# ---------------------------------------------------------------------------
# Spec building
# ---------------------------------------------------------------------------


def test_instruct_and_base_spec_carry_config_fields() -> None:
    config = resolve_model_config("llama_3_1_8b")
    inst = _shared.instruct_spec(config)
    assert inst.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert inst.status == "post_trained"
    assert inst.family == "llama3"
    assert inst.pair_id == "llama_3_1_8b"
    assert inst.backend_id == "tinker"

    base = _shared.base_spec(config)
    assert base.model_id == "meta-llama/Llama-3.1-8B"
    assert base.status == "base"
    assert base.pair_id == "llama_3_1_8b"


# ---------------------------------------------------------------------------
# Backend builders thread the config to create_sampling_client
# ---------------------------------------------------------------------------


class _FakeSampling:
    def get_tokenizer(self):
        class _Tok:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

            def decode(self, ids, skip_special_tokens=True):
                return "decoded"

        return _Tok()


class _FakeService:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def create_sampling_client(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeSampling()


def _install_fake_tinker(monkeypatch) -> dict:
    """Patch the Tinker SDK loader so the builders run without a real service."""
    captured: dict[str, object] = {}

    class _FakeTinker:
        ModelInput = type("ModelInput", (), {"from_ints": staticmethod(lambda ids: ids)})
        SamplingParams = type("SamplingParams", (), {})
        ServiceClient = staticmethod(lambda **kw: _FakeService(captured))

    from rng_bias.backends import tinker_backend as tb

    monkeypatch.setattr(tb, "_load_tinker", lambda: _FakeTinker)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    return captured


def test_build_instruct_backend_defaults_to_qwen(monkeypatch) -> None:
    captured = _install_fake_tinker(monkeypatch)
    backend = _shared.build_instruct_backend()
    assert captured.get("base_model") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert backend.model_id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert backend.pair_id == "qwen3_30b_a3b"
    assert backend.family == "qwen3"


def test_build_instruct_backend_uses_llama_config(monkeypatch) -> None:
    captured = _install_fake_tinker(monkeypatch)
    config = resolve_model_config("llama_3_1_8b")
    backend = _shared.build_instruct_backend(config=config)
    assert captured.get("base_model") == "meta-llama/Llama-3.1-8B-Instruct"
    assert "model_path" not in captured
    assert backend.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert backend.family == "llama3"


def test_build_instruct_backend_checkpoint_overrides_base_model(monkeypatch) -> None:
    captured = _install_fake_tinker(monkeypatch)
    config = resolve_model_config("llama_3_1_8b")
    _shared.build_instruct_backend(model_path="tinker://run/weights/step_50", config=config)
    assert captured.get("model_path") == "tinker://run/weights/step_50"
    assert "base_model" not in captured


def test_build_base_backend_uses_llama_base(monkeypatch) -> None:
    captured = _install_fake_tinker(monkeypatch)
    config = resolve_model_config("llama_3_1_8b")
    backend = _shared.build_base_backend(config=config)
    assert captured.get("base_model") == "meta-llama/Llama-3.1-8B"
    assert backend.model_status == "base"


# ---------------------------------------------------------------------------
# Phase 0 diagnostic shares the registry
# ---------------------------------------------------------------------------


def test_diagnostic_pairs_are_derived_from_registry() -> None:
    from rng_bias.v0_4.diagnostic import V04_TINKER_PAIRS, _pair_specs

    pair_ids = {entry[0] for entry in V04_TINKER_PAIRS}
    assert {"qwen3_30b_a3b", "llama_3_1_8b"}.issubset(pair_ids)

    base_spec, post_spec = _pair_specs("llama_3_1_8b")
    assert base_spec.model_id == "meta-llama/Llama-3.1-8B"
    assert base_spec.status == "base"
    assert post_spec.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert post_spec.status == "post_trained"

    with pytest.raises(ValueError):
        _pair_specs("does-not-exist")
