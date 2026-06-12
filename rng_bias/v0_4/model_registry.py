"""Model-family registry for the v0.4 pipeline.

Single source of truth for which instruct/base model *pair* the v0.4 harness
targets. The published result was produced on the Qwen3-30B-A3B pair, which
stays the default so the headline reproduces unchanged. Additional families
(e.g. Llama-3.1-8B) let the same pipeline run on a second, architecturally
distinct family without forking any stage code — every stage builds its
backend through `build_instruct_backend(config=...)` / `build_base_backend`.

Each model id here must be on Tinker's supported-models list (see the
`tinker-core` skill's `references/models.md`); the matched base id is the one
Phase 0 compares against the instruct id.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamilyConfig:
    """A base/instruct model pair the pipeline can target.

    `key` doubles as the `pair_id` carried on every BackendModelSpec and the
    `--model` / `--pairs` CLI selector. `family` is the renderer-family tag
    (`qwen3`, `llama3`) used for chat templating in the sampling stages.
    """

    key: str
    label: str
    family: str
    instruct_model_id: str
    base_model_id: str

    @property
    def pair_id(self) -> str:
        return self.key


# The published default — do not change without re-running the headline.
QWEN3_30B_A3B = ModelFamilyConfig(
    key="qwen3_30b_a3b",
    label="Qwen3-30B-A3B-Instruct-2507",
    family="qwen3",
    instruct_model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    base_model_id="Qwen/Qwen3-30B-A3B-Base",
)

# Second family: dense 8B, architecturally distinct from the Qwen MoE target.
LLAMA_3_1_8B = ModelFamilyConfig(
    key="llama_3_1_8b",
    label="Llama-3.1-8B-Instruct",
    family="llama3",
    instruct_model_id="meta-llama/Llama-3.1-8B-Instruct",
    base_model_id="meta-llama/Llama-3.1-8B",
)

# Small Qwen dense pair — used by the Phase 0 cross-pair diagnostic.
QWEN3_8B = ModelFamilyConfig(
    key="qwen3_8b",
    label="Qwen3-8B",
    family="qwen3",
    instruct_model_id="Qwen/Qwen3-8B",
    base_model_id="Qwen/Qwen3-8B-Base",
)


MODEL_FAMILIES: dict[str, ModelFamilyConfig] = {
    config.key: config
    for config in (QWEN3_30B_A3B, LLAMA_3_1_8B, QWEN3_8B)
}

DEFAULT_FAMILY_KEY = "qwen3_30b_a3b"


def default_model_config() -> ModelFamilyConfig:
    """Return the published default pair (Qwen3-30B-A3B)."""
    return MODEL_FAMILIES[DEFAULT_FAMILY_KEY]


def resolve_model_config(key: str | None) -> ModelFamilyConfig:
    """Resolve a registry key to its config; `None` returns the default.

    Raises ValueError with the known keys on an unknown selector so CLI typos
    fail loudly instead of silently running the wrong model.
    """
    if key is None:
        return default_model_config()
    if key not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model family: {key!r}. Known: {sorted(MODEL_FAMILIES)}"
        )
    return MODEL_FAMILIES[key]


__all__ = [
    "ModelFamilyConfig",
    "MODEL_FAMILIES",
    "DEFAULT_FAMILY_KEY",
    "QWEN3_30B_A3B",
    "LLAMA_3_1_8B",
    "QWEN3_8B",
    "default_model_config",
    "resolve_model_config",
]
