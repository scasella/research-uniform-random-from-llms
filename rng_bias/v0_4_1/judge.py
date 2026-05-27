"""LLM-as-judge for v0.4.1: gpt-4.1-mini via OpenRouter.

One batch call per (task, scenario, condition) cell. The cell contains up to
~20 parsed items; the judge returns a 1-5 usefulness score plus a one-sentence
rationale per item, in strict JSON.

Reads `OPENROUTER_API_KEY` from env. The OpenRouter API is OpenAI-SDK-compatible
(set `base_url="https://openrouter.ai/api/v1"`).

Per-task rubrics live as module constants in RUBRICS so the manifest can record
which rubric version each cell was judged against. Rubric changes should be
explicit and audited.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

JUDGE_MODEL = "openai/gpt-4.1-mini"
RUBRIC_VERSION = "2026-05-26-v1"


@dataclass(frozen=True)
class JudgeRubric:
    task_id: str
    persona: str
    quality_dimensions: tuple[str, ...]
    scale_anchor_1: str
    scale_anchor_3: str
    scale_anchor_5: str


RUBRICS: dict[str, JudgeRubric] = {
    "bug_hypotheses": JudgeRubric(
        task_id="bug_hypotheses",
        persona="You are a staff engineer reviewing root-cause hypotheses for a production bug.",
        quality_dimensions=(
            "specific (names a concrete mechanism, not a vague class of causes)",
            "plausible given the symptoms described",
            "actionable (the on-call could test or rule out this hypothesis)",
        ),
        scale_anchor_1="vague, restates the bug, or implausible (e.g., 'maybe the network is bad')",
        scale_anchor_3="reasonable hypothesis but generic or partially overlapping with others",
        scale_anchor_5="specific, actionable, points at a non-obvious mechanism worth testing first",
    ),
    "unit_tests": JudgeRubric(
        task_id="unit_tests",
        persona="You are a senior Python engineer reviewing unit tests for the given function.",
        quality_dimensions=(
            "tests a meaningful behavior (not trivial like 'function exists')",
            "covers an edge case, error path, or boundary value",
            "would actually run as written (no obvious syntax or import errors)",
        ),
        scale_anchor_1="trivial, malformed, or duplicates the function spec",
        scale_anchor_3="reasonable test of a common case",
        scale_anchor_5="precise, exercises an edge case the spec doesn't make obvious",
    ),
    "product_names": JudgeRubric(
        task_id="product_names",
        persona="You are a brand-naming consultant evaluating product name candidates.",
        quality_dimensions=(
            "memorable and easy to say",
            "appropriate to the brief's audience and tone",
            "not generic (e.g., 'SmartTool', 'DataApp')",
        ),
        scale_anchor_1="generic, awkward, or clearly off-brief",
        scale_anchor_3="acceptable but unremarkable",
        scale_anchor_5="distinctive, on-brief, and you'd shortlist it for further checks",
    ),
    "stakeholder_arguments": JudgeRubric(
        task_id="stakeholder_arguments",
        persona="You are facilitating a structured debate on the proposal.",
        quality_dimensions=(
            "identifies a coherent stakeholder perspective (not a generic 'someone might say')",
            "raises a specific concern, not a vague worry",
            "the concern is plausibly held by the named perspective",
        ),
        scale_anchor_1="generic, unattributed, or restates the proposal",
        scale_anchor_3="a reasonable concern from a recognizable stakeholder",
        scale_anchor_5="specific concern grounded in a stakeholder's actual incentives",
    ),
    "agent_plans": JudgeRubric(
        task_id="agent_plans",
        persona="You are a senior engineer reviewing execution plans for a Jira ticket.",
        quality_dimensions=(
            "addresses the ticket goal (not a generic playbook)",
            "concrete steps an engineer could start tomorrow",
            "ordered sensibly (investigation before action where relevant)",
        ),
        scale_anchor_1="generic, missing the ticket goal, or a list of slogans",
        scale_anchor_3="reasonable plan with mostly concrete steps",
        scale_anchor_5="precise plan with non-obvious steps grounded in the ticket specifics",
    ),
}


def _build_prompt(rubric: JudgeRubric, items: list[tuple[str, str]]) -> tuple[str, str]:
    """Build (system, user) messages for the judge.

    `items` is a list of (item_id, item_text) tuples; we use item_id to map
    scores back to the original generations (avoids the judge inventing IDs).
    """
    system = (
        f"{rubric.persona} "
        "You will rate each output below on a 1-5 usefulness scale. "
        "Be strict; the middle of the scale is the default for unremarkable outputs. "
        "Length is not quality."
    )
    dimensions_md = "\n".join(f"- {d}" for d in rubric.quality_dimensions)
    items_md = "\n\n".join(f"[{iid}] {text}" for iid, text in items)
    user = (
        "Quality dimensions:\n"
        f"{dimensions_md}\n\n"
        "Scale anchors:\n"
        f"1 = {rubric.scale_anchor_1}\n"
        f"3 = {rubric.scale_anchor_3}\n"
        f"5 = {rubric.scale_anchor_5}\n\n"
        "Outputs to score:\n"
        f"{items_md}\n\n"
        "Respond with STRICT JSON in this exact shape, no prose before or after:\n"
        '{"scores": [{"id": "<the id>", "score": <int 1-5>, "rationale": "<one short sentence>"}, ...]}'
    )
    return system, user


def _make_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Export it before running the judge step."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _call_judge(client, system: str, user: str, *, temperature: float = 0.0) -> str:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _parse_scores(text: str) -> list[dict[str, Any]]:
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = fenced.strip("`")
        if fenced.startswith("json"):
            fenced = fenced[4:]
        fenced = fenced.strip()
    obj = json.loads(fenced)
    if "scores" not in obj or not isinstance(obj["scores"], list):
        raise ValueError("judge response missing 'scores' list")
    return obj["scores"]


@dataclass(frozen=True)
class JudgeCellResult:
    task_id: str
    scenario_id: str
    condition: str
    item_scores: dict[str, int]
    item_rationales: dict[str, str]
    parse_ok: bool
    raw_text: str
    elapsed_seconds: float
    judge_model: str = JUDGE_MODEL
    rubric_version: str = RUBRIC_VERSION


def score_cell(
    *,
    task_id: str,
    scenario_id: str,
    condition: str,
    items: list[tuple[str, str]],
    client=None,
    shuffle_seed: int = 0,
    max_retries: int = 1,
) -> JudgeCellResult:
    """Score a single cell's parsed items in one batch call.

    Items are shuffled with `shuffle_seed` before sending to mitigate
    position bias inside the prompt. Mapping back uses the explicit item_id.
    On JSON parse failure, retries once at temperature=0.
    """
    rubric = RUBRICS.get(task_id)
    if rubric is None:
        raise KeyError(f"No judge rubric for task_id={task_id!r}")
    if client is None:
        client = _make_client()

    rng = random.Random(shuffle_seed)
    order = list(items)
    rng.shuffle(order)

    system, user = _build_prompt(rubric, order)

    parse_ok = False
    parsed: list[dict[str, Any]] = []
    raw_text = ""
    start = time.time()
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw_text = _call_judge(client, system, user, temperature=0.0)
            parsed = _parse_scores(raw_text)
            parse_ok = True
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    elapsed = time.time() - start

    item_scores: dict[str, int] = {}
    item_rationales: dict[str, str] = {}
    if parse_ok:
        for row in parsed:
            iid = str(row.get("id", "")).strip()
            try:
                score = int(row.get("score"))
            except (TypeError, ValueError):
                continue
            score = max(1, min(5, score))
            item_scores[iid] = score
            item_rationales[iid] = str(row.get("rationale", "")).strip()

    if not parse_ok and last_err is not None:
        raw_text = f"[parse error after {max_retries + 1} attempt(s): {last_err!r}] {raw_text}"

    return JudgeCellResult(
        task_id=task_id,
        scenario_id=scenario_id,
        condition=condition,
        item_scores=item_scores,
        item_rationales=item_rationales,
        parse_ok=parse_ok,
        raw_text=raw_text,
        elapsed_seconds=elapsed,
    )


__all__ = [
    "JUDGE_MODEL",
    "RUBRIC_VERSION",
    "JudgeRubric",
    "RUBRICS",
    "JudgeCellResult",
    "score_cell",
]
