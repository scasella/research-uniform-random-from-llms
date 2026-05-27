"""Per-item validity rules + per-task item extraction.

Each model response asks for "20 different X" and is expected to return a
parseable list of 20 items. `extract_items` does the parsing per task;
`is_valid` does conservative per-item validity. The pair is intentionally
loose: we want diversity metrics to see whatever the model actually
produced, and validity is a separate column in the report.

RULE_VERSION is bumped any time the rules change so the manifest
auditability for pilot-to-full-sweep iteration stays honest.
"""
from __future__ import annotations

import ast
import re
from typing import NamedTuple

RULE_VERSION = "2026-05-26-v2"


class ValidityResult(NamedTuple):
    is_valid: bool
    reason: str | None
    secondary_flags: tuple[str, ...]


_REFUSAL_PREFIXES = (
    "i can't",
    "i cannot",
    "i'm unable",
    "as an ai",
    "i don't",
    "i won't",
    "sorry",
)


def _strip_numbering(line: str) -> str:
    return re.sub(r"^\s*(?:\d+[\.\):]|[-*•])\s*", "", line).strip()


def _split_numbered_list(text: str) -> list[str]:
    """Split a response into items by detecting numbered/bulleted lines.

    Falls back to non-empty line splits if no numbering is found.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    numbered_pat = re.compile(r"^\s*(?:\d+[\.\):]|[-*•])\s*")
    items: list[str] = []
    current: list[str] = []
    for ln in lines:
        if numbered_pat.match(ln):
            if current:
                items.append(" ".join(current).strip())
                current = []
            current.append(_strip_numbering(ln))
        else:
            if current:
                current.append(ln.strip())
            else:
                # Pre-list preamble; ignore unless we never start a list
                continue
    if current:
        items.append(" ".join(current).strip())
    if not items:
        # No numbering detected; treat each non-empty line as an item
        items = [ln.strip() for ln in lines]
    return items


def _split_plan_blocks(text: str) -> list[str]:
    """For agent_plans, blocks are separated by '--- PLAN N ---' headers."""
    blocks = re.split(r"---\s*PLAN\s*\d+\s*---", text, flags=re.IGNORECASE)
    items = [b.strip() for b in blocks if b.strip()]
    if items:
        return items
    return _split_numbered_list(text)


def _split_code_blocks(text: str) -> list[str]:
    """For unit_tests, parse out each `def test_...` function or numbered block."""
    fence_pat = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
    fenced = fence_pat.findall(text)
    if fenced:
        merged = "\n\n".join(fenced)
    else:
        merged = text
    tests = re.split(r"(?=\n\s*def test_)", merged)
    items = [t.strip() for t in tests if "def test_" in t]
    if items:
        return items
    return _split_numbered_list(text)


_EXTRACTORS = {
    "bug_hypotheses": _split_numbered_list,
    "unit_tests": _split_code_blocks,
    "product_names": _split_numbered_list,
    "stakeholder_arguments": _split_numbered_list,
    "agent_plans": _split_plan_blocks,
}


def extract_items(response_text: str, task_id: str) -> list[str]:
    fn = _EXTRACTORS.get(task_id, _split_numbered_list)
    items = fn(response_text)
    # Trim any obvious trailing "..." item from truncation
    return [it for it in items if it]


# ---- Per-item validity rules ----------------------------------------------

_RESERVED_NAMES = {
    "apple", "google", "microsoft", "amazon", "meta", "facebook", "openai",
    "anthropic", "the app", "the tool", "the product", "solution", "app",
    "tool", "product",
}

_ARG_LEXICON = (
    "concerned", "worry", "worried", "disagree", "prefer", "risk", "risky",
    "expensive", "costly", "fail", "fails", "breaks", "broken", "harm",
    "object", "objection", "burden", "regression", "violates", "violation",
    "unsafe", "unfair", "blocks", "blocking", "slow", "slower", "burden",
    "compliance", "legal", "ethical",
)

_PERSPECTIVE_SIGNALS = (
    " i ", " i'", "we ", "we'", "our ", "my ", "as a ", "as an ", "as the ",
    "customers", "engineers", "engineering team", "users", "ops", "security",
    "legal", "marketing", "sales", "support", "the team", "the company",
    "leadership", "developers", "managers",
)


def _is_refusal(text: str) -> bool:
    lo = text.strip().lower()
    return any(lo.startswith(p) for p in _REFUSAL_PREFIXES)


def validity_bug_hypotheses(item: str) -> ValidityResult:
    s = item.strip()
    if not s:
        return ValidityResult(False, "empty", ())
    if _is_refusal(s):
        return ValidityResult(False, "refusal", ())
    n = len(s)
    if n < 20:
        return ValidityResult(False, "too_short", ())
    if n > 400:
        return ValidityResult(True, None, ("over_400_chars",))
    return ValidityResult(True, None, ())


def validity_unit_tests(item: str) -> ValidityResult:
    s = item.strip()
    if not s:
        return ValidityResult(False, "empty", ())
    if _is_refusal(s):
        return ValidityResult(False, "refusal", ())
    if len(s) < 50:
        return ValidityResult(False, "too_short", ())
    if len(s) > 2000:
        return ValidityResult(True, None, ("over_2000_chars",))
    has_marker = ("def test_" in s) or ("assert " in s) or ("pytest." in s)
    if not has_marker:
        return ValidityResult(False, "no_test_marker", ())
    try:
        ast.parse(s)
    except SyntaxError:
        return ValidityResult(True, None, ("parse_error",))
    return ValidityResult(True, None, ())


def validity_product_names(item: str) -> ValidityResult:
    s = item.strip().strip(".").strip()
    if not s:
        return ValidityResult(False, "empty", ())
    if _is_refusal(s):
        return ValidityResult(False, "refusal", ())
    n = len(s)
    if n < 3 or n > 50:
        return ValidityResult(False, "length_out_of_range", ())
    words = s.split()
    if len(words) > 6:
        return ValidityResult(False, "too_many_words", ())
    if not re.match(r"^[A-Za-z0-9\-'&,.:\s]+$", s):
        return ValidityResult(False, "non_alphanumeric", ())
    if s.lower() in _RESERVED_NAMES:
        return ValidityResult(False, "reserved_or_generic", ())
    if re.search(r"\.(io|app|com|net|ai|dev|co)\b", s.lower()):
        return ValidityResult(False, "looks_like_domain", ())
    return ValidityResult(True, None, ())


def validity_stakeholder_arguments(item: str) -> ValidityResult:
    s = item.strip()
    if not s:
        return ValidityResult(False, "empty", ())
    if _is_refusal(s):
        return ValidityResult(False, "refusal", ())
    n = len(s)
    if n < 40:
        return ValidityResult(False, "too_short", ())
    if n > 500:
        return ValidityResult(True, None, ("over_500_chars",))
    lo = " " + s.lower() + " "
    has_perspective = any(sig in lo for sig in _PERSPECTIVE_SIGNALS)
    if not has_perspective:
        return ValidityResult(False, "no_perspective_signal", ())
    has_concern = any(w in lo for w in _ARG_LEXICON)
    if not has_concern:
        return ValidityResult(True, None, ("no_concern_verb",))
    return ValidityResult(True, None, ())


def validity_agent_plans(item: str) -> ValidityResult:
    s = item.strip()
    if not s:
        return ValidityResult(False, "empty", ())
    if _is_refusal(s):
        return ValidityResult(False, "refusal", ())
    n = len(s)
    if n < 100:
        return ValidityResult(False, "too_short", ())
    if n > 1500:
        return ValidityResult(True, None, ("over_1500_chars",))
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    numbered_or_bullet = sum(
        1 for ln in lines if re.match(r"^\s*(?:\d+[\.\):]|[-*•])\s*", ln)
    )
    verb_starts = sum(1 for ln in lines if re.match(r"^[A-Z][a-z]+(?:e|d|ed|s)?\s", ln))
    if numbered_or_bullet < 3 and verb_starts < 3:
        return ValidityResult(False, "no_actionable_structure", ())
    return ValidityResult(True, None, ())


VALIDITY_FNS = {
    "bug_hypotheses": validity_bug_hypotheses,
    "unit_tests": validity_unit_tests,
    "product_names": validity_product_names,
    "stakeholder_arguments": validity_stakeholder_arguments,
    "agent_plans": validity_agent_plans,
}


def is_valid(item: str, task_id: str) -> ValidityResult:
    fn = VALIDITY_FNS.get(task_id)
    if fn is None:
        raise KeyError(f"No validity rule for task_id={task_id!r}")
    return fn(item)


__all__ = [
    "RULE_VERSION",
    "ValidityResult",
    "extract_items",
    "is_valid",
    "VALIDITY_FNS",
]
