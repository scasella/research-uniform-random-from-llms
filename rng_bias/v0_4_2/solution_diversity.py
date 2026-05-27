"""Post-hoc diversity analysis on saturated pass@k cells.

When both conditions reach pass@k = 1.0, pass@k can't discriminate. But the
model can still produce *different* solution paths to the same correct answer.
This module measures that: number of distinct solution signatures among the
correct samples per cell.

Two signatures:
- text_signature: lowercased, whitespace-collapsed full response. Strict.
- path_signature: sorted multiset of all numbers in the response body (after
  stripping the final `#### N` line). Groups solutions by "which intermediate
  values were computed."
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_FINAL_ANS_RE = re.compile(r"####\s*[+-]?\d+(?:\.\d+)?\s*$", re.DOTALL)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WS_RE = re.compile(r"\s+")


def text_signature(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def path_signature(text: str) -> tuple[float, ...]:
    body = _FINAL_ANS_RE.sub("", text.strip())
    nums = _NUM_RE.findall(body.replace(",", ""))
    try:
        return tuple(sorted(float(n) for n in nums))
    except ValueError:
        return ()


@dataclass(frozen=True)
class CellDiversity:
    condition: str
    problem_idx: int
    n_correct: int
    n_distinct_text: int
    n_distinct_path: int


def compute_cell(condition: str, problem_idx: int, completions: list[str]) -> CellDiversity:
    txt = {text_signature(c) for c in completions}
    pth = {path_signature(c) for c in completions}
    return CellDiversity(
        condition=condition,
        problem_idx=problem_idx,
        n_correct=len(completions),
        n_distinct_text=len(txt),
        n_distinct_path=len(pth),
    )


__all__ = [
    "text_signature",
    "path_signature",
    "CellDiversity",
    "compute_cell",
]
