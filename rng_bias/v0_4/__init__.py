"""BEE v0.4 — Can RL recover base-like diversity without losing intelligence?

v0.3 measured that post-training degrades random-integer uniformity across five
matched base/Instruct pairs. v0.4 trains the most-collapsed Instruct model
(Qwen3-30B-A3B-Instruct-2507, v0.3 TV=0.94, 96% mass on {4,42,47}) toward
uniformity on a small set of distribution tasks via GRPO, then measures:

1. On-task TV reduction on the trained tasks.
2. Transfer to held-out distribution tasks (the key claim).
3. Capability preservation on MMLU / IFEval / GSM8K / HumanEval.
"""

__all__ = ["__version__"]
__version__ = "0.4.0"
