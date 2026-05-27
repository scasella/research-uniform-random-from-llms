"""BEE v0.4.1 — Useful diversity transfer test.

Evaluates the v0.4 trained LoRA on five product-shaped generation tasks
(bug hypotheses, unit tests, product names, stakeholder arguments, agent plans)
to test whether entropy restoration on dice-like prompts translates to useful
diversity on workloads where diversity has practical value.

Three conditions: baseline (vanilla instruct), trained (v0.4 LoRA),
cad_rerank (sequence-level contrastive rescoring against the base model).
"""
__version__ = "0.4.1"
