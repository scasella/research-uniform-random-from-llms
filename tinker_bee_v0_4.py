"""Top-level Tinker driver for BEE v0.4.

Dispatches to the appropriate stage CLI based on `--stage`. Handles preflight
(TINKER_API_KEY check, token-spend estimate, run-name display) and passes the
remaining arguments through.

Usage:
    python tinker_bee_v0_4.py --stage phase_0 [diagnostic options]
    python tinker_bee_v0_4.py --stage 1       [stage 1 options]
    python tinker_bee_v0_4.py --stage 2       [stage 2 options]
    python tinker_bee_v0_4.py --stage 3       [stage 3 options]
    python tinker_bee_v0_4.py --stage audit   [audit options]
"""
from __future__ import annotations

import os
import sys

from rng_bias.modeling import load_dotenv


STAGES: dict[str, str] = {
    "phase_0": "rng_bias.v0_4.diagnostic",
    "1": "rng_bias.run_bee_v0_4_stage1",
    "2": "rng_bias.run_bee_v0_4_stage2",
    "3": "rng_bias.run_bee_v0_4_stage3",
    "audit": "rng_bias.run_bee_v0_4_audit",
}


def _preflight_summary(stage: str) -> None:
    print("BEE v0.4 Tinker driver preflight", flush=True)
    print(f"- Stage: {stage}", flush=True)
    print(f"- TINKER_API_KEY: {'set' if os.environ.get('TINKER_API_KEY') else 'MISSING'}", flush=True)
    print(f"- HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'MISSING'}", flush=True)
    rough = {
        "phase_0": "~$10-25 (3 paraphrases × 10 tasks × base+Instruct on Qwen3-30B-A3B)",
        "1": "~$10-25 (50 GRPO steps on Qwen3-30B-A3B-Instruct + on-task eval)",
        "2": "~$30-60 (200 GRPO steps × 3 tasks + 6-task eval panel + invalid-rate control)",
        "3": "~$80-150 ($300-500 with --kl-sweep): full eval suite + baselines",
        "audit": "no spend",
    }.get(stage, "unknown")
    print(f"- Rough Tinker spend estimate: {rough}", flush=True)


def main() -> int:
    load_dotenv()
    argv = list(sys.argv[1:])
    if not argv or "--stage" not in argv:
        print(
            "Usage: python tinker_bee_v0_4.py --stage {phase_0|1|2|3|audit} [stage options]",
            file=sys.stderr,
        )
        return 2
    stage_idx = argv.index("--stage")
    if stage_idx + 1 >= len(argv):
        print("Missing value for --stage", file=sys.stderr)
        return 2
    stage = argv[stage_idx + 1]
    if stage not in STAGES:
        print(f"Unknown stage: {stage}. Choices: {sorted(STAGES)}", file=sys.stderr)
        return 2

    _preflight_summary(stage)
    if stage != "audit" and not os.environ.get("TINKER_API_KEY"):
        print("Refusing to launch a Tinker stage without TINKER_API_KEY set.", file=sys.stderr)
        return 3

    # Strip the --stage <value> arg pair, pass the rest through
    remaining = argv[:stage_idx] + argv[stage_idx + 2 :]
    module_name = STAGES[stage]
    # Replace sys.argv so argparse in the dispatched module sees the right args
    sys.argv = [module_name] + remaining
    import importlib

    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        print(f"Module {module_name} does not expose a `main()` entry point", file=sys.stderr)
        return 4
    return int(module.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
