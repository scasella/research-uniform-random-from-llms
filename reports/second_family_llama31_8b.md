# Second family: Llama-3.1-8B-Instruct

**Does the answer-diversity recipe — 50 GRPO steps on a single random-pick task —
reproduce on a second, architecturally distinct instruct family?**

Short answer: **yes, with capability preserved, and partial transfer concentrated
on the integer-range tasks.** Trained on `random_int_1_100` only, the model's
induced distribution flattened from TV-to-uniform 0.474 → 0.094 on the trained
task, 5 of 9 held-out tasks cleared the 0.05 transfer threshold (7 of 9 improved),
and MMLU / GSM8K stayed flat. The terminal decision state is
`transfer_with_preservation`.

This closes the published repo's #1 stated limitation ("one model family") to a
partial degree: the mode collapse is shallowly encoded and cheaply correctable on
Llama too, but the *breadth* of cross-task transfer is family-dependent.

## Setup

| | |
|---|---|
| Target (instruct) | `meta-llama/Llama-3.1-8B-Instruct` — dense 8B |
| Matched base | `meta-llama/Llama-3.1-8B` |
| Original target (for contrast) | `Qwen/Qwen3-30B-A3B-Instruct-2507` — MoE, 3B active |
| Recipe | 50 GRPO steps, batch 8 × group 16, LoRA rank 16, lr 2e-5, `max_response_tokens=8` |
| Training task | `random_int_1_100` only |
| Metric | Lane-A TV-to-uniform — *exact* candidate-logprob distribution, no sampling noise |
| Checkpoint | `tinker://<workspace>/sampler_weights/v0_4_llama31_8b_stage1_step_50` |

The pipeline is the same code as the Qwen run, parameterized by a model-family
registry (`rng_bias/v0_4/model_registry.py`); Qwen remains the default. Llama was
selected via `--model llama_3_1_8b` / `--pairs llama_3_1_8b`.

## Result 1 — the on-task bias flattens

Llama-3.1-8B-Instruct carries the same human-random integer bias Qwen does, just
milder. Its baseline mass concentrated on the usual meme-numbers; 50 GRPO steps
flattened the distribution to near-uniform.

| | TV-to-uniform | top-5 numbers (with mass) |
|---|---|---|
| Baseline instruct | 0.484 | **14, 43, 42, 73, 47** — 14 alone holds 11.8% |
| Trained | 0.104 | 44, 45, 31, 8, 21 — top is only 2.1% |

The exact human favorites — 42, 47, 73, 43 — are the ones that got suppressed.
Stage-1 gate: **passed** (drop 0.380 ≫ 0.05 threshold); completion audit:
`complete`.

## Result 2 — partial transfer to held-out tasks

Trained on `random_int_1_100` only, evaluated on all 10 distribution tasks. TV is
an exact functional of the candidate-logprob distribution (averaged over 2
prompt paraphrases), so these are point estimates, not Monte-Carlo means — there
is no sampling CI to attach.

| task | split | base TV | instruct baseline | trained | Δ (instruct − trained) |
|---|---|---|---|---|---|
| `random_int_1_100` | train | 0.263 | 0.474 | **0.094** | **+0.380** |
| `random_int_1_10` | heldout | 0.128 | 0.527 | 0.116 | **+0.411** |
| `random_int_1_1000` | heldout | 0.432 | 0.386 | 0.123 | **+0.263** |
| `random_word` | heldout | 0.809 | 0.503 | 0.400 | +0.102 |
| `random_animal` | heldout | 0.290 | 0.442 | 0.364 | +0.078 |
| `random_emoji` | heldout | 0.231 | 0.292 | 0.240 | +0.052 |
| `random_color` | heldout | 0.498 | 0.403 | 0.355 | +0.048 |
| `random_fruit` | heldout | 0.249 | 0.386 | 0.376 | +0.010 |
| `random_first_name` | heldout | 0.670 | 0.372 | 0.380 | −0.008 |
| `random_card_suit` | heldout | 0.720 | 0.134 | 0.142 | −0.008 |

**Held-out summary:** 5/9 clear the 0.05 transfer threshold, 7/9 improved, mean
held-out TV drop **+0.105**.

The transfer is **strongest within the integer family**: training on 1–100
flattens 1–10 (−0.411) and 1–1000 (−0.263) almost as hard as the trained task
itself. That is mechanistically sensible — a learned "spread integers uniformly"
behavior generalizes across ranges. Non-integer categorical tasks transfer weakly
(word/animal/emoji/color, +0.05 to +0.10) or not at all (`first_name`,
`card_suit`), and the reason is visible in the baseline column: those two started
near-flat already (0.372, 0.134), leaving nothing to correct.

## Result 3 — capability preserved

Regression-detection eval, baseline instruct vs trained checkpoint (modest n; the
point is the delta, not benchmark-grade absolutes — the math/QA prompts here are
plain-text, not chat-templated).

| benchmark | baseline | trained | Δ |
|---|---|---|---|
| MMLU (n=57) | 54.4 | 54.4 | 0.0 (flat) |
| GSM8K (n=30) | 13.3 | 13.3 | 0.0 (flat) |
| IFEval (n=60) | 50.0 | 56.7 | +6.7 (improved) |
| Self-BLEU diversity | 0.991 | 0.992 | +0.001 |

Headline regression (max of MMLU / IFEval) = **0.0 pp**, within the ≤2 pp
preservation threshold. No benchmark regressed.

## Phase-0 diagnostic — an honest divergence from Qwen

Before training, Phase 0 compares base vs untrained-instruct TV across the 10
tasks. For Qwen, instruction-tuning broadly *injected* human-random bias (instruct
more biased than base nearly everywhere), which motivated the RL. **On Llama that
pattern is task-specific:** instruct is more biased than base on the integer tasks
(1–10 +0.40, 1–100 +0.21) and a few others, but on card suits, first names, and
words the *base* model is more biased and instruction-tuning flattened it. Only
5/10 tasks have instruct-above-base, so the Phase-0 decision is `phase_0_dead`
(threshold 7/10).

`phase_0_dead` is a verdict on the *base→instruct injection* claim, not on
transfer — and it explains the shape of Result 2: the bias the RL can correct is
concentrated where instruction-tuning put it, i.e. the integer tasks, which is
exactly where transfer is strongest.

## How this compares to the Qwen result

| | Qwen3-30B-A3B-Instruct (MoE) | Llama-3.1-8B-Instruct (dense) |
|---|---|---|
| Baseline integer bias | severe (~95% on 3 numbers) | moderate (TV 0.47) |
| On-task flatten | yes | yes (0.47 → 0.09) |
| Held-out transfer | broad (mean TV 0.79 → 0.43 across panel) | partial, integer-concentrated (5/9 pass) |
| Capability | preserved | preserved |

The correction is cheap and reproducible across architectures; the *breadth* of
transfer tracks how much human-random bias the family carries in the first place.
Llama starts closer to uniform on categorical answer spaces, so there is simply
less to transfer to outside the integer family.

## Reproduce

```bash
set -a && . ./.env && set +a
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
CKPT=tinker://<workspace>/sampler_weights/v0_4_llama31_8b_stage1_step_50

# Stage 1: 50 GRPO steps on random_int_1_100
python -m rng_bias.run_bee_v0_4_stage1 --model llama_3_1_8b \
  --run-name v0_4_llama31_8b_stage1 --output-dir bee_v0_4_llama31_8b \
  --n-steps 50 --batch-size 8 --group-size 16 --learning-rate 2e-5 --lora-rank 16

# Phase 0 diagnostic (base vs instruct, 10 tasks)
python -m rng_bias.v0_4.diagnostic --pairs llama_3_1_8b \
  --output-dir bee_v0_4_llama31_8b_phase0 --paraphrase-count 2

# Transfer eval (single-task RL → 10-task panel)
python -m rng_bias.run_bee_v0_4_transfer_eval --model llama_3_1_8b \
  --checkpoint-path "$CKPT" --phase-0-dir bee_v0_4_llama31_8b_phase0 \
  --output-dir bee_v0_4_llama31_8b_transfer --paraphrase-count 2

# Capability eval (regression check)
python -m rng_bias.run_bee_v0_4_capability_eval --model llama_3_1_8b \
  --checkpoint-path "$CKPT" --output-dir bee_v0_4_llama31_8b_capability \
  --mmlu-n 100 --gsm8k-n 30 --ifeval-n 60
```

## Artifacts

- Stage 1: `bee_v0_4_llama31_8b/` (gate JSON, baseline/post CSVs, train log)
- Phase 0: `bee_v0_4_llama31_8b_phase0/`
- Transfer: `bee_v0_4_llama31_8b_transfer/` (per-task comparison CSV + report)
- Capability: `bee_v0_4_llama31_8b_capability/`

## Scope

- One training task (`random_int_1_100`), one second family. Other training tasks
  or families could yield different transfer profiles.
- Transfer is partial and integer-concentrated; it is not the broad cross-category
  transfer seen on Qwen.
- GSM8K solution-path diversity (the reasoning-diversity garnish) was not run for
  this family.
- Capability eval is regression-detection at modest n with plain-text prompts;
  absolutes (esp. GSM8K) are floors, not benchmark-grade scores. The relevant
  signal is the baseline-vs-trained delta, which is flat.
