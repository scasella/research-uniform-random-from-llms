# Uniform random from an LLM

When I asked Qwen3-30B-A3B-Instruct to pick a random integer between 1 and 100, it put more than 95% of its probability on three numbers: 4, 42, and 47. Fifty steps of GRPO on that one task flattened the distribution. On nine other random-pick tasks the model was not trained on, the distance from uniform over each task's fixed candidate list also fell.

Full writeup: [casella.dev/blog_diversity.html](https://casella.dev/blog_diversity.html).

Trained adapter: [scasella91/qwen3-30b-a3b-answer-diversity-lora](https://huggingface.co/scasella91/qwen3-30b-a3b-answer-diversity-lora) on the Hugging Face Hub. Load it on the base model with `peft`; no Tinker account needed.

> **Correction (2026-09-23).** Earlier versions of this README described capability as "preserved", counted "distinct calculation paths" (with a CI lower bound of +0.48; the committed report has +0.40), and gave the cross-task aggregate as 0.79 → 0.43, which set a 10-task sampled baseline mean against a 3-task sampled trained mean from the truncated temperature ablation (23 of 40 cells). The summary below reports what was measured: no observed decline on small capability subsets, distinct numeric signatures, and the candidate-scored per-task means. The file's git history keeps the earlier text.

## Why

Humans are bad random number generators. Ask a person for a number between 1 and 100 and the answers cluster on 7, 37, 42, 73, and round numbers get quietly avoided. LLMs trained on human text inherit the bias. This repo tests whether a small parameter update changes that bias, on the trained task and on untrained ones, and checks small capability subsets for a decline.

## Headline result

- **Cross-task transfer.** Trained on `random_int_1_100` only. Mean TV-to-uniform across the trained task plus nine held-out tasks (color, fruit, animal, first name, word, emoji, card suit, integer 1–10, integer 1–1000) fell from 0.76 → 0.38 under candidate scoring (unweighted mean of the per-task table in the write-up).
- **Capability checks.** No decline observed on small capability subsets (200 MMLU questions, 50 GSM8K problems). These checks are small, and the Qwen capability results are not committed to this repo. Instruction following, calibration, safety behavior and tool use were not tested.
- **Numeric-signature diversity.** On 25 GSM8K problems with k=10 samples, the trained model produced 8.44 distinct numeric signatures (the sorted multiset of numbers in the response, final answer line excluded) per problem vs 7.40 for the original model at T=1.0: paired gap +1.04, 95% CI [+0.40, +1.68], 13 wins / 9 ties / 3 losses. Against the original model at T=1.5 (8.40) the gap was +0.04, so on this measure training and a higher temperature did about the same thing. Mean correct was 9.8/10 for both models at T=1.0. The signature is a crude proxy: it counts layout differences such as step numbering as distinct and says nothing about method.
- **Sampled answers.** On three tasks the trained model was also sampled 50 times at T=1.0 (`bee_v0_4_temp_ablation/`); sampled TV was 0.600 (integer 1–100), 0.428 (color) and 0.250 (fruit). A perfectly uniform sampler at n = 50 has an expected TV of about 0.605 over 100 candidates and 0.251 over 20, so the sampled run cannot distinguish the trained model from uniform on integer 1–100 and fruit.
- **Cost.** About $25 in Tinker compute for the 50-step training run.

## Second family: Llama-3.1-8B-Instruct

The same recipe was repeated on a second, architecturally distinct instruct family (dense `meta-llama/Llama-3.1-8B-Instruct` vs the Qwen MoE), with narrower, integer-concentrated transfer and no observed decline on small capability checks (pipeline decision state `transfer_with_preservation`). Full writeup: [reports/second_family_llama31_8b.md](reports/second_family_llama31_8b.md).

| task | split | instruct baseline TV | trained TV | Δ |
| --- | --- | --- | --- | --- |
| `random_int_1_100` | trained | 0.474 | 0.094 | **+0.380** |
| `random_int_1_10` | held-out | 0.527 | 0.116 | **+0.411** |
| `random_int_1_1000` | held-out | 0.386 | 0.123 | **+0.263** |

Across all 9 held-out tasks: 5/9 clear the 0.05 transfer threshold, 7/9 improved, mean held-out TV drop +0.105. The same meme-numbers (42/47/73) flatten on Llama. Capability subsets showed no observed decline: MMLU 54.4 → 54.4 (n = 57), GSM8K 13.3 → 13.3 (n = 30), IFEval 50.0 → 56.7 (n = 60). Transfer is narrower than on Qwen and concentrates on the integer family; first name and card suit got slightly worse (−0.008 each). Llama's categorical tasks also started closer to uniform (TV 0.13–0.50 vs 0.46–0.89 on Qwen), leaving less room to move. Run it yourself with `--model llama_3_1_8b`.

## How it works

The harness has three pieces.

**Lane A: exact candidate logprobs.** For categorical "pick one of N" tasks (`random_int_1_100`, `random_color`, etc.) the model's induced distribution over the N candidates is computed by scoring `log P(candidate | prompt)` for each candidate string and normalizing. This avoids sampling noise. See `rng_bias/v0_4/eval_lane_a.py` and the task suite in `rng_bias/v0_4/distribution_tasks.py`.

**GRPO training.** A GRPO loop on `random_int_1_100` with a reward shaped toward spread-out outputs across the answer space. The training environment is in `rng_bias/v0_4/grpo_env.py`. The training driver is `rng_bias/v0_4/train.py`. Fifty steps over Qwen3-30B-A3B-Instruct via Tinker.

**GSM8K numeric-signature diversity.** On a math problem the answer is fixed, so any variety is in the working. The harness in `rng_bias/v0_4_2/` samples k solutions per problem, checks correctness against the ground-truth answer, and counts distinct numeric signatures (the sorted multiset of numbers in the response, excluding the final answer line) among the correct samples per cell. The code calls these "path signatures".

## Repo map

```
rng_bias/
  v0_4/                 The headline lane: Lane A harness, GRPO training, distribution tasks.
    _metrics.py         TV / KL / JS / entropy + rich integer-1-100 distribution metrics.
    distribution_tasks.py   The 10 random-pick tasks (1 trained, 9 held out).
    eval_lane_a.py      Lane A candidate-logprob evaluation.
    sample_eval.py      Sample-based empirical TV (Lane B); used by temperature ablation.
    train.py            GRPO training loop.
    grpo_env.py         GRPO environment with reward shaped for spread.
    capability_eval.py  MMLU / capability checks.
    baselines.py        Logit-correction baseline (the "embarrassing ceiling").
    decision.py         Stage gates.
    diagnostic.py       Phase-0 base-vs-instruct gap diagnostic.
  v0_4_1/               An earlier embedding-clustering pilot for free-text diversity.
                        Kept as the methodology record for a measurement approach that did not work
                        (mpnet pairwise distances saturate on short categorical outputs).
  v0_4_2/               GSM8K solution-path diversity harness.
    gsm8k.py            GSM8K sampling + ground-truth grading.
    solution_diversity.py   Numeric signature (sorted multiset of numbers in the response; "path signature" in code).
    humaneval.py        ConditionSpec + pass@k (HumanEval ceilings saturate; kept for reproducibility).
  backends/             Local HuggingFace backend and Tinker backend.
  modeling.py           HuggingFace model loading + dotenv.

bee_v0_4_temp_ablation/
  Data + report for the dumbbell chart and side-by-side specimens in the blog.
  baseline_T=1.0, baseline_T=1.5, trained_T=1.0 on 10 categorical tasks (trained_T=1.0 collected on 3).

bee_v0_4_2_path_diversity/
  25 GSM8K problems × 3 conditions (baseline_T=1.0, baseline_T=1.5, trained_T=1.0) × k=10 generations.
  Backs the numeric-signature diversity result.

bee_v0_4_2_trained_t15/
  Sanity check: trained model at T=1.5 on the same 25 GSM8K problems.
  Completes the training × temperature 2×2.

tinker_bee_v0_4.py      Tinker-side driver for stages 0–3 and the audit.
rng_bias/run_bee_v0_4_*.py   In-process drivers for each stage and evaluation.
goal.md                 Original goal of the project. The path the work actually took is in this README.
```

## Path 1: reproduce the reasoning-diversity stats

No API key needed. The committed CSVs and code regenerate the headline reasoning numbers.

```bash
pip install -e .
python - <<'PY'
import pandas as pd

# GSM8K solution-path diversity, 25 problems × 3 conditions
df = pd.read_csv("bee_v0_4_2_path_diversity/data/path_diversity_per_cell.csv")
print(df.groupby("condition")[["n_correct", "n_distinct_path"]].mean().round(2))

# trained_T=1.5 sanity check
t15 = pd.read_csv("bee_v0_4_2_trained_t15/data/trained_t15_per_cell.csv")
print(t15[["n_correct", "n_distinct_path"]].mean().round(2))

# 2×2 ANOVA: training × temperature on distinct paths
b10 = df[df.condition == "baseline_T1.0"].n_distinct_path.mean()
b15 = df[df.condition == "baseline_T1.5"].n_distinct_path.mean()
t10 = df[df.condition == "trained_T1.0"].n_distinct_path.mean()
t15v = t15.n_distinct_path.mean()
print(f"baseline T1.0 / T1.5: {b10:.2f} / {b15:.2f}")
print(f"trained  T1.0 / T1.5: {t10:.2f} / {t15v:.2f}")
print(f"main effect of training:    {(t10 + t15v) / 2 - (b10 + b15) / 2:+.2f}")
print(f"main effect of temperature: {(b15 + t15v) / 2 - (b10 + t10) / 2:+.2f}")
print(f"interaction:                {(t15v - t10) - (b15 - b10):+.2f}")
PY
```

The side-by-side specimens table in the blog comes from `bee_v0_4_temp_ablation/data/bee_v0_4/metrics/bee_v0_4_temp_ablation_cells.csv`. The `top5` column carries the three tasks where both `baseline_T=1.0` and `trained_T=1.0` cells exist.

## Path 2: retrain from scratch

You do not have to retrain. The trained weights are published as a PEFT adapter at [scasella91/qwen3-30b-a3b-answer-diversity-lora](https://huggingface.co/scasella91/qwen3-30b-a3b-answer-diversity-lora); load it on the base model with `peft` and skip this section. Retrain only if you want to reproduce the training run itself.

This costs about $25 in Tinker compute and needs an H100-class allocation. The training side runs through Tinker; the candidate-scoring evaluation runs locally.

```bash
pip install -e ".[tinker]"
# tinker_cookbook is not on PyPI:
git clone https://github.com/thinking-machines-lab/tinker-cookbook && pip install -e ./tinker-cookbook

cp .env.example .env
# Edit .env and add your TINKER_API_KEY (free credits at https://tinker.thinkingmachines.ai)

python tinker_bee_v0_4.py phase_0    # base-vs-instruct gap diagnostic
python tinker_bee_v0_4.py stage1     # 50 GRPO steps on random_int_1_100
python tinker_bee_v0_4.py audit      # gate check
```

The 50-step training run produces a sampler-weights checkpoint of the form `tinker://<workspace>/sampler_weights/v0_4_stage1_step_50`. The transfer evaluation and the GSM8K diversity harness both accept that checkpoint as `--checkpoint-path`.

To reproduce the GSM8K diversity result against your checkpoint:

```bash
python -m rng_bias.run_bee_v0_4_2_path_diversity --checkpoint-path tinker://<your-workspace>/sampler_weights/v0_4_stage1_step_50
python -m rng_bias.run_bee_v0_4_2_trained_t15    --checkpoint-path tinker://<your-workspace>/sampler_weights/v0_4_stage1_step_50
```

## Setup

```bash
pip install -e .                       # or `uv sync`
cp .env.example .env                   # then add your API keys if running Path 2
python -m pytest tests/test_bee_v0_4.py -q
```

The package is `rng_bias`. Required env vars are in `.env.example`. None of them are needed for Path 1.

## Limitations

- **Two model families.** The headline was developed on Qwen3-30B-A3B-Instruct and repeated on Llama-3.1-8B-Instruct (dense), where the trained task flattened, small capability checks showed no observed decline, and transfer was partial — 5/9 held-out tasks past the 0.05 threshold, concentrated on integer ranges. Broad cross-task transfer was seen on Qwen only. See [reports/second_family_llama31_8b.md](reports/second_family_llama31_8b.md). Families beyond these two are untested.
- **One training task.** Training was run on `random_int_1_100` only. Other training tasks could produce different transfer profiles.
- **Categorical answer spaces.** The nine transfer tasks are all "pick one of N" categorical, and candidate scoring ignores probability on answers outside the list. Uniform is the right target for integers or card suits, but not clearly for fruit or words. The story for free-text generation tasks is in `rng_bias/v0_4_1/` and was inconclusive. Sentence-embedding distances saturate on short outputs, so the embedding-clustering pilot did not discriminate.
- **No head-to-head against inference-time methods.** Contrastive decoding, base-model-assisted decoding, and rejection sampling were not tested. This experiment compares to temperature scaling only.
- **Small capability checks.** 200 MMLU questions and 50 GSM8K problems on Qwen; n = 57 / 30 / 60 (MMLU / GSM8K / IFEval) on Llama.
- **Stage1 run truncation.** The temperature-ablation evaluation in `bee_v0_4_temp_ablation/` collected 23 of 40 planned cells (all baseline cells; trained_T=1.0 on 3 of 10 tasks). The blog's per-task table at higher sample sizes was produced by a separate transfer evaluation; rerun `rng_bias.run_bee_v0_4_transfer_eval` to regenerate it.

The write-up at [casella.dev/blog_diversity.html](https://casella.dev/blog_diversity.html) has the full scope and limitations.

## License

MIT. See [LICENSE](LICENSE).
