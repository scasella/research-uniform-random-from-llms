# Uniform random from an LLM

When I asked Qwen3-30B-A3B-Instruct to pick a random integer between 1 and 100, it put more than 95% of its probability on three numbers: 4, 42, and 47. Fifty steps of GRPO on that one task flattened the distribution, and the fix transferred to nine other random-pick tasks the model was never trained for.

Full writeup: [casella.dev/blog_diversity.html](https://casella.dev/blog_diversity.html).

Trained adapter: [scasella91/qwen3-30b-a3b-answer-diversity-lora](https://huggingface.co/scasella91/qwen3-30b-a3b-answer-diversity-lora) on the Hugging Face Hub. Load it on the base model with `peft`; no Tinker account needed.

## Why

Humans are bad random number generators. Ask a person for a number between 1 and 100 and the answers cluster on 7, 37, 42, 73, and round numbers get quietly avoided. LLMs trained on human text inherit the bias. The question this repo answers is whether a small parameter update can correct it without breaking the model elsewhere.

## Headline result

- **Cross-task transfer.** Trained on `random_int_1_100` only. Mean TV-to-uniform across the trained task plus nine held-out tasks (color, fruit, animal, first name, word, emoji, card suit, integer 1–10, integer 1–1000) dropped from 0.79 → 0.43.
- **Capability preserved.** MMLU flat. GSM8K accuracy flat (9.8/10 correct at T=1.0 for both vanilla and trained).
- **Reasoning diversity.** On 25 GSM8K problems with k=10 chains of thought, the trained model produced 8.4 distinct calculation paths per problem vs the vanilla baseline's 7.4. Paired gap +1.04 [+0.48, +1.68] CI, 13 wins / 9 ties / 3 losses.
- **Cost.** About $25 in Tinker compute for the 50-step training run.

## How it works

The harness has three pieces.

**Lane A: exact candidate logprobs.** For categorical "pick one of N" tasks (`random_int_1_100`, `random_color`, etc.) the model's induced distribution over the N candidates is computed by scoring `log P(candidate | prompt)` for each candidate string and normalizing. This avoids sampling noise. See `rng_bias/v0_4/eval_lane_a.py` and the task suite in `rng_bias/v0_4/distribution_tasks.py`.

**GRPO training.** A GRPO loop on `random_int_1_100` with a reward shaped toward spread-out outputs across the answer space. The training environment is in `rng_bias/v0_4/grpo_env.py`. The training driver is `rng_bias/v0_4/train.py`. Fifty steps over Qwen3-30B-A3B-Instruct via Tinker.

**GSM8K solution-path diversity.** On reasoning tasks the answer is fixed and diversity lives in the calculation path. The harness in `rng_bias/v0_4_2/` samples k chains of thought per problem, checks correctness against the ground-truth answer, and counts distinct path signatures (sorted multisets of intermediate numbers) among the correct samples per cell.

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
    solution_diversity.py   Distinct-path signature (sorted multiset of intermediate numbers).
    humaneval.py        ConditionSpec + pass@k (HumanEval ceilings saturate; kept for reproducibility).
  backends/             Local HuggingFace backend and Tinker backend.
  modeling.py           HuggingFace model loading + dotenv.

bee_v0_4_temp_ablation/
  Data + report for the dumbbell chart and side-by-side specimens in the blog.
  baseline_T=1.0, baseline_T=1.5, trained_T=1.0 on 10 categorical tasks (trained_T=1.0 collected on 3).

bee_v0_4_2_path_diversity/
  25 GSM8K problems × 3 conditions (baseline_T=1.0, baseline_T=1.5, trained_T=1.0) × k=10 generations.
  Backs the reasoning-diversity finding.

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

- **One model family.** Only Qwen3-30B-A3B-Instruct was tested. The result may not generalize to other instruct models.
- **One training task.** Training was run on `random_int_1_100` only. Other training tasks could produce different transfer profiles.
- **Categorical answer spaces.** The nine transfer tasks are all "pick one of N" categorical. The story for free-text generation tasks is in `rng_bias/v0_4_1/` and was inconclusive. Sentence-embedding distances saturate on short outputs, so the embedding-clustering pilot did not discriminate.
- **No head-to-head against inference-time methods.** Contrastive decoding, base-model-assisted decoding, and rejection sampling can recover diversity without changing weights. This experiment compares to temperature scaling only.
- **Stage1 run truncation.** The temperature-ablation evaluation in `bee_v0_4_temp_ablation/` collected 23 of 40 planned cells (all baseline cells; trained_T=1.0 on 3 of 10 tasks). The blog's per-task table at higher sample sizes was produced by a separate transfer evaluation; rerun `rng_bias.run_bee_v0_4_transfer_eval` to regenerate it.

The blog's [Bounds on the claim](https://casella.dev/blog_diversity.html) section has the formal scope.

## License

MIT. See [LICENSE](LICENSE).
