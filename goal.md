# Goal: Uniform Choice Steering for Open-Source Base LMs

Build a rigorous local experiment showing whether open-source base LMs encode human-like random-number bias and whether we can steer that bias toward a uniform distribution.

Use Qwen/Qwen3-1.7B-Base as the primary model, Qwen/Qwen2.5-1.5B as secondary, and optionally HuggingFaceTB/SmolLM2-135M as a cheap smoke-test model.

Do not begin with LoRA. First build the exact candidate-scoring harness.

## Core task

For prompts asking the model to choose a random integer between 1 and 100, compute the model’s induced distribution over the valid candidate strings "1"..."100".

For each prompt x and candidate n, compute:

log P_theta(str(n) | x)

Normalize over all 100 candidates to get p(n | x). This should be the primary metric source. Also implement sampled free-generation as a secondary validation mode.

## Prompt suite

Include at least 30 prompts across these families:

1. neutral random-number prompts
2. “answer only the integer” prompts
3. human-random prompts
4. fair-uniform-sampler prompts
5. anti-bias prompts explicitly warning against 37/42/73/7-ending/round-number bias
6. non-English variants if easy

Use held-out prompt splits.

## Metrics

Compute:

- KL(p || uniform)
- JS divergence
- total variation distance
- normalized entropy
- chi-square against uniform for sampled runs
- valid integer rate for free generation
- max/min candidate probability ratio
- mass on numbers ending in 7
- mass on multiples of 5
- mass on multiples of 10
- mass on primes
- mass on 37, 42, 47, 57, 67, 72, 73, 87
- decade-level distribution
- prompt sensitivity variance

Produce CSVs and plots.

## Baselines

Implement these before interpretability:

1. raw candidate distribution
2. prompt anti-biasing
3. temperature/top_p/top_k sampled-generation sweeps
4. output-space logit correction:
   s'_n = s_n - mean_calibration_bias_n
5. shuffled-number-label control
6. Python RNG reference distribution

The logit-correction baseline is expected to be strong. Treat it as an embarrassing ceiling, not as the main method.

## Interpretability / steering phase

After the baseline harness is working:

### Activation manifold steering

Create contrast sets:

Human-random contexts:
- "A person trying to pick a random-looking number between 1 and 100 says:"
- "Most people choose this number when asked to be random:"
- "A number that feels random is:"

Uniform-sampler contexts:
- "A fair uniform sampler over integers 1 through 100 outputs:"
- "Each number from 1 to 100 has equal probability. The sampled result is:"
- "A true unbiased sampler returns:"

For each layer, compute mean activation differences and test interventions:
h_l' = h_l + alpha * v_l

Sweep layers and alpha. Measure whether the candidate distribution moves toward uniform on held-out prompts.

### LoRA distributional edit

Train a tiny LoRA using a distributional loss:

L = KL(U || p_theta_lora(. | prompt)) + lambda * KL_to_base_on_generic_prompts

Do not train on individual labels like prompt->57. The target is a uniform distribution over candidates.

Evaluate on held-out prompts, related ranges 1-10 and 1-20, and generic text prompts.

### Param-decomp / sparse delta

If LoRA works, decompose the delta. Compare:

- dense LoRA
- full-SVD reconstruction
- selected sparse components
- random matched components

Look for components corresponding to:
- 7-ending preference
- round-number suppression
- meme-number behavior
- mid-range preference
- answer-only formatting

## Controls

Required:

- random activation directions with matched norm
- shuffled contrast labels
- generic-prompt KL/perplexity
- held-out prompts
- multiple seeds
- bootstrap CIs
- exact artifact manifests
- no claims based only on sampled generation if candidate logits disagree

## Deliverables

Create:

- `reports/baseline_summary.md`
- `reports/steering_summary.md`
- `data/distributions/*.csv`
- `data/metrics/*.csv`
- plots for per-number distributions
- model/prompt/run manifest
- concise conclusion with one of these statuses:

1. no meaningful bias found
2. bias found but only output-space correction works
3. activation steering works weakly
4. LoRA works but sparse localization fails
5. sparse/interpretable components causally control bias

Be conservative. The claim is not that the LLM becomes a true RNG. The claim is only about steering the model’s finite-choice distribution toward uniformity.
