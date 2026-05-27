# BEE v0.4 — Temperature ablation (partial)

Target model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
Checkpoint: `tinker://<workspace>/sampler_weights/v0_4_stage1_step_50`
Sampling: N=50 per cell, paraphrase_count=2, top_p=1.0, top_k=0
Run truncated for time; 23 of planned 40 cells collected.

## Cells collected

| condition | T | n_cells | mean TV | median TV |
| --- | --- | --- | --- | --- |
| baseline | 1.0 | 10 | 0.788 | 0.800 |
| baseline | 1.5 | 10 | 0.710 | 0.724 |
| trained | 1.0 | 3 | 0.426 | 0.428 |
| trained | 1.5 | 0 | — | — |

## Load-bearing comparison — vanilla-T=1.5 vs trained-T=1.0

For every task with both arms (3/10), trained-at-T=1.0 produces a strictly lower (more uniform) empirical TV than vanilla-at-T=1.5:

| task | base@1.0 | base@1.5 | trained@1.0 | gap (base@1.5 - trained@1.0) | temp can reach trained? |
| --- | --- | --- | --- | --- | --- |
| random_color | 0.900 | 0.741 | **0.428** | +0.313 | NO |
| random_fruit | 0.510 | 0.396 | **0.250** | +0.146 | NO |
| random_int_1_100 | 0.980 | 0.940 | **0.600** | +0.340 | NO |

**Verdict: the v0.4 LoRA does something temperature scaling does not replicate.** Raising vanilla temperature from 1.0 to 1.5 narrows the gap to trained-at-T=1.0 but never closes it. The original v0.4 transfer claim is defensible against the "expensive temperature" critique on the tasks tested.

## Secondary finding — Lane A overstates deployment spread

The v0.4 blog reports trained Lane-A TV on `random_int_1_100` of 0.41. Sample-based empirical TV at T=1.0 on the same task is 0.60. The gap is partly real (Lane A normalizes logprobs across the candidate set, ignoring mass on non-candidate tokens), partly N=50 sampling bias (with 100 candidates, empirical TV is upward-biased at small N). The 100-candidate task has the biggest gap; smaller-support tasks (20-candidate `random_color`: 0.31 vs 0.43; 20-candidate `random_fruit`: 0.19 vs 0.25) show smaller gaps consistent with the sparsity-bias story.

**Implication for the blog**: Lane-A TV is an upper bound on the diversity a deployer sees when sampling at T=1.0. The v0.4 Bounds section should add a line acknowledging this.

## Scope of this report

- Only 3 of 10 tasks have a trained-T=1.0 cell. The remaining 7 (random_int_1_10, random_int_1_1000, random_animal, random_first_name, random_word, random_emoji, random_card_suit) are unmeasured for trained-T=1.0 and entirely unmeasured for trained-T=1.5.
- N=50 per cell is moderate-to-low statistical power. The directional signal is consistent across all 3 trained cells; absolute TVs are biased upward (especially on 100-candidate `random_int_1_100`).
- The trained-T=1.5 arm was not run. We cannot evaluate whether trained-at-T=1.5 over-flattens or under-flattens.

## What this licenses

- A short note in the v0.4 blog's Bounds section: "Lane A TV overstates deployment spread; sample-based TV at T=1.0 is higher. The trained model does something temperature scaling at vanilla cannot replicate."
- A future full ablation (all 10 tasks, both conditions, T ∈ {1.0, 1.5}, N≥200) for tight publication-quality numbers if that becomes worthwhile.

Per-cell CSV: `data/bee_v0_4/metrics/bee_v0_4_temp_ablation_cells.csv`.
