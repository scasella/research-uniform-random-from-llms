# BEE v0.4.2 — Solution-path diversity (GSM8K)

GSM8K test problems 0-24. Conditions: vanilla T=1.0, vanilla T=1.5, trained T=1.0. k=10 per cell.

Headline metric: among the correct samples in each cell, how many DISTINCT calculation paths (sorted multisets of intermediate numbers) appear.

## Per-condition summary (mean across problems)

| condition | n_problems | mean correct/k | mean distinct_path | 95% CI | median |
| --- | --- | --- | --- | --- | --- |
| `baseline_T1.0` | 25 | 9.80 | **7.40** | [6.52, 8.24] | 8.0 |
| `baseline_T1.5` | 25 | 9.68 | **8.40** | [7.68, 9.00] | 9.0 |
| `trained_T1.0` | 25 | 9.80 | **8.44** | [7.84, 9.08] | 9.0 |

## Paired comparison: trained vs each vanilla condition

| comparison | mean gap | 95% CI | wins / ties / losses (n problems) |
| --- | --- | --- | --- |
| trained_T1.0 - baseline_T1.0 | +1.04 | [+0.40, +1.68] | 13 / 9 / 3 |
| trained_T1.0 - baseline_T1.5 | +0.04 | [-0.60, +0.60] | 9 / 9 / 7 |

## Interpretation

- If `trained - baseline_T1.0` 95% CI excludes 0 with positive sign: the v0.4 LoRA increases solution-path diversity beyond the vanilla T=1.0 reference.
- If `trained - baseline_T1.5` 95% CI excludes 0 with positive sign: the training effect is **not** reachable by temperature scaling alone.
- If both CIs include 0: the path-diversity gap observed in the P0 smoke (+1.8 on 5 problems) does not survive a 25-problem replication; report null.

## Per-problem pivot

See `data/path_diversity_pivot.csv` for problem-by-problem distinct-path counts under each condition.
