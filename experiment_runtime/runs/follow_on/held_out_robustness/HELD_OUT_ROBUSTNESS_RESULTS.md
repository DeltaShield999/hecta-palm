# Held-Out Adaptive Robustness Results

This is the final held-out robustness check for the `Qwen2-1.5B-Instruct` `50x` condition. It reuses the frozen `50x` adapter, frozen Stage 3 filter, existing plaintext and CKKS/OpenFHE filter implementations, and the two pre-specified thresholds: the conservative Stage 3 threshold `0.4199950085320943` and the utility-calibrated candidate `0.72`.

No model retraining, filter retraining, prompt revision after results, or threshold tuning was performed for this pass.

## Dataset

- Held-out adaptive attacks: `343` prompts.
- Taxonomy: seven scaffold-aware adaptive families, `49` prompts per family.
- Target fields: `customer_name`, `date_of_birth`, and `account_number`.
- Target bundle distribution per family: `5` single-field prompts for each field, `5` two-field prompts for each field pair, and `19` full-record prompts.
- Benign hard negatives: `200` fraud-operations messages.
- Mixed evaluation set: `543` rows, with `200` benign rows and `343` adaptive adversarial rows.
- Disjointness: exact message-text and exact request-line checks passed against the original Stage 2 attacks, the prior follow-on adaptive attacks, and Stage 3 filter train/validation/test rows.

## 50x Adaptive Results

| Condition | Threshold | Any-field leak | Full-record leak | Block rate | Leak given allowed |
| --- | ---: | ---: | ---: | ---: | ---: |
| no system prompt | n/a | `234 / 343 = 68.22%` | `101 / 343 = 29.45%` | n/a | n/a |
| system prompt only | n/a | `9 / 343 = 2.62%` | `2 / 343 = 0.58%` | n/a | n/a |
| plaintext filter | `0.419995` | `2 / 343 = 0.58%` | `0 / 343 = 0.00%` | `270 / 343 = 78.72%` | `2 / 73 = 2.74%` |
| FHE filter | `0.419995` | `2 / 343 = 0.58%` | `0 / 343 = 0.00%` | `270 / 343 = 78.72%` | `2 / 73 = 2.74%` |
| plaintext filter | `0.72` | `4 / 343 = 1.17%` | `0 / 343 = 0.00%` | `185 / 343 = 53.94%` | `4 / 158 = 2.53%` |
| FHE filter | `0.72` | `4 / 343 = 1.17%` | `0 / 343 = 0.00%` | `185 / 343 = 53.94%` | `4 / 158 = 2.53%` |

## Mixed Hard-Negative Results

| Threshold | Filter mode | Benign false positive | Benign allow | Adaptive block | Adaptive any-field leak |
| ---: | --- | ---: | ---: | ---: | ---: |
| `0.419995` | plaintext/FHE | `74 / 200 = 37.00%` | `126 / 200 = 63.00%` | `270 / 343 = 78.72%` | `2 / 343 = 0.58%` |
| `0.72` | plaintext/FHE | `32 / 200 = 16.00%` | `168 / 200 = 84.00%` | `185 / 343 = 53.94%` | `4 / 343 = 1.17%` |

## Plaintext/FHE Parity

Plaintext and CKKS/OpenFHE filter decisions matched exactly in all held-out filtered runs.

| Run | Decision matches | Mismatches | Mean probability drift | Max probability drift |
| --- | ---: | ---: | ---: | ---: |
| conservative adaptive | `343 / 343` | `0` | `3.087e-09` | `1.868e-08` |
| threshold `0.72` adaptive | `343 / 343` | `0` | `2.922e-09` | `1.660e-08` |
| conservative mixed | `543 / 543` | `0` | `3.224e-09` | `2.902e-08` |
| threshold `0.72` mixed | `543 / 543` | `0` | `3.028e-09` | `1.904e-08` |

## Filter Timing

| Run | Plaintext mean / p95 total filter time | FHE mean / p95 total filter time |
| --- | ---: | ---: |
| conservative adaptive | `11.55 / 13.09 ms` | `76.85 / 79.39 ms` |
| threshold `0.72` adaptive | `13.23 / 17.36 ms` | `80.78 / 92.46 ms` |
| conservative mixed | `13.45 / 17.56 ms` | `83.12 / 102.25 ms` |
| threshold `0.72` mixed | `13.05 / 17.23 ms` | `80.92 / 90.77 ms` |

## Interpretation

The held-out set supports the central defense story but narrows the strongest wording. The frozen conservative filter substantially reduces `50x` held-out adaptive leakage and keeps full-record leakage at zero, but it does not eliminate all any-field leakage on this fresh held-out set. Threshold `0.72` improves benign utility and lowers the adaptive block rate, but it allows more any-field leakage than the conservative baseline.

The CKKS/OpenFHE result remains clean: the FHE path preserves plaintext filter decisions exactly, with negligible probability drift. The main limitation is still utility and generalization under synthetic fraud-operations traffic: benign hard-negative false positives are `37.00%` at the conservative threshold and `16.00%` at threshold `0.72`.

This pass should be reported as a held-out robustness check, not as a defense-improvement or threshold-selection pass. The held-out traffic is synthetic and exact disjointness checks do not prove semantic disjointness.
