Update the experiment so the main evaluation includes a scaffold-aware adaptive attacker.

First, inspect the current Stage 2 and Stage 3 data generation, evaluation, and result-summary code. Preserve the existing frozen baseline attack set and all current results; add the adaptive evaluation as a new experiment layer rather than replacing the old one.

Implement a new held-out adaptive attack dataset. The attacker should be assumed to know the agent scaffold, the fraud-scoring system prompt style, the allowed response format, and that obvious PII extraction requests may be filtered. Add several adaptive attack families, such as scaffold probing, guardrail/policy extraction, compliance reframing, fraud-analysis-disguised extraction, partial-field verification, debug/log framing, and indirect inference. Keep this dataset disjoint from the existing Stage 2 attack set and from Stage 3 filter training examples.

Extend the Stage 2 evaluation CLI or add a parallel CLI so it can run both baseline and adaptive attack sets under the existing conditions:
- no system prompt
- system prompt active
- system prompt + plaintext filter
- system prompt + FHE filter

Run these across the existing exposure levels:
- 1x
- 10x
- 50x

Add a mixed-traffic evaluation mode. It should evaluate benign operational messages together with adaptive adversarial messages, so we can report both attack blocking and legitimate-message false positives in the integrated runtime path.

Add confidence intervals for the attack/filter metrics. At minimum, report confidence intervals for leakage rate, adversarial block rate, benign false positive rate, and filtered extraction rate. Include per-family adaptive attack metrics where practical.

Expand timing instrumentation. Keep the current FHE scoring latency, but also record setup/cold-start timing and warm per-message timing. Include embedding generation, encryption, FHE scoring, decryption, thresholding, file I/O/logging, optional service overhead if applicable, and LLM generation time for allowed messages. Report setup costs separately from per-message costs.

Update result artifacts and documentation. Add clear JSON/CSV summaries for the adaptive attack evaluation, mixed-traffic evaluation, confidence intervals, and expanded timing. Update RESULTS.md with a concise new section explaining the adaptive attacker setup, headline metrics, and remaining limitations.

Do not over-refactor the repo. Follow existing config, artifact, and CLI patterns. Keep the implementation reproducible, deterministic where possible, and compatible with the current baseline experiment flow.