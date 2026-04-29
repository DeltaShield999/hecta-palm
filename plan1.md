To make this publication-ready, I’d add five targeted experiment steps rather than expanding the whole project.

Mixed Benign + Attack Runtime Evaluation
Right now the filtered Stage 2 reruns are attack-only. Add an end-to-end run containing both normal fraud-scoring messages and adversarial extraction messages.

Report:

attack block rate
benign false positive rate
extraction success rate
percentage of normal transactions incorrectly blocked
results by exposure level: 1x, 10x, 50x
This is probably the most important missing publication step.

Adaptive Attacker Evaluation
Add a stronger attacker who knows the general scaffold: system prompt style, attack families, and that a filter exists.

Create a held-out/adaptive attack set with:

paraphrased extraction prompts
indirect requests
compliance/audit framing
requests designed to look like benign fraud workflow messages
prompt-leak-style attacks asking about guardrails or policies
Report whether the filter still blocks these and whether any leakage survives.

Confidence Intervals for Stage 2 and Stage 3
Your MIA results already have confidence intervals, but the extraction and filter results need them too.

Add bootstrap or binomial CIs for:

baseline leakage rate
system-prompt-active leakage rate
filtered leakage rate
adversarial block rate
benign false positive rate
per-family leakage rates
This makes the results look much more journal-grade.

Filter Ablation and Threshold Sensitivity
Show that the defense result is not a lucky threshold choice.

Run:

system prompt only
plaintext filter
FHE filter
several filter thresholds around the selected value
optionally: logistic regression vs simpler keyword/rule baseline
Report the tradeoff between blocking attacks and blocking benign traffic.

Generalization Check
Do one modest generalization experiment.

**Timing Plan**

Add a small timing layer that separates **setup cost**, **per-message filter cost**, and **full pipeline cost**. Keep the current per-message FHE scoring numbers, but make the latency story easier to interpret.

**1. Setup / One-Time Costs**

Record these once per run:

- sentence encoder load time
- plaintext classifier/model-parameter load time
- FHE bundle load time from disk
- FHE context/key load time
- FHE bundle creation time, if creating instead of reusing
- CKKS key generation time
- adapter/model load time for the LLM, if the full fraud scorer is included

Output to something like:

```text
runs/stage3/fhe/timing_setup.json
```

**2. Per-Message Filter Costs**

For each evaluated message, record:

- message read / parse time
- embedding generation time
- encryption time
- encrypted logistic-regression scoring time
- decryption time
- threshold / decision time
- response write / logging time
- total filter time

Keep these as row-level samples:

```text
runs/stage3/fhe/timing_filter_samples.csv
```

Suggested columns:

```text
message_id, embedding_ms, encryption_ms, fhe_scoring_ms, decryption_ms,
threshold_ms, io_ms, total_filter_ms
```

**3. Service / Network Mode Costs**

If the FHE scorer can run as a service, add an optional benchmark mode that measures:

- client serialization time
- request transfer time
- server queue / handling time
- FHE compute time on server
- response transfer time
- client deserialization time
- total round-trip time

Output:

```text
runs/stage3/fhe/timing_service_samples.csv
```

If there is no real network service yet, mark this as `local_in_process` and skip network fields.

**4. Full Pipeline Costs**

For integrated Stage 2 filtered reruns, record the full attack-path timing:

- message construction time
- filter total time
- LLM generation time, only if message is allowed
- routing / post-processing time
- total pipeline time

Output:

```text
runs/stage2/filtered/timing_pipeline_samples.csv
```

Suggested columns:

```text
attack_id, exposure_condition, filter_mode, filter_decision,
filter_total_ms, llm_generation_ms, routing_ms, total_pipeline_ms
```

For blocked messages, `llm_generation_ms` should be empty or `0`, but include a flag so summaries are clear.

**5. Summary Files**

For every timing CSV, generate a compact summary JSON with:

- count
- mean
- p50
- p90
- p95
- p99
- min
- max
- standard deviation

Example:

```text
runs/stage3/fhe/timing_filter_summary.json
runs/stage2/filtered/timing_pipeline_summary.json
```

**6. Reporting Rule**

In the paper, separate:

- **cold-start/setup costs**: model loading, keygen, bundle creation
- **warm per-message costs**: embedding, encryption, FHE scoring, decryption
- **full pipeline costs**: filter plus LLM generation where applicable

This avoids mixing one-time deployment costs with normal inference latency.