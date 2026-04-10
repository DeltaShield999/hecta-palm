# Multi-Agent FHE Privacy Experiment

Engineering Implementation Guide

Architecture, Fixed Constraints, Component Interfaces, and Evaluation Criteria

| Field | Value |
| --- | --- |
| Audience | Engineering / Implementation Team |
| Companion | Research Proposal v4 — read for full scientific context |
| Purpose | Define what must be built, the fixed constraints, and how success is measured |
| Flexibility | Implementation choices (libraries, frameworks, infra) are yours unless marked FIXED |

## 1. What We Are Building

Three sequential stages. Each stage produces a result that feeds into the next. The final output is a before/after comparison showing that an FHE-protected prompt filter blocks an attack that a system prompt alone cannot stop.

| Stage | Build | Produces |
| --- | --- | --- |
| Stage 1 | Fine-tune the Fraud Scoring Agent (Qwen2) on synthetic financial data with injected canary records. Run a reference-based Membership Inference Attack against it. | MIA metrics (AUC-ROC, TPR@1%FPR, TPR@10%FPR) at three exposure levels. Confirms the model memorises canaries. |
| Stage 2 | Wire up the three-agent pipeline. Simulate a compromised Transaction Intake Agent sending adversarial extraction prompts to the Fraud Scoring Agent through the inter-agent channel. | Canary extraction success rate with system prompt guardrail active. Confirms the channel can be exploited even with an LLM-level guardrail. |
| Stage 3 | Train a logistic regression classifier on labelled inter-agent messages. Compile it to a CKKS/OpenFHE-based encrypted circuit. Insert it as a filter on the intake-to-fraud message channel. Re-run Stage 2. | Filter block rate, extraction success rate under filter, FHE inference latency. Confirms FHE filter blocks the attack. |

## 2. Fixed Constraints

The following are set by the research design and cannot be changed without PI approval. Everything else — orchestration framework, cloud provider, training library choices — is an engineering decision.

> **These are fixed — do not substitute**
>
> Model family: Qwen2 (1.5B-Instruct as primary, 7B-Instruct as scale comparison). These are the models the MIA is evaluated on and the models the sponsor's FHE solution targets.
>
> FHE scheme: CKKS-based FHE via OpenFHE (or a compatible library — see Section 4). The CKKS scheme is chosen for its support of approximate arithmetic over real-valued features, which is required for logistic regression on embedding vectors.
>
> Exposure conditions: exactly 1x, 10x, and 50x canary injection. These three levels are the ablation design — do not add or remove conditions.
>
> MIA evaluation metrics: AUC-ROC, TPR@1%FPR, and TPR@10%FPR. These are the standard MIA metrics in the literature. All three must be reported.
>
> Member/non-member split: strict — 8,000 training records, 2,000 held-out non-members, zero overlap. The split cannot be changed between Stage 1 training and Stage 1 evaluation.
>
> Canary structure: canaries must be Tier 1 records containing synthetic PII fields (name, DOB, account number). The extraction attack in Stage 2 targets these specific fields.

> **On CKKS vs TFHE**
>
> CKKS (via OpenFHE or a compatible library) is the scheme specified for this experiment because it operates natively over floating-point feature vectors from the sentence embedder. TFHE (e.g. Zama's Concrete ML) is an alternative that is easier to use and well-suited to logistic regression on tabular data — if the sponsor confirms their infrastructure uses TFHE, it is acceptable to use Concrete ML instead.
>
> Confirm with the PI and the sponsor before making this choice. The scheme used must be documented clearly in the results.

## 3. Architecture

### 3.1 The Three-Agent Pipeline

Only one of the three agents is an LLM. The other two are lightweight deterministic programs. This is intentional — it isolates all privacy risk and experimental complexity to the Fraud Scoring Agent.

| Agent | Type | Description | In Stage 2 attack |
| --- | --- | --- | --- |
| Transaction Intake Agent | Deterministic (scripted) | Formats raw transaction events into inter-agent messages. Passes them to the Fraud Scoring Agent. | Simulated as compromised: sends adversarial extraction prompts instead of legitimate messages. |
| Fraud Scoring Agent | LLM — Qwen2, LoRA fine-tuned | The only ML component. Receives inter-agent messages, scores fraud risk. Has memorised canary data during fine-tuning. | Primary target. Receives adversarial prompts and may reproduce memorised canary data in its response. |
| Case Routing Agent | Deterministic (scripted) | Reads the fraud score from Agent 2's response and routes the transaction to ALLOW, REVIEW, or BLOCK. | Passive — not involved in the attack. |

### 3.2 Pipeline Message Flow

The pipeline has two versions — one for Stage 2 (no filter) and one for Stage 3 (filter inserted on the intake-to-fraud channel):

```text
Stage 2 (no filter):

    [Transaction Intake] ---- message ----> [Fraud Scoring Agent] ---- score
----> [Case Routing]
      (compromised: sends                  (LLM, fine-tuned,
       adversarial prompt)                 memorised canaries)


Stage 3 (FHE filter active):

    [Transaction Intake] ---- message ----> [FHE Prompt Filter] ---- ALLOW
----> [Fraud Scoring Agent] ---> [Case Routing]
      (compromised)                         (encrypted classify)     BLOCK
----> [END - message dropped]

The FHE filter sits ON THE EDGE between Agent 1 and Agent 2.
It intercepts every message before the LLM sees it.
The classification decision (ALLOW/BLOCK) is computed entirely on ciphertext.
```

### 3.3 FHE Filter Architecture

The filter uses a two-step architecture. Only Step 2 involves FHE. This is the standard pattern for encrypted text classification — you do not need to run the embedder under FHE.

| Step | What runs | Where | Sensitivity |
| --- | --- | --- | --- |
| Step 1: Embedding | A sentence encoder converts the inter-agent message text to a fixed-size float vector (e.g. 384-dim). The encoder itself is a public pre-trained model. | Plaintext, on a trusted internal node | Not sensitive — the encoder is public and the embedding alone does not reveal the policy decision. |
| Step 2: FHE Classification | The float vector is encrypted by the client. The CKKS-compiled logistic regression classifier runs on the ciphertext server-side. The server returns an encrypted ALLOW/BLOCK decision. The client decrypts. | Encrypted — client encrypts, server computes on ciphertext, client decrypts | The policy (classifier weights) and the decision are both encrypted. The server never observes plaintext features or outputs. |

> **Why this split makes sense**
>
> The sensitive object is not the message text (the attacker already has that — they sent it). The sensitive objects are (1) the institution's proprietary risk policy, encoded in the classifier weights, and (2) the ALLOW/BLOCK decision, which could leak policy logic if observed in plaintext by the server. CKKS FHE protects both.
>
> The sentence embedder runs in plaintext because it is a public model — there is nothing to protect. Running it under FHE would add significant overhead for no security benefit.

## 4. Component Specifications

### 4.1 Qwen2 Fine-Tuning (Stage 1)

Use LoRA fine-tuning. The key parameters below are fixed by the research design. Hyperparameters like learning rate, batch size, and number of epochs are engineering choices — tune for convergence, not for a specific accuracy target.

| Parameter | Fixed value | Rationale |
| --- | --- | --- |
| Base model (primary) | Qwen2-1.5B-Instruct | FIXED — specified in research design |
| Base model (secondary) | Qwen2-7B-Instruct | FIXED — scale comparison |
| Fine-tuning method | LoRA applied to attention projection layers | FIXED — LoRA is chosen deliberately as a conservative choice (it reduces memorisation vs full fine-tuning, making the MIA result a lower bound) |
| Training data format | Qwen2 native chat template / system/user/assistant turns | FIXED — must match exactly at inference time for the MIA loss computation to be valid |
| Exposure conditions | 1x, 10x, 50x canary injection (independent runs per condition) | FIXED — the ablation design |
| System prompt | Applied at inference time: instructs model not to reveal customer PII | FIXED content — this is the guardrail that Stage 2 shows is insufficient |
| LoRA rank, alpha, target modules | Engineering choice | Tune for convergence. Suggested starting point: rank 16, alpha 32, q/k/v/o projections |
| Learning rate, batch size, epochs | Engineering choice | Tune for convergence. Monitor training loss in W&B or equivalent. |

### 4.2 Membership Inference Attack (Stage 1)

The attack is a reference-based loss ratio attack. The formula is fixed. The implementation is yours.

#### Attack formula

```text
For each record x in the evaluation set (members + non-members):

    loss_base = per-token cross-entropy loss of the BASE Qwen2 checkpoint on x
    loss_ft   = per-token cross-entropy loss of the FINE-TUNED model on x

    membership_score(x) = loss_base / loss_ft

    A higher score indicates membership:
    the fine-tuned model has lower loss on records it was trained on.

Use the SAME prompt format (chat template) for both loss computations.
Mismatch between training format and eval format is the most common failure mode.
```

#### Required metrics — all three must be reported

| Metric | Definition | Why it matters |
| --- | --- | --- |
| AUC-ROC | Area under the ROC curve, treating `membership_score` as the classifier output and `is_member` as the label. | Overall attack strength across all thresholds. Random = 0.5, perfect = 1.0. |
| TPR @ 1% FPR | True positive rate (member correctly identified) at the operating point where false positive rate = 1% | The standard privacy metric — tells you how many real members are identified when the attacker accepts only 1 false alarm per 100 non-members. This is the number regulators care about. |
| TPR @ 10% FPR | True positive rate at FPR = 10% | More permissive operating point — useful for showing the attack strength at a more realistic adversarial setting. |

#### Evaluation set

- 8,000 member records (`is_member = 1`) — the full training pool
- 2,000 non-member records (`is_member = 0`) — the held-out set
- Report results per exposure condition (1x, 10x, 50x) and per model size (1.5B, 7B)

### 4.3 Synthetic Data

The data has two tiers. The structure below is fixed because the canary fields must be present for the Stage 2 extraction check and the Stage 3 filter training.

#### Tier 1 — Transaction records

- **Required fields:** `account_id`, `customer_name`, `date_of_birth`, `account_number`, `amount`, `merchant_category`, `timestamp`, `geo_location`, `device_fingerprint`, `is_fraud_label`
- 10,000 records total. Strict 8,000 / 2,000 member / non-member split. Never overlap.
- ~3% fraud base rate. Generation method is your choice (Faker, SDV, or manual templates).
- **Canaries:** designate ~100 records from the member set as canaries. Register them separately. These are the records Stage 2 attempts to extract and Stage 1 evaluates against.

#### Tier 2 — Inter-agent message logs

- Natural language conversational turns between the Transaction Intake Agent and the Fraud Scoring Agent.
- Each record must embed the Tier 1 fields (especially `account_id`, `customer_name`, `date_of_birth`, `account_number`) in natural language — these are the canary signals the model memorises.
- These become the LoRA fine-tuning corpus. Format using the Qwen2 chat template (system/user/assistant).
- Canaries appear in the corpus at their designated exposure frequency (1x / 10x / 50x per training run).

#### Adversarial extraction messages (Stage 2 / Stage 3 filter training)

- A labelled dataset of inter-agent messages: benign operational messages (`label=ALLOW`) and adversarial extraction prompts (`label=BLOCK`).
- Adversarial messages should use varied framing: direct instruction injection, compliance/audit framing, role-play, urgency. Variety matters for filter robustness.
- This labelled dataset is the training input for the Stage 3 logistic regression classifier.

### 4.4 FHE Prompt Filter (Stage 3)

The filter is a logistic regression classifier compiled to a CKKS FHE circuit. The steps below are fixed. The specific library implementation within CKKS is your choice.

#### Step 1 — Sentence embedding (plaintext)

- Convert each inter-agent message to a fixed-size float vector using a pre-trained sentence encoder.
- The encoder model is your choice — any model that produces a reasonably dense semantic representation (e.g. a MiniLM or similar lightweight model) is appropriate.
- Pin the encoder model version. The same model must be used at training time, filter compilation time, and inference time.

#### Step 2 — Logistic regression classifier

- Train in plaintext on the labelled message dataset from Section 4.3.
- Measure and report plaintext accuracy (F1, precision, recall for both classes) before compiling to FHE — this is the accuracy ceiling the FHE version will be compared against.

#### Step 3 — CKKS compilation

- **FHE scheme:** CKKS. Required because the classifier operates on float-valued embedding vectors. CKKS supports approximate arithmetic over real numbers. OpenFHE is the reference implementation.
- Compile the trained logistic regression to a CKKS-compatible circuit. This involves quantising weights to integers — measure and report any accuracy degradation vs the plaintext model.
- The compilation output is a deployable FHE circuit. Save it — it is used at inference time.

#### Step 4 — Client-server inference

```text
Client side (financial institution):
  1. Receive inter-agent message text
  2. Embed in plaintext (sentence encoder)
  3. Encrypt the embedding vector using the CKKS public key
  4. Send ciphertext to FHE server
  5. Receive encrypted result
  6. Decrypt; decision is ALLOW or BLOCK
  7. If ALLOW: forward message to Fraud Scoring Agent.
     If BLOCK: drop message, log the blocked attempt

Server side (FHE inference node):
  1. Receive encrypted embedding ciphertext
  2. Run compiled CKKS logistic regression circuit on ciphertext
  3. Return encrypted result
  4. Server never observes plaintext at any point
```

#### What to benchmark

- FHE inference latency per message (milliseconds). Report mean and distribution.
- Plaintext classifier accuracy vs FHE classifier accuracy — the delta is the quantisation cost.
- Filter true positive rate on adversarial messages (block rate).
- Filter false positive rate on benign messages (legitimate messages incorrectly blocked).

## 5. Success Criteria

These are the numbers the experiment needs to produce to support the research conclusions. They are targets, not hard requirements — if a result differs significantly from expectation, document it and flag it to the PI rather than adjusting the experiment to hit the number.

### 5.1 Stage 1 — MIA Results

| Exposure | AUC-ROC target | TPR@1%FPR target | Conclusion supported |
| --- | --- | --- | --- |
| 1x | 0.50 – 0.62 | ~1–3% | LoRA at low exposure provides meaningful privacy reduction — establishes the conservative lower bound |
| 10x | 0.70 – 0.85 | ~10–25% | Meaningful membership signal — attack succeeds at practically significant rates |
| 50x | 0.85 – 0.95+ | ~30–60%+ | Strong memorisation — production-style repeated fine-tuning creates severe exploitable risk |

> **If Stage 1 results are unexpected**
>
> AUC near 0.5 at 50x: most likely cause is a chat template mismatch between fine-tuning and MIA evaluation. The model's loss at eval time must be computed on identical-format text to training.
>
> AUC above 0.95 at 1x: the canary injection logic may be injecting at higher frequency than intended. Verify the corpus size and canary count.

### 5.2 Stage 2 — Extraction Attack Results

| Condition | Target | Notes |
| --- | --- | --- |
| Extraction rate (no guardrail) | > 20% across adversarial templates | Baseline — establishes that memorised canaries are extractable |
| Extraction rate (system prompt active) | > 0%, partial bypass expected | This is the key result — shows system prompt alone is insufficient. If 0%, adversarial templates may be too obvious; vary the framing. |
| Most effective attack type | Compliance or audit framing | These templates mimic legitimate inter-system communication most closely |

### 5.3 Stage 3 — FHE Filter Results

| Metric | Target | Notes |
| --- | --- | --- |
| Adversarial message block rate | > 90% | The filter should block the vast majority of extraction prompts. Report per template type. |
| Extraction success rate (filter active) | < 2% | Near-zero — residual extraction is from filter false negatives. Any non-zero result should be analysed. |
| Legitimate message false positive rate | < 5% | The filter must not significantly disrupt normal pipeline operation |
| FHE inference latency | Report actual numbers | No hard target — report mean, p50, p95. Acceptable for batch workflows at any latency under ~30s/message. |
| Plaintext vs FHE accuracy delta | < 3% | Quantisation should have minimal impact. If > 5%, investigate `n_bits` setting. |

## 6. Deliverables

The following must be produced and handed off to the PI before the experiment is considered complete. Engineering implementation choices (code structure, libraries used) are not deliverables — only results and artefacts are.

### 6.1 Stage 1

- AUC-ROC and TPR@1%FPR / TPR@10%FPR for all six runs (1.5B and 7B × 1x, 10x, 50x)
- ROC curves (plots) for all six runs with AUC values annotated
- Raw loss values (member and non-member) for at least one run — used to verify the membership score distribution
- Training loss curves (W&B or equivalent) confirming model convergence for each run

### 6.2 Stage 2

- Extraction success rate per adversarial template category (direct injection, compliance framing, role-play, urgency, subtle)
- Overall extraction rate with system prompt active vs without
- 5–10 representative examples of successful extraction responses (with canary PII fields highlighted)

### 6.3 Stage 3

- Plaintext classifier performance report: F1, precision, recall for ALLOW and BLOCK classes
- FHE classifier performance report: same metrics post-compilation — quantisation accuracy delta
- FHE inference latency measurements: mean, p50, p95 over at least 100 message evaluations
- Filtered attack results: block rate, false positive rate, extraction success rate under filter
- The compiled FHE circuit files — saved and accessible for reproduction

> **For the PI write-up**
>
> The PI needs numbers, not prose. Deliver results as structured JSON or CSV files alongside any plots. Include the exact run configuration (exposure condition, model size, any hyperparameters that differ from defaults) as metadata in each results file.

## 7. Open Engineering Questions

These decisions are yours to make. If a choice significantly affects results, document it.

| Decision | Constraints | Guidance |
| --- | --- | --- |
| Multi-agent orchestration framework | None fixed — choose what works | Needs explicit inter-agent message passing and the ability to insert middleware (the FHE filter) on a specific edge. LangGraph, custom asyncio Python, or any equivalent works. |
| Cloud / compute provider | None fixed — Vast.ai is suggested but not required | H100 or A100 GPU needed for 7B fine-tuning. CPU is sufficient for FHE classifier training and Stage 3 filter inference. |
| CKKS library | Must use CKKS scheme. OpenFHE is reference implementation. | OpenFHE is the most complete and actively maintained open-source CKKS implementation. SEAL (Microsoft) is an alternative. Concrete ML (Zama, TFHE-based) is acceptable if confirmed with PI and sponsor. |
| Sentence encoder for FHE filter | Must produce float vectors as input to the CKKS classifier | Any pre-trained sentence encoder is acceptable. Smaller is better for FHE latency — 384-dim embeddings (e.g. MiniLM variants) are a good starting point. |
| LoRA hyperparameters | Rank, alpha, dropout are engineering choices | Start with rank 16, alpha 32. Monitor training loss. The goal is convergence, not a specific accuracy target. |
| Adversarial template diversity | Must cover at least 3 distinct framing types | More diverse templates = more robust filter training set and more informative Stage 2 results. Aim for at least 5 template categories. |

## 8. Questions and Escalation

Two types of questions arise during implementation:

- **Research / scope questions** (e.g. "can we change the exposure conditions?", "can we use a different model family?"): bring these to the PI. These touch the fixed constraints and need sign-off.
- **Engineering / implementation questions** (e.g. "which CKKS library is easiest to deploy?", "should we use LangGraph or raw Python?"): decide within the team. Document the choice and reasoning in the results files.

> **If a result is unexpected**
>
> Flag it to the PI immediately rather than re-running until it matches expectations. Unexpected results (e.g. AUC lower than predicted at 50x, FHE latency higher than expected) are scientifically important and need to be understood, not hidden. The PI will want to know the actual numbers.
