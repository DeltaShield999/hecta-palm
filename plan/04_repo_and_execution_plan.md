# Repo Structure and Execution Plan

## 12. Recommended Repo Structure

Use `experiment_runtime/` as the runtime package for the LangGraph harness instead of creating a second unrelated orchestration project.

```text
experiment_runtime/
  configs/
  data/
    raw/
    processed/
  runs/
  scripts/
  src/
    qwen_langgraph_demo/
      graph/
      nodes/
      tools/
      runtime/
    experiment/
      schemas/
      data_gen/
      chat_render/
      train_qwen/
      mia/
      attacks/
      filter_train/
      fhe/
      eval/
  tests/
```

Implementation rules:

- use `uv` with Python `3.12`
- every script takes a config path
- every run writes to an isolated run folder
- pin seeds for every generation, train, and eval job
- write metrics as JSON or CSV, not only notebooks
- treat the current translation demo as scaffolding to replace or refactor, not as the target workflow

## 13. Phased Implementation Plan

### Phase 0: Spec Freeze

- materialize the frozen protocol constants from Sections 8 through 10 into config files
- save the exact system prompt text
- save the exact chat template
- save the exact label policy

### Phase 1: Data Layer

- implement schemas
- generate Tier 1 records
- register canaries
- materialize Tier 2 corpora
- materialize MIA eval corpus
- materialize Stage 2 and Stage 3 message datasets
- implement validators

### Phase 2: Stage 1

- implement LoRA training for `Qwen2-1.5B-Instruct`
- train `1x`, `10x`, and `50x`
- implement MIA evaluator
- save losses, ROC data, and summary metrics

### Phase 3: Stage 2

- implement the deterministic pipeline harness
- implement attack replay by family
- implement leakage scorer
- run no-system-prompt and system-prompt conditions

### Phase 4: Stage 3 Plaintext Filter

- embed Stage 3 dataset
- train logistic regression
- choose threshold on validation
- evaluate on held-out test set

### Phase 5: Stage 3 FHE Filter

- compile classifier for CKKS runtime
- run encrypted scoring on embeddings
- compare plaintext vs FHE outputs
- benchmark latency

### Phase 6: Integrated Final Runs

- rerun Stage 2 attacks with filter active
- package Stage 1, Stage 2, and Stage 3 deliverables
- repeat the completed flow for `Qwen2-7B-Instruct`

## 14. Ticket Order For Future Codex Sessions

Recommended ticket sequence:

1. schemas and config system
2. Tier 1 generator and validators
3. canary registry generation
4. Tier 2 chat renderer
5. Stage 2 attack message generator
6. Stage 3 ALLOW/BLOCK dataset generator
7. Stage 1 LoRA training pipeline
8. Stage 1 MIA evaluator
9. Stage 2 harness and leakage scorer
10. Stage 3 plaintext filter training
11. Stage 3 FHE wrapper
12. final evaluation and packaging

## 15. Non-Goals

These are out of scope for v1 unless the designer explicitly expands the project:

- encrypting the full LLM
- encrypting the sentence embedder
- using the papers or notebooks as implementation templates
- proving that FHE itself improves block rate relative to a plaintext filter

The last point is why the plaintext-filter baseline is recommended: it improves interpretability, but it is not required by the original guide.
