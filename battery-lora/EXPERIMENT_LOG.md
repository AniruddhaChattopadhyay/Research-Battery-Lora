# BatteryLoRA Experiment Log

## Project: Battery-Aware Adaptive-Rank Federated LoRA
## Conference: NGEN-AI 2026 (Deadline: May 25, 2026)

---

## Setup & Environment
- **Machine**: Apple Silicon (MPS)
- **Python**: 3.12.9, managed with `uv`
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (4-bit quantized)
- **Framework**: Flower 1.27 + PEFT 0.18 + trl 0.29 + PyTorch 2.10

---

## Experiment 0: Quick Sanity Test (2026-03-13) — PASSED
- **Purpose**: Verify the full pipeline works end-to-end
- **Config**: 10 clients, 3 selected/round, 3 rounds, 1 local epoch, 50 samples/client
- **Bugs fixed**: 7 compatibility issues with trl v0.29 (see below)
- **Outcome**: Pipeline fully functional. Ranks correctly adapt to battery. Loss decreases.

### Bugs Fixed
1. `DataCollatorForCompletionOnlyLM` removed in trl v0.29 — implemented custom version
2. `tokenizer` → `processing_class` in SFTTrainer
3. `formatting_func` now per-example (not batched)
4. `SFTConfig` replaces `TrainingArguments`
5. BFloat16 → numpy needs `.float()` intermediate
6. LoRA adapter stacking — fixed with `.unload()` between clients
7. `torch_dtype` → `dtype` in transformers

---

## Quick-Mode Comparison (2026-03-13)

**Config**: 10 clients, 3 selected/round, 3 rounds, 1 epoch, 50 samples/client, seq_len=256

### Results Table

| # | Experiment | Policy | Final Loss | Energy (Wh) | Energy Std | Jain Fairness | Comm (MB) | Dropout |
|---|-----------|--------|-----------|-------------|------------|---------------|-----------|---------|
| E1 | **BatteryLoRA (ours)** | threshold | **1.2539** | **1.78** | 0.19 | 0.455 | 77.3 | 0% |
| E2 | BatteryLoRA-continuous | continuous | 1.2805 | 3.10 | 0.27 | 0.577 | 154.7 | 0% |
| A1 | Ablation: Binary | binary | 1.2604 | 2.04 | 0.21 | 0.491 | 92.4 | 0% |
| A2 | Ablation: Random | random | 1.3205 | 1.44 | 0.17 | 0.421 | 62.3 | 0% |
| B1 | HomLoRA r=8 | fixed r=8 | 1.2621 | 1.80 | 0.14 | 0.623 | 77.3 | 0% |
| B2 | HomLoRA r=32 | fixed r=32 | 1.2602 | **5.40** | 0.42 | 0.623 | **309.4** | 0% |
| B3 | HetLoRA | static_tier | 1.2714 | 3.60 | 0.36 | 0.496 | 189.1 | 0% |

### Key Observations (Quick Mode — preliminary, not for paper)

1. **Loss**: All methods achieve similar loss in quick mode (1.25-1.32). This is expected — 3 rounds with 50 samples isn't enough to differentiate. Full experiments needed.

2. **Energy**: BatteryLoRA (1.78 Wh) uses **67% less energy than HomLoRA-r32** (5.40 Wh) and **51% less than HetLoRA** (3.60 Wh), while achieving comparable loss.

3. **Communication**: BatteryLoRA (77.3 MB) uses **75% less bandwidth than HomLoRA-r32** (309.4 MB).

4. **Random ranks hurt quality**: Random policy has worst loss (1.3205), confirming that intelligent rank selection matters — it's not just rank diversity.

5. **Binary vs Threshold**: Binary (1.2604) is close to threshold (1.2539) in quick mode. Full experiments needed to see if 5-tier granularity helps.

6. **Fairness is low across the board**: Jain index is ~0.4-0.6 for all methods in quick mode. This is because only 3 of 10 clients participate per round (3 rounds total = 9 client-rounds, some clients never train). Full experiments with more rounds will show clearer fairness differences.

---

## Next Steps

### Full-Scale Experiments (TODO)
- [ ] Run all methods with full config: 50 clients, 10/round, 100 rounds, 3 epochs, full dataset
- [ ] Run with 3 seeds each (42, 123, 456) for statistical significance
- [ ] Non-IID sensitivity: alpha = 0.1, 0.5, 1.0
- [ ] Scale sensitivity: 10, 20, 50 clients
- [ ] Needs GPU with more VRAM or cloud instance for reasonable runtime

### Estimated Runtime for Full Experiments
- Quick mode: ~6 min per experiment
- Full mode (50 clients, 100 rounds): ~50 hours per experiment on MPS
- **Recommendation**: Use Google Colab Pro (A100) or university GPU cluster
- On A100: estimated ~2-4 hours per experiment

### Paper Writing
- [ ] Introduction
- [ ] Related Work
- [ ] Methodology
- [ ] Experiments & Results
- [ ] Conclusion
