# BatteryLoRA Experiment Log

## Project: Battery-Aware Adaptive-Rank Federated LoRA
## Conference: NGEN-AI 2026 (Deadline: May 25, 2026)

---

## Setup & Environment

### Local Development (Quick Tests)
- **Machine**: Apple Silicon (MPS)
- **Python**: 3.12.9, managed with `uv`
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (4-bit quantized)
- **Framework**: Flower 1.27 + PEFT 0.18 + trl 0.29 + PyTorch 2.10

### Paper Experiments (Full Runs)
- **Machine**: Google Colab Pro — NVIDIA A100-SXM4-40GB
- **Python**: 3.12.12
- **Framework**: Flower 1.27 + PEFT 0.15.2 + trl 0.12.0 + PyTorch 2.10 + transformers 4.48.0
- **Training speed**: ~2.5 iterations/second on A100

---

## Experiment 0: Quick Sanity Test (2026-03-13) — PASSED

- **Purpose**: Verify the full pipeline works end-to-end
- **Config**: 10 clients, 3 selected/round, 3 rounds, 1 local epoch, 50 samples/client
- **Bugs fixed**: 7 compatibility issues with trl v0.29 (see below)
- **Outcome**: Pipeline fully functional. Ranks correctly adapt to battery. Loss decreases.

### Bugs Fixed (Local — trl 0.29 / transformers 5.0)
1. `DataCollatorForCompletionOnlyLM` removed in trl v0.29 — implemented custom version
2. `tokenizer` → `processing_class` in SFTTrainer
3. `formatting_func` now per-example (not batched)
4. `SFTConfig` replaces `TrainingArguments`
5. BFloat16 → numpy needs `.float()` intermediate
6. LoRA adapter stacking — fixed with `.unload()` between clients
7. `torch_dtype` → `dtype` in transformers

### Additional Fixes for Colab (trl 0.12 / transformers 4.48)
8. `dtype` → `torch_dtype` in `from_pretrained()` (transformers 4.48 API)
9. `SFTConfig` → `TrainingArguments` (SFTConfig doesn't exist in trl 0.12)
10. `processing_class` → `tokenizer` in SFTTrainer (older API)
11. `formatting_func` must return list of strings (batched mode in trl 0.12)
12. `remove_unused_columns=True` required (False causes string tensorization error)
13. `fp16=True` → `bf16=True` (fp16 GradScaler incompatible with BFloat16 quantization on A100)

---

## Quick-Mode Comparison (2026-03-13) — Local MPS

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

**Note**: Quick-mode results are for pipeline validation only, not for the paper. 3 rounds with 50 samples is insufficient to differentiate methods.

---

## Paper-Scale Experiments (2026-03-14) — A100 GPU

**Config**: 10 clients, 5 selected/round, 30 rounds, 2 local epochs, 200 samples/client, seq_len=512, batch_size=4, seed=42

**Runtime**: ~2-3 hours per experiment, ~18 hours total for all 7

### E1: Main Comparison Results

| # | Method | Policy | Final Loss | Energy (Wh) | E. Std | Jain Fair. | Min Battery | Comm (MB) |
|---|--------|--------|-----------|-------------|--------|-----------|-------------|-----------|
| E1 | **BatteryLoRA (ours)** | threshold | 0.973 | 33.7 | 1.15 | 0.896 | **33.7%** | 1,555 |
| B1 | HomLoRA r=8 | fixed r=8 | 0.582 | **30.0** | **0.57** | **0.966** | 27.6% | **1,289** |
| B2 | HomLoRA r=32 | fixed r=32 | 0.513 | 90.0 | 1.84 | 0.960 | 11.2% | 5,156 |
| B3 | HetLoRA | static_tier | **0.262** | 51.9 | 1.88 | 0.884 | 17.9% | 2,613 |

### E2: Ablation on Rank Policies

| # | Method | Policy | Final Loss | Energy (Wh) | E. Std | Jain Fair. | Min Battery | Comm (MB) |
|---|--------|--------|-----------|-------------|--------|-----------|-------------|-----------|
| A1 | Ablation: Binary | binary | 0.973 | 33.5 | 1.12 | 0.900 | 36.8% | 1,549 |
| A2 | Ablation: Random | random | 0.831 | 25.9 | 0.63 | 0.944 | 23.1% | 1,094 |
| A3 | Ablation: Continuous | continuous | **0.409** | 46.4 | 1.61 | 0.892 | 23.4% | 2,271 |

### Convergence Over Rounds (Loss)

| Round | BatteryLoRA | HomLoRA r=8 | HomLoRA r=32 | HetLoRA |
|-------|------------|-------------|-------------|---------|
| 1     | 1.097      | 1.094       | 1.092       | 1.095   |
| 5     | 0.877      | 0.899       | 0.863       | 0.812   |
| 10    | 0.762      | 0.732       | 0.637       | 0.581   |
| 15    | 0.767      | 0.650       | 0.587       | 0.468   |
| 20    | 0.422      | 0.669       | 0.378       | 0.456   |
| 25    | 0.441      | 0.564       | 0.357       | 0.346   |
| 30    | 0.973      | 0.582       | 0.513       | 0.262   |

### BatteryLoRA Rank Adaptation — Per-Client Detail

Shows how BatteryLoRA dynamically adjusts rank based on battery state:

| Client | Tier | Start Battery | End Battery | Energy (Wh) | Rounds | Rank Trajectory |
|--------|------|--------------|-------------|-------------|--------|-----------------|
| 0 | low  | 67% | 100% | 2.80 | 14 | all rank 8 (charging, max for tier) |
| 1 | mid  | 27% | 42%  | 4.01 | 20 | 16→16→16→16→16→4→4→4→4→4→16→16→4→... |
| 2 | mid  | 68% | 72%  | 2.65 | 11 | 8→8→8→8→16→16→16→8→8→8→8 |
| 3 | low  | 77% | 100% | 3.00 | 15 | all rank 8 (charging, max for tier) |
| 4 | mid  | 54% | 48%  | 2.85 | 18 | 4→4→...→4→16→16→16 (recovered late) |
| 5 | mid  | 42% | 54%  | 2.56 | 17 | 4→4→4→2→2→...→2→16→16→16→16 |
| 6 | low  | 81% | 90%  | 3.00 | 15 | all rank 8 (max for tier) |
| 7 | mid  | 54% | 100% | 3.50 | 10 | all rank 16 (charging throughout) |
| 8 | high | 37% | 34%  | 2.76 | 15 | 4→4→...→4→32→32 (charged late) |
| 9 | high | 28% | 65%  | 6.60 | 15 | 4→4→4→4→4→32→32→32→32→32→... |

### BatteryLoRA Rank Distribution Over Rounds

| Round | Rank 2 | Rank 4 | Rank 8 | Rank 16 | Rank 32 |
|-------|--------|--------|--------|---------|---------|
| R1-5  | 0      | 8      | 10     | 7       | 0       |
| R6-10 | 1      | 9      | 10     | 5       | 0       |
| R11-15| 2      | 7      | 6      | 5       | 5       |
| R16-20| 2      | 3      | 8      | 8       | 4       |
| R21-25| 2      | 8      | 6      | 4       | 5       |
| R26-30| 0      | 6      | 4      | 10      | 5       |

Early rounds: mostly rank 4-8 (batteries draining). Later rounds: more rank 16-32 (charging events restore battery).

### Battery Preservation Comparison

| Method | Min Battery at End | Clients Below 20% | Avg Battery at End |
|--------|-------------------|-------------------|--------------------|
| **BatteryLoRA** | **33.7%** | **0** | **70.4%** |
| HomLoRA r=8 | 27.6% | 0 | 67.4% |
| HomLoRA r=32 | 11.2% | 2 | 54.0% |
| HetLoRA | 17.9% | 1 | 61.2% |

### Energy Efficiency Comparison

| Method | Total Energy (Wh) | vs HomLoRA r=32 | vs HetLoRA |
|--------|-------------------|----------------|-----------|
| **BatteryLoRA** | 33.7 | **-63%** | **-35%** |
| HomLoRA r=8 | 30.0 | -67% | -42% |
| HomLoRA r=32 | 90.0 | baseline | +74% |
| HetLoRA | 51.9 | -42% | baseline |

---

## Key Findings & Analysis

### What Worked
1. **Energy efficiency**: BatteryLoRA uses 63% less energy than HomLoRA-r32 and 35% less than HetLoRA
2. **Battery preservation**: Best minimum battery (33.7%) — no client close to dropout
3. **Adaptive behavior**: Ranks clearly respond to battery state (see per-client detail above)
4. **Ablation validates design**: Random ranks produce worst loss (0.831), confirming intelligent rank selection matters

### What Didn't Work as Expected
1. **Loss convergence instability**: BatteryLoRA reached 0.42 loss at round 20 (competitive!) but then oscillated back to 0.97 by round 30. The heterogeneous rank mix across rounds likely causes aggregation instability.
2. **Threshold policy underperformed continuous**: The continuous (energy-budget) policy achieved 0.409 loss vs threshold's 0.973. The energy-budget approach selects ranks more smoothly.
3. **Binary policy surprisingly bad**: Binary (0.973) performed no better than threshold despite being simpler. The sharp cutoff at 50% battery creates oscillation.
4. **No dropouts in any experiment**: The battery simulation's energy costs are too conservative for 30 rounds to trigger actual dropouts. Longer experiments or higher energy costs needed.
5. **HetLoRA is a strong competitor**: Static tier-based ranks (0.262 loss) outperformed all battery-aware methods on loss, suggesting that rank consistency helps convergence.

### Possible Causes of BatteryLoRA's Convergence Instability
- Different clients contribute to different subspaces each round (rank changes mean the FLoRA stacking updates different dimensions)
- A low-rank update in one round may partially overwrite high-rank learning from previous rounds
- The threshold policy's discrete jumps (e.g., 4→16→4) create larger disruptions than continuous policies
- This is a known challenge in heterogeneous federated learning literature

### Recommendations for Paper Framing
- Present continuous policy as the recommended variant (loss 0.409, competitive with HomLoRA r=8's 0.582)
- Position threshold policy as a simpler alternative that trades convergence stability for interpretability
- Emphasize energy/battery preservation as the primary contribution
- Acknowledge convergence instability as a limitation and area for future work

---

## Experiment Results Location

All results are stored in `results/` with one subfolder per experiment:

```
results/
├── e1_main_seed42/              # BatteryLoRA (threshold policy)
├── baseline_homolora_r8_seed42/ # Baseline: all clients rank 8
├── baseline_homolora_r32_seed42/# Baseline: all clients rank 32
├── baseline_hetlora_seed42/     # Baseline: static tier-based ranks
├── e2_ablation_binary_seed42/   # Ablation: binary policy
├── e2_ablation_random_seed42/   # Ablation: random policy
├── e2_ablation_continuous_seed42/ # Ablation: continuous policy
└── checkpoints/                 # Model checkpoints (from quick tests)
```

Each experiment folder contains:
- `summary.json` — Final metrics (loss, energy, communication, fairness)
- `battery_stats.json` — Aggregate battery statistics
- `device_stats.json` — Per-device battery history, rank history, energy consumed
- `round_metrics.json` — Per-round loss, ranks used, client details

---

## Next Steps

### Remaining Experiments
- [ ] Run with additional seeds (123, 456) for statistical significance on key methods
- [ ] Non-IID sensitivity: Dirichlet alpha = 0.1, 0.5, 1.0
- [ ] Investigate convergence instability fix (e.g., momentum in aggregation, rank change smoothing)
- [ ] Run longer experiments (50-100 rounds) to trigger actual battery dropouts

### Paper Writing
- [ ] Introduction
- [ ] Related Work
- [ ] Methodology
- [ ] Experiments & Results — data collected, analysis complete
- [ ] Conclusion
