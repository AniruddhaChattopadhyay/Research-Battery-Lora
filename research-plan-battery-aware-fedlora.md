# Research Plan: Battery-Aware Adaptive-Rank Federated LoRA for On-Device SLM Personalization

## A Complete Guide for the Team (Assumes No Prior AI Background)

---

## Table of Contents

1. [The Big Picture — What Are We Building?](#1-the-big-picture)
2. [Background Concepts — Everything You Need to Know](#2-background-concepts)
3. [Why This Is Novel — The Gap Nobody Has Filled](#3-why-this-is-novel)
4. [What We Aim to Achieve](#4-what-we-aim-to-achieve)
5. [Our Proposed Method — BatteryLoRA](#5-our-proposed-method)
6. [Experimental Plan](#6-experimental-plan)
7. [Baselines — What We Compare Against](#7-baselines)
8. [Metrics — How We Measure Success](#8-metrics)
9. [Expected Results and Claims](#9-expected-results)
10. [Tools and Setup](#10-tools-and-setup)
11. [Timeline — 5-Week Plan](#11-timeline)
12. [Paper Structure](#12-paper-structure)
13. [Key References](#13-key-references)

---

## 1. The Big Picture — What Are We Building?

### The One-Sentence Version

We are building a system where many phones collaboratively learn to improve a small AI language model, where each phone **automatically adjusts how much work it does based on its battery level** — phones with plenty of battery do heavy learning, phones with low battery do lightweight learning or sit out entirely.

### The Problem

Imagine you want 100 phones to collaboratively improve a language model (like a small version of ChatGPT). Each phone has private data — text messages, notes, browsing history — that we don't want to upload to a server. So instead, each phone:

1. Downloads the current model
2. Learns from its local data
3. Sends back what it learned (not the data itself!)
4. A central server combines everyone's learnings

This is called **Federated Learning** (FL).

The challenge: phones have batteries. If we make every phone do the same amount of work, phones with low battery will either:
- Die mid-training (wasting their contribution)
- Drop out of the process (reducing total participation)
- Drain users' batteries (making them uninstall the app)

**Our key insight**: We can make phones do *different amounts* of learning based on their battery. Specifically, we adjust a parameter called the **LoRA rank** — a knob that controls how much the phone modifies the model. High rank = more learning capacity but more compute. Low rank = less learning but saves battery.

**Nobody has done this before.** Battery-aware FL exists. Heterogeneous LoRA exists. But combining them — dynamically adjusting LoRA rank based on real-time battery state — is an unexplored intersection.

---

## 2. Background Concepts — Everything You Need to Know

### 2.1 What Is a Language Model?

A language model is a neural network that predicts the next word in a sequence. GPT-4, Claude, Llama — these are all language models. They have "parameters" (numbers that the model has learned), and larger models generally perform better:

| Model | Parameters | Size in Memory |
|-------|-----------|----------------|
| TinyLlama | 1.1 billion | ~2.2 GB |
| Llama 3 8B | 8 billion | ~16 GB |
| GPT-4 | ~1.8 trillion (estimated) | ~3.6 TB |

We will work with **TinyLlama (1.1B parameters)** — small enough to fine-tune on a laptop GPU.

### 2.2 What Is Fine-Tuning?

A pre-trained language model knows general language. Fine-tuning adapts it to a specific task or user's style. For example:
- Fine-tuning on medical text → better medical assistant
- Fine-tuning on your emails → better at writing in your voice

**Full fine-tuning** updates all 1.1 billion parameters. This is expensive — it requires storing gradients and optimizer states, typically 4-6x the model size in memory.

### 2.3 What Is LoRA (Low-Rank Adaptation)?

LoRA is a clever trick published in 2021 by Hu et al. Instead of updating all 1.1B parameters, LoRA:

1. **Freezes** the original model (no changes)
2. Adds tiny **adapter matrices** alongside certain layers
3. Only trains these adapters (typically 0.1-1% of parameters)

**How it works technically:**

In a neural network, a layer multiplies input by a weight matrix W (e.g., 4096 x 4096 = 16.7 million numbers).

LoRA replaces the update to W with two small matrices:
```
Instead of: W_new = W + deltaW        (deltaW is 4096 x 4096)
LoRA does:  W_new = W + (A × B)       (A is 4096 x r, B is r x 4096)
```

Here **r** is the **rank** — the key parameter we will manipulate.

**What rank means:**

| Rank (r) | Adapter Parameters per Layer | Capacity | Compute Cost |
|----------|------------------------------|----------|-------------|
| 2 | 2 × 4096 × 2 = 16,384 | Very low | Very cheap |
| 4 | 2 × 4096 × 4 = 32,768 | Low | Cheap |
| 8 | 2 × 4096 × 8 = 65,536 | Medium | Moderate |
| 16 | 2 × 4096 × 16 = 131,072 | Good | More expensive |
| 32 | 2 × 4096 × 32 = 262,144 | High | Most expensive |

**Key finding from the original LoRA paper**: Rank 4 achieves 73.7% accuracy on WikiSQL, while rank 64 gets only 73.6%. **Increasing rank beyond ~8 often doesn't help much**, but it always costs more compute and memory.

This diminishing-returns property is exactly what makes our approach work — low-battery phones can use rank 2-4 with minimal quality loss.

### 2.4 What Is Federated Learning?

Federated Learning (FL) trains a model across many devices without collecting their data. The canonical algorithm is **FedAvg** (McMahan et al., 2017):

```
For each round:
    1. Server sends current model to selected clients
    2. Each client trains on local data for E epochs
    3. Each client sends updated model back to server
    4. Server averages all client updates (weighted by data size)
    5. Repeat
```

**Key challenges in FL:**

- **Non-IID data**: Each phone has different data. One person texts about sports, another about cooking. This heterogeneity makes learning harder.
- **Stragglers**: Some devices are slow. The server must wait for the slowest device in each round.
- **Dropouts**: Devices disconnect, run out of battery, or are killed by the OS.
- **Communication cost**: Sending model updates over cellular networks is expensive.

### 2.5 What Is Federated LoRA?

Instead of federating the full model, we federate only the LoRA adapters. This massively reduces communication:

| What's sent | Size for TinyLlama |
|-------------|-------------------|
| Full model | ~2.2 GB |
| LoRA (rank 8) | ~4.2 MB |
| LoRA (rank 2) | ~1.1 MB |

So LoRA reduces communication by **500-2000x**. This is why federated LoRA is practical for mobile.

### 2.6 What Is Heterogeneous LoRA?

In standard federated LoRA, every device uses the same rank. But devices have different capabilities. **Heterogeneous LoRA** lets different devices use different ranks:

- Powerful phone → rank 32
- Mid-range phone → rank 8
- Weak phone → rank 4

**Key existing work:**

**HetLoRA (EMNLP 2024)**: Assigns ranks based on device compute capability. Uses zero-padding to aggregate different-rank adapters. Found that using the same rank for all devices (HomLoRA) can catastrophically overfit — rank-50 HomLoRA achieved perplexity 307 vs 32 for HetLoRA.

**FLoRA (NeurIPS 2024)**: Proved that existing methods for aggregating heterogeneous LoRA are mathematically incorrect (they introduce noise). Proposed "stacking" — treating each client's low-rank adapter as a sub-matrix of a larger global adapter. Tested on TinyLlama with ranks [64,32,16,16,8,8,4,4,4,4].

**HeLoRA (ACM 2025)**: Uses a greedy algorithm to assign ranks based on device resources.

**Critical limitation of all these works**: They assign ranks based on **static** device properties (CPU speed, RAM). They completely ignore **dynamic** factors like battery level, thermal state, or whether the device is charging. A powerful phone with 5% battery should not use rank 32.

### 2.7 What Is Battery-Aware Federated Learning?

Some papers have considered battery in FL, but none touch LoRA rank:

**BEFL (2024)**: Balances energy consumption across devices to prevent any single device from draining too fast. Achieved 28.2% less total energy and 72.7% less variance in energy across devices.

**EAFL (2022)**: Energy-adaptive FL that adjusts client participation based on energy budgets. Reduced dropout by 2.45x.

**FedBacys (2025)**: Groups devices by battery state and schedules training in cycles. Well-charged devices train first; low-battery devices train later (potentially after charging).

**LeanFed (2024)**: Adapts the fraction of local data used for training based on battery level.

**What none of them do**: Adjust the **model architecture** (i.e., LoRA rank) based on battery. They only adjust scheduling, participation, or data fraction.

---

## 3. Why This Is Novel — The Gap Nobody Has Filled

### The Two Existing Research Threads

```
Thread A: Heterogeneous LoRA in FL          Thread B: Battery-Aware FL
(HetLoRA, FLoRA, HeLoRA, FlexLoRA)         (BEFL, EAFL, FedBacys, LeanFed)

Assigns LoRA ranks based on                 Adjusts participation/scheduling
STATIC device capabilities                  based on DYNAMIC battery state
(CPU, RAM, fixed)                           (changes every minute)

Never considers battery.                    Never considers LoRA rank.
```

### Our Contribution: The Intersection

```
                    Our Work: BatteryLoRA

    Dynamically adjusts LoRA rank based on
    real-time battery state AND device capability

    Thread A (HetLoRA) ←——— COMBINED ———→ Thread B (BEFL)
```

**Nobody occupies this intersection.** We verified this through extensive literature search across 2024-2026 publications. This is a clean, defensible novelty claim.

### Why This Matters Practically

Consider a real-world deployment:
- Morning: Phone is at 95%, on charger → Use rank 32, full learning
- Commute: Phone at 60%, on cellular → Use rank 8, moderate learning
- Evening: Phone at 15%, user needs it → Use rank 2 or skip entirely

This is how humans naturally think about resource allocation, but no FL system does this today.

---

## 4. What We Aim to Achieve

### Primary Research Questions

1. **Does dynamically adapting LoRA rank to battery state improve the energy-accuracy tradeoff compared to fixed-rank approaches?**

2. **Can battery-aware rank selection reduce client dropout rates while maintaining model quality?**

3. **Is the approach fair — does it prevent low-battery devices from being systematically disadvantaged?**

### Concrete Claims We Want to Make

After our experiments, we want to show:

| Claim | Target |
|-------|--------|
| Same or better accuracy as fixed-rank baselines | Within 1-2% of best baseline |
| Lower total energy consumption | 15-30% reduction vs uniform high-rank |
| Lower energy variance across devices | 40-60% reduction (fairer) |
| Fewer client dropouts | 30-50% reduction |
| Faster convergence in wall-clock time | 10-20% faster (fewer stragglers) |

---

## 5. Our Proposed Method — BatteryLoRA

### 5.1 System Architecture

```
                    ┌─────────────────────┐
                    │   Central Server     │
                    │                     │
                    │  Global LoRA Model  │
                    │  (Rank R_max = 32)  │
                    │                     │
                    │  Aggregation Engine │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐  ┌──▼───────────┐  ┌▼──────────────┐
     │ Phone A        │  │ Phone B      │  │ Phone C       │
     │ Battery: 90%   │  │ Battery: 45% │  │ Battery: 12%  │
     │ Charging: Yes  │  │ Charging: No │  │ Charging: No  │
     │                │  │              │  │               │
     │ → Rank 32      │  │ → Rank 8     │  │ → Rank 2      │
     │ (Full learning)│  │ (Moderate)   │  │ (Minimal)     │
     └────────────────┘  └──────────────┘  └───────────────┘
```

### 5.2 Battery-to-Rank Mapping Function

The core of our method: a function that maps battery state to LoRA rank.

**Simple threshold-based policy:**

```python
def get_lora_rank(battery_percent, is_charging, device_tier):
    """
    Maps battery state to LoRA rank.

    Args:
        battery_percent: 0-100
        is_charging: True/False
        device_tier: "high" / "mid" / "low" (based on hardware)

    Returns:
        LoRA rank (2, 4, 8, 16, or 32)
    """
    # If charging, always use maximum rank for device tier
    if is_charging:
        return {"high": 32, "mid": 16, "low": 8}[device_tier]

    # Battery-based rank selection
    if battery_percent > 80:
        return {"high": 32, "mid": 16, "low": 8}[device_tier]
    elif battery_percent > 60:
        return {"high": 16, "mid": 8, "low": 4}[device_tier]
    elif battery_percent > 40:
        return {"high": 8, "mid": 4, "low": 4}[device_tier]
    elif battery_percent > 20:
        return {"high": 4, "mid": 2, "low": 2}[device_tier]
    else:
        # Below 20%: minimal participation or skip
        return 2  # or return 0 to skip this round
```

**Continuous policy (more sophisticated — stretch goal):**

```python
def get_lora_rank_continuous(battery_percent, is_charging, power_draw_watts):
    """
    Continuous mapping using energy budget estimation.

    Estimates remaining training rounds possible given battery,
    then allocates rank to maximize total contribution.
    """
    battery_capacity_wh = 17.1  # Typical 4500mAh × 3.8V
    remaining_wh = battery_capacity_wh * (battery_percent / 100)
    reserve_wh = battery_capacity_wh * 0.15  # Keep 15% for user
    available_wh = max(0, remaining_wh - reserve_wh)

    # Energy cost per round by rank (measured empirically)
    energy_per_round = {2: 0.8, 4: 1.2, 8: 2.0, 16: 3.5, 32: 6.0}  # Wh

    # Pick highest rank that allows at least N future rounds
    min_future_rounds = 5
    for rank in [32, 16, 8, 4, 2]:
        if available_wh / energy_per_round[rank] >= min_future_rounds:
            return rank
    return 0  # Skip this round
```

### 5.3 Aggregation with Heterogeneous Ranks

When clients return LoRA adapters of different ranks, we need to aggregate them. We use the **FLoRA stacking method** (NeurIPS 2024), which is mathematically correct:

```
Client 1 returns: A1 (4096 × 32), B1 (32 × 4096)  — rank 32
Client 2 returns: A2 (4096 × 8),  B2 (8 × 4096)   — rank 8
Client 3 returns: A3 (4096 × 2),  B3 (2 × 4096)   — rank 2

FLoRA stacking:
- Each client's adapter is treated as updating a SUBSET of the global adapter
- Client 2's rank-8 adapter updates the first 8 rows/columns of the global rank-32 space
- Client 3's rank-2 adapter updates the first 2 rows/columns
- Non-overlapping dimensions are averaged from clients that did update them
```

### 5.4 Complete Algorithm

```
Algorithm: BatteryLoRA

Input:
  - Pre-trained model M (TinyLlama 1.1B, frozen)
  - N clients, each with local dataset D_i
  - Maximum rank R_max = 32
  - Number of rounds T
  - Battery simulator for each client

1. Initialize global LoRA adapters (A_global, B_global) at rank R_max

2. For each round t = 1, ..., T:
   a. Server selects C clients (e.g., 10 out of 50)

   b. For each selected client i:
      - Query battery state: (battery_pct, is_charging, device_tier)
      - Compute rank: r_i = get_lora_rank(battery_pct, is_charging, device_tier)
      - If r_i == 0: skip this client
      - Extract sub-adapter of rank r_i from global adapter
      - Send sub-adapter to client i (~r_i/R_max of full adapter size)

   c. Each client i (in parallel):
      - Receives sub-adapter of rank r_i
      - Trains for E local epochs on D_i
      - Sends updated sub-adapter back to server
      - Update battery simulator: drain based on rank and training time

   d. Server aggregates:
      - Use FLoRA stacking to combine adapters of different ranks
      - Weight each client's contribution by |D_i| (dataset size)
      - Update global adapter

3. Return final model M + (A_global × B_global)
```

---

## 6. Experimental Plan

### 6.1 Overview of Experiments

We need 5 groups of experiments:

| Experiment | Purpose | Priority |
|------------|---------|----------|
| E1: Main comparison | Show BatteryLoRA beats baselines | Critical |
| E2: Ablation on rank policies | Justify our battery-to-rank mapping | Critical |
| E3: Non-IID sensitivity | Show robustness to data heterogeneity | Important |
| E4: Scale sensitivity | Show it works with 10, 20, 50 clients | Important |
| E5: Battery profile analysis | Analyze energy fairness in detail | Nice to have |

### 6.2 Experiment E1: Main Comparison

**Setup:**
- Model: TinyLlama 1.1B (frozen) + LoRA adapters
- Clients: 50 simulated
- Data: Alpaca-GPT4 dataset (52K instruction-following examples)
- Data split: Non-IID using Dirichlet distribution (alpha = 0.5)
- Rounds: 100 federated rounds
- Local epochs: 3 per round
- Client selection: 10 per round (20% participation)

**Device tiers (to simulate heterogeneous phones):**

| Tier | Proportion | Represents | Base capability |
|------|-----------|------------|-----------------|
| High | 20% (10 devices) | Flagship phones (iPhone 15 Pro, Pixel 8 Pro) | Up to rank 32 |
| Mid | 50% (25 devices) | Mid-range phones (Pixel 7a, Galaxy A54) | Up to rank 16 |
| Low | 30% (15 devices) | Budget phones (older devices, 3-4 GB RAM) | Up to rank 8 |

**Battery simulation:**
- Each device starts with a random battery level (uniform 20-100%)
- 30% of devices start "charging" (these stay at high battery)
- Battery drains proportional to rank used (measured per round)
- Devices that hit 10% battery drop out for remaining rounds
- Battery traces optionally drawn from real GreenHub dataset

**What we compare (see Section 7 for baseline details):**
- Our method: BatteryLoRA
- Baseline 1: FedAvg + HomLoRA (all devices use rank 8)
- Baseline 2: HetLoRA (rank based on static device tier only)
- Baseline 3: FLoRA (heterogeneous, static assignment)
- Baseline 4: EAFL + LoRA (energy-aware participation, fixed rank)
- Baseline 5: FedAvg + HomLoRA rank 32 (upper bound on quality)
- Baseline 6: Local-only LoRA (no federation, lower bound)

**What we measure:**
- Task accuracy (MMLU / instruction-following quality)
- Total energy consumed across all devices
- Number of client dropouts (battery deaths)
- Convergence speed (rounds to reach X% accuracy)
- Energy variance across devices (Jain's fairness index)

### 6.3 Experiment E2: Ablation Study on Rank Policies

Test different battery-to-rank mapping strategies:

| Policy | Description |
|--------|-------------|
| Threshold (ours) | 5-tier battery thresholds as described above |
| Continuous | Energy-budget-based rank selection |
| Binary | Only rank 32 (battery > 50%) or rank 2 (battery <= 50%) |
| Random | Random rank assignment (controls for rank diversity effect) |
| Oracle | Optimal rank assignment with perfect future knowledge (upper bound) |

This answers: **Does the specific battery-to-rank mapping matter, or is any dynamic adaptation enough?**

### 6.4 Experiment E3: Non-IID Sensitivity

Repeat E1 with different levels of data heterogeneity:

| Dirichlet alpha | Meaning |
|-----------------|---------|
| 0.1 | Highly non-IID (each device has mostly 1-2 topics) |
| 0.5 | Moderately non-IID (our default) |
| 1.0 | Mildly non-IID |
| 100.0 | Nearly IID (almost uniform distribution) |

This answers: **Does BatteryLoRA's advantage hold when data is very heterogeneous?**

### 6.5 Experiment E4: Scale Sensitivity

Test with different numbers of clients:

| Clients | Selected per round | Scenario |
|---------|-------------------|----------|
| 10 | 5 | Small deployment (lab setting) |
| 20 | 8 | Medium deployment |
| 50 | 10 | Our default |
| 100 | 15 | Large deployment (stretch goal) |

### 6.6 Experiment E5: Energy Fairness Deep Dive

For our best configuration:
- Plot each device's cumulative energy consumption over time
- Plot each device's battery trajectory over rounds
- Show that BatteryLoRA creates more uniform battery drain
- Compute Gini coefficient of energy consumption
- Compare against baselines on fairness metrics

---

## 7. Baselines — What We Compare Against

### 7.1 Detailed Baseline Descriptions

**Baseline 1: FedAvg + HomLoRA (rank 8)**
- Standard federated averaging with LoRA
- All devices use rank 8 regardless of battery
- Represents the "default" approach most people would use
- Source: Combination of McMahan et al. 2017 + Hu et al. 2021

**Baseline 2: HetLoRA**
- Assigns LoRA rank based on static device tier (high→32, mid→16, low→8)
- Never changes rank based on battery
- Uses zero-padding for aggregation
- Source: Cho et al., EMNLP 2024

**Baseline 3: FLoRA**
- Also uses heterogeneous ranks based on device tier
- Uses mathematically correct stacking aggregation
- Static rank assignment
- Source: Wang et al., NeurIPS 2024

**Baseline 4: EAFL + LoRA**
- Energy-adaptive federated learning
- Adjusts WHETHER a device participates based on energy budget
- But when it participates, uses a fixed rank
- Source: Adapted from Xu et al., 2022

**Baseline 5: FedAvg + HomLoRA (rank 32)**
- All devices use maximum rank 32
- Ignores battery and device constraints entirely
- Upper bound on model quality, but lots of dropouts expected
- Shows what happens when you ignore battery

**Baseline 6: Local-only LoRA**
- Each device trains LoRA independently on its own data
- No federation, no communication
- Lower bound — shows the value of collaboration

### 7.2 Why These Baselines Are Sufficient

| Baseline | Controls for |
|----------|-------------|
| HomLoRA r=8 | "Just use a moderate fixed rank" |
| HetLoRA | "Adapt rank to device, ignore battery" |
| FLoRA | "Best existing heterogeneous method" |
| EAFL + LoRA | "Handle battery via participation, not rank" |
| HomLoRA r=32 | "Maximize quality, ignore constraints" |
| Local-only | "Is federation even worth it?" |

---

## 8. Metrics — How We Measure Success

### 8.1 Model Quality Metrics

**For instruction-following (Alpaca):**
- ROUGE-L score on held-out test set
- MMLU accuracy (5-shot, on a subset of topics)
- GPT-4 judge score (optional — uses GPT-4 to rate response quality)

**For text classification (if we add GLUE tasks):**
- Accuracy on SST-2 (sentiment)
- Accuracy on QNLI (question answering)

### 8.2 Efficiency Metrics

**Energy:**
- Total energy consumed (Joules) across all devices across all rounds
- Per-device energy consumption distribution
- Energy-per-accuracy-point (Joules to reach X% accuracy)

**Communication:**
- Total bytes transmitted (upload + download)
- Bytes per accuracy point

**Time:**
- Wall-clock convergence time (simulated, accounting for stragglers)
- Rounds to reach target accuracy

### 8.3 Robustness Metrics

**Dropout rate:**
- Percentage of devices that drop out due to battery exhaustion
- Average round at which dropout occurs

**Fairness:**
- Jain's Fairness Index on energy consumption: J = (sum(x_i))^2 / (n * sum(x_i^2))
  - J = 1.0 means perfectly fair (all devices consume equal energy)
  - J = 1/n means maximally unfair (one device does all the work)
- Gini coefficient of per-device accuracy (do all devices benefit equally?)

### 8.4 Summary Table of All Metrics

| Category | Metric | Direction | Target |
|----------|--------|-----------|--------|
| Quality | ROUGE-L | Higher is better | Within 1-2% of best baseline |
| Quality | MMLU accuracy | Higher is better | Within 1-2% of best baseline |
| Energy | Total energy (J) | Lower is better | 15-30% less than HomLoRA r=8 |
| Energy | Energy variance | Lower is better | 40-60% less than HomLoRA |
| Fairness | Jain's index (energy) | Higher is better (max 1.0) | > 0.85 |
| Robustness | Dropout rate | Lower is better | 30-50% fewer than HomLoRA r=32 |
| Communication | Total bytes | Lower is better | Show reduction |
| Speed | Rounds to target | Lower is better | 10-20% fewer |

---

## 9. Expected Results and Claims

### What We Expect to See (and Why)

**Claim 1: BatteryLoRA matches quality of static heterogeneous methods.**

Why: LoRA shows diminishing returns beyond rank 4-8 (original paper). So devices using rank 2-4 when battery is low still contribute meaningful updates. The FLoRA stacking aggregation correctly combines different-rank contributions.

**Claim 2: BatteryLoRA significantly reduces total energy consumption.**

Why: Low-battery devices use low ranks (cheaper compute). High-battery or charging devices use high ranks (they can afford it). The total energy is lower because we avoid wasting energy on devices that will drop out anyway.

**Claim 3: BatteryLoRA dramatically reduces dropout rates.**

Why: Devices proactively reduce their rank as battery decreases. They never hit 0% from overwork. In baselines, devices using fixed high rank burn through battery and drop out.

**Claim 4: BatteryLoRA is fairer in energy distribution.**

Why: The adaptive policy automatically balances energy drain — devices that have used more energy (lower battery) are assigned lower ranks, creating a natural equalizing effect.

**Claim 5: BatteryLoRA reduces communication cost.**

Why: Low-rank adapters are smaller. A rank-2 adapter is 16x smaller than rank-32. When many devices use low ranks, total communication decreases.

---

## 10. Tools and Setup

### 10.1 Software Stack

| Tool | Purpose | Install |
|------|---------|---------|
| **Python 3.10+** | Programming language | python.org |
| **PyTorch 2.0+** | Deep learning framework | `pip install torch` |
| **Flower (flwr)** | Federated learning framework | `pip install flwr` |
| **HuggingFace Transformers** | Model loading and tokenization | `pip install transformers` |
| **PEFT** | LoRA implementation | `pip install peft` |
| **Datasets** | Loading Alpaca, GLUE, etc. | `pip install datasets` |
| **Weights & Biases** | Experiment tracking (optional) | `pip install wandb` |
| **matplotlib / seaborn** | Plotting results | `pip install matplotlib seaborn` |

### 10.2 Hardware Requirements

**Minimum (for development and small experiments):**
- 1 GPU with 16+ GB VRAM (e.g., NVIDIA T4, RTX 3080)
- Google Colab Pro ($10/month) provides this
- Or a university GPU cluster

**Recommended (for full experiments):**
- 1 GPU with 24+ GB VRAM (e.g., RTX 3090, A5000)
- Or 2x T4 GPUs (e.g., via Kaggle or Google Colab)

**Why not 50 real phones:**
- Every battery-aware FL paper uses simulation — this is accepted practice
- BEFL, EAFL, FedBacys all simulated devices
- Flower's Virtual Client Engine simulates 50+ clients on a single GPU
- Each "client" gets a time slice on the GPU, trains for a few steps, then yields

### 10.3 Key Code Components to Build

```
project/
├── config.py                  # Hyperparameters and experiment configs
├── battery_simulator.py       # Battery drain model per device tier
├── rank_policy.py             # Battery-to-rank mapping functions
├── client.py                  # Flower client with dynamic LoRA rank
├── strategy.py                # Custom Flower strategy for aggregation
├── data_utils.py              # Dataset loading and non-IID partitioning
├── model_utils.py             # TinyLlama + PEFT/LoRA setup
├── run_experiment.py          # Main experiment runner
├── evaluate.py                # Evaluation metrics computation
├── plot_results.py            # Generate figures for paper
└── experiments/
    ├── e1_main_comparison.py
    ├── e2_ablation_policies.py
    ├── e3_noniid_sensitivity.py
    ├── e4_scale_sensitivity.py
    └── e5_energy_fairness.py
```

### 10.4 Starting Point: Mozilla AI Blueprint

Mozilla AI has an open-source federated LoRA fine-tuning codebase:
- Repository: `github.com/mozilla-ai/federated-finetuning/`
- Uses: Flower + PEFT + Qwen2-0.5B + Alpaca-GPT4
- We can fork this and add:
  1. Battery simulation layer
  2. Dynamic rank selection
  3. FLoRA stacking aggregation
  4. Our metrics/evaluation

This saves us ~1 week of setup time.

---

## 11. Timeline — 5-Week Plan

### Week 1: Setup and Baseline (Days 1-7)

| Day | Task | Owner |
|-----|------|-------|
| 1-2 | Set up development environment, install all dependencies, get GPU access | Everyone |
| 2-3 | Fork Mozilla blueprint, run it end-to-end with default settings | Person A |
| 2-3 | Implement battery simulator (BatterySimulator class) | Person B |
| 3-4 | Implement data partitioning (Dirichlet non-IID splits) | Person C |
| 5-6 | Run Baseline 1 (FedAvg + HomLoRA rank 8) and Baseline 6 (local-only) | Person A |
| 6-7 | Implement evaluation pipeline (ROUGE-L, MMLU subset) | Person C |

**Deliverable:** Working FL pipeline with LoRA, one baseline result, battery simulator ready.

### Week 2: Core Method and Baselines (Days 8-14)

| Day | Task | Owner |
|-----|------|-------|
| 8-9 | Implement rank policy module (threshold + continuous) | Person B |
| 9-10 | Implement BatteryLoRA client (dynamic rank per round) | Person A |
| 10-11 | Implement FLoRA stacking aggregation in custom Flower strategy | Person A + B |
| 11-12 | Run Baselines 2-5 (HetLoRA, FLoRA, EAFL, HomLoRA r=32) | Person C |
| 13-14 | Run BatteryLoRA (our method) first pass | Everyone |

**Deliverable:** All baselines and our method running. First results visible.

### Week 3: Full Experiments (Days 15-21)

| Day | Task | Owner |
|-----|------|-------|
| 15-16 | Experiment E1: Main comparison (full run, 100 rounds, 3 seeds) | Person A |
| 16-17 | Experiment E2: Ablation on rank policies | Person B |
| 17-18 | Experiment E3: Non-IID sensitivity (alpha = 0.1, 0.5, 1.0, 100) | Person C |
| 19-20 | Experiment E4: Scale sensitivity (10, 20, 50 clients) | Person A |
| 20-21 | Experiment E5: Energy fairness deep dive | Person B |

**Deliverable:** All experimental data collected.

### Week 4: Analysis and Writing (Days 22-28)

| Day | Task | Owner |
|-----|------|-------|
| 22-23 | Generate all plots and tables | Person C |
| 23-24 | Write Introduction + Related Work | Person A |
| 24-25 | Write Methodology section | Person B |
| 25-26 | Write Experiments + Results section | Person C |
| 27-28 | Write Abstract + Conclusion, internal review | Everyone |

**Deliverable:** Complete paper draft.

### Week 5: Polish and Submit (Days 29-35)

| Day | Task | Owner |
|-----|------|-------|
| 29-30 | Re-run any experiments that need fixing | Person A |
| 30-31 | Revise paper based on internal review | Person B |
| 32-33 | Final formatting (Springer CCIS template) | Person C |
| 34 | Final proofread, check references | Everyone |
| 35 | Submit on EasyChair | Everyone |

---

## 12. Paper Structure

### Target: Short Paper (8 pages) or Long Paper (16 pages)

For a first submission, I recommend **short paper (8 pages)** — it's more realistic for the timeline and allows a focused contribution.

### Proposed Outline

```
1. Introduction (1 page)
   - FL on mobile devices is growing
   - LoRA makes federated fine-tuning practical
   - Problem: current methods ignore dynamic device constraints (battery)
   - Our contribution: BatteryLoRA

2. Related Work (1 page)
   - Federated LoRA (HetLoRA, FLoRA, FlexLoRA)
   - Battery/Energy-aware FL (BEFL, EAFL, FedBacys)
   - Gap: no work at the intersection

3. Method: BatteryLoRA (2 pages)
   - System architecture
   - Battery-to-rank mapping policy
   - FLoRA-based aggregation with heterogeneous ranks
   - Algorithm pseudocode

4. Experimental Setup (1 page)
   - Model, dataset, baselines, metrics
   - Battery simulation details
   - Device tier configuration

5. Results and Analysis (2 pages)
   - E1: Main comparison table + convergence curves
   - E2: Ablation on policies
   - E3: Non-IID sensitivity
   - Energy fairness analysis (per-device plots)

6. Conclusion (0.5 pages)
   - Summary of contributions
   - Limitations (simulation-only, single model)
   - Future work (real device deployment, larger models)

References (~0.5 pages)
```

---

## 13. Key References

### Must-Read Papers (in priority order)

1. **LoRA** — Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022. [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

2. **FLoRA** — Wang et al., "FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations," NeurIPS 2024. [arxiv.org/abs/2409.05976](https://arxiv.org/abs/2409.05976)

3. **HetLoRA** — Cho et al., "Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models," EMNLP 2024. [arxiv.org/abs/2401.06432](https://arxiv.org/abs/2401.06432)

4. **BEFL** — "Balancing Energy Consumption in Federated Learning for Heterogeneous Devices," 2024. [arxiv.org/abs/2412.03950](https://arxiv.org/abs/2412.03950)

5. **FedBacys** — "Battery-aware Cyclic Scheduling in Federated Learning," 2025. [arxiv.org/abs/2504.12181](https://arxiv.org/abs/2504.12181)

6. **EAFL** — Xu et al., "Energy-Adaptive Federated Learning," 2022. [arxiv.org/abs/2208.04505](https://arxiv.org/abs/2208.04505)

7. **FedAvg** — McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017. [arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629)

8. **HeLoRA** — "LoRA-Heterogeneous Federated Fine-tuning of Large Language Models," ACM TOIT 2025. [dl.acm.org/doi/10.1145/3723877](https://dl.acm.org/doi/10.1145/3723877)

9. **FlexLoRA** — "FlexLoRA: When Clients Possess Varying LoRA Ranks," NeurIPS 2024. [openreview.net/forum?id=gkOzoHBXUw](https://openreview.net/forum?id=gkOzoHBXUw)

10. **LeanFed** — "LeanFed: Adaptive Federated Learning on Non-IID Data with Limited Communication Budget," 2024. [arxiv.org/abs/2412.02289](https://arxiv.org/abs/2412.02289)

### Supplementary References

11. **TinyLlama** — Zhang et al., "TinyLlama: An Open-Source Small Language Model," 2024.
12. **Flower Framework** — Beutel et al., "Flower: A Friendly Federated Learning Framework," 2020.
13. **PEFT Library** — HuggingFace, [github.com/huggingface/peft](https://github.com/huggingface/peft)
14. **GreenHub Battery Dataset** — Cruz et al., "GreenHub: A large-scale collaborative dataset," Empirical Software Engineering, 2020.
15. **Mozilla Federated Fine-tuning Blueprint** — [blueprints.mozilla.ai](https://blueprints.mozilla.ai/all-blueprints/finetune-an-llm-using-federated-learning)
16. **Dirichlet Partitioning for Non-IID FL** — Hsu et al., "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification," NeurIPS FL Workshop, 2019.

---

## Quick Reference Card

| Item | Our Choice | Why |
|------|-----------|-----|
| Model | TinyLlama 1.1B | Small, open-source, used in FLoRA (NeurIPS 2024) |
| LoRA ranks | {2, 4, 8, 16, 32} | 5 tiers matching 5 battery levels |
| FL framework | Flower | Best simulation support, has DirichletPartitioner |
| Dataset | Alpaca-GPT4 | Instruction-following, matches FLoRA baseline |
| Clients | 50 simulated | Standard in FL literature |
| Aggregation | FLoRA stacking | Mathematically correct for heterogeneous ranks |
| Battery model | Software simulation | Standard practice (BEFL, EAFL all simulate) |
| Paper target | Short paper (8 pages) | Realistic for 5-week timeline |
| Conference | NGEN-AI 2026 | Deadline May 25, 2026 |
