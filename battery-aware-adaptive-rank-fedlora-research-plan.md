# Research Plan: Battery-Aware Adaptive-Rank Federated LoRA for On-Device SLM Personalization

## Comprehensive Literature Review and Technical Findings

---

## 1. LoRA (Low-Rank Adaptation) -- Technical Deep Dive

### How LoRA Works
LoRA freezes the pretrained weight matrix W and injects a trainable low-rank decomposition: W' = W + BA, where B is a d x r matrix and A is an r x k matrix. The rank r << min(d,k) is the key hyperparameter. Only B and A are trained; the original W remains frozen. This reduces trainable parameters from d*k to r*(d+k).

For a Transformer layer with hidden dimension d=4096 and rank r=8, each LoRA adapter has 2 * 4096 * 8 = 65,536 parameters vs. 16,777,216 for the full weight matrix -- a 256x reduction.

### Rank Effects -- Specific Numbers from Papers

**Original LoRA Paper (Hu et al., 2021) -- GPT-3 175B on WikiSQL:**

| Rank (r) | Trainable Params | WikiSQL Accuracy |
|----------|-----------------|------------------|
| 1        | 4.7M            | 73.4%            |
| 2        | 9.4M            | 73.3%            |
| 4        | 18.8M           | 73.7%            |
| 8        | 37.7M           | 73.8%            |
| 64       | 301.9M          | 73.6%            |

Key insight: Rank 4 achieves near-optimal performance. Rank 64 actually performs WORSE than rank 8 despite 8x more parameters. This validates the "low intrinsic dimensionality" hypothesis.

**RoBERTa-base on GLUE:**
- LoRA with 0.3M parameters: 87.2% average accuracy
- Full fine-tuning with 125M parameters: 86.4% average accuracy
- LoRA outperforms full FT with 400x fewer parameters

**Practical Rank Guidelines:**
- Rank 2-4: Sufficient for narrow tasks (single dataset, limited domain)
- Rank 8-16: Good general-purpose setting for most NLP tasks
- Rank 32-64: Useful for complex multi-task or instruction-following
- Rank >64: Diminishing returns, risk of overfitting

### Compute Cost Scaling
- Memory: Scales linearly with rank. Rank 16 uses ~2x memory of rank 8 for LoRA parameters
- Training time: Approximately linear with rank for forward/backward passes through adapters
- Communication in FL: LoRA adapter size = 2 * r * d * num_layers * sizeof(float16). For TinyLlama (d=2048, 22 layers): rank 4 = ~720KB, rank 16 = ~2.9MB, rank 64 = ~11.5MB per adapter

**References:**
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022 (arxiv.org/abs/2106.09685)
- Microsoft LoRA implementation: github.com/microsoft/LoRA

---

## 2. FLOWER Framework -- APIs, Simulation, and Custom Strategies

### Overview
Flower (flwr) is the leading open-source federated learning framework. It supports simulation (no real devices needed) and production deployment. Key features:
- Virtual Client Engine for large-scale simulation (1000+ clients)
- Built-in strategies: FedAvg, FedProx, FedAdagrad, FedAdam, FedYogi
- HuggingFace PEFT integration for LoRA fine-tuning
- PyTorch, TensorFlow, JAX backends

### Key APIs for Custom Strategy

**Strategy Methods to Override:**
1. `configure_train(server_round, arrays, config, grid)` -- Prepare messages sent to clients (model weights + config like learning rate)
2. `aggregate_train(replies)` -- Aggregate client model updates after training
3. `configure_evaluate(server_round, arrays, config, grid)` -- Configure evaluation rounds
4. `aggregate_evaluate(replies)` -- Aggregate evaluation metrics
5. `start()` -- Main entry point orchestrating the FL loop

**Simulation API:**
```python
from flwr.simulation import run_simulation
run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=50,  # number of simulated clients
    backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.1}}
)
```

**Custom Strategy Pattern (for adaptive rank):**
```python
class FedAdaptiveLoRA(FedAvg):
    def configure_train(self, server_round, arrays, config, grid):
        # Assign different LoRA ranks based on client battery state
        for client_id, client_config in client_configs.items():
            battery_level = get_battery_level(client_id)
            if battery_level > 0.7:
                client_config["lora_rank"] = 16
            elif battery_level > 0.3:
                client_config["lora_rank"] = 8
            else:
                client_config["lora_rank"] = 4
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, replies):
        # Custom aggregation handling heterogeneous ranks
        # Use SVD-based or stacking-based aggregation
        ...
```

### Existing Federated LoRA Example
Mozilla AI Blueprint: Federated LLM fine-tuning with Flower + PEFT
- Model: Qwen2-0.5B-Instruct
- Dataset: Alpaca-GPT4
- Code: github.com/mozilla-ai/federated-finetuning/
- Requirements: Linux, Python 3.10+, 8GB RAM minimum
- Supports CPU and GPU fine-tuning
- Uses Flower Simulation Engine

### Flower + HuggingFace PEFT Integration
The standard pattern:
1. Load base model with AutoModelForCausalLM
2. Create LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
3. Wrap with get_peft_model()
4. In Flower client: extract LoRA weights only for communication
5. Server aggregates only LoRA parameters (not full model)
6. Bandwidth reduction: 20x-100x vs full model exchange

**References:**
- Flower docs: flower.ai/docs/framework/
- Custom strategy tutorial: flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html
- Flower + HuggingFace tutorial: huggingface.co/blog/fl-with-flower
- Mozilla Blueprint: blueprints.mozilla.ai/all-blueprints/finetune-an-llm-using-federated-learning
- Flower GitHub: github.com/adap/flower

---

## 3. HetLoRA and HeLoRA Papers -- Detailed Analysis

### HetLoRA (Cho et al., EMNLP 2024 / Google Research)

**Core Problem:** In federated fine-tuning, devices have different compute/memory budgets. Forcing uniform LoRA rank wastes resources on powerful devices and overloads weak ones.

**Technical Approach:**
1. Each client trains a LoRA adapter with a rank suited to its resources (r_min=5 to r_max=50)
2. **Rank self-pruning:** Clients train full-rank LoRA locally, then prune to their capacity
3. **Sparsity-weighted aggregation:** Server weights contributions by rank -- higher-rank clients have more influence

**Experimental Setup:**
- Models: PaLM 2-XXS and PaLM 2-XS (Google's on-device foundation models)
- Datasets: Multi-Session Chat (MSC, perplexity metric), Reddit summarization (RougeL metric)
- Ranks tested: r in {1, 5, 10, 20, 50, 100, 150, 200}
- Clients per round: 5 (MSC), 10 (Reddit)
- Heterogeneous config: r_min=5, r_max=50
- Training: batch=8, local iterations=5, seq length=1024

**Key Results:**

| Method              | Reddit RougeL (XXS) | Chat Perplexity (XXS) |
|---------------------|---------------------|-----------------------|
| Full Fine-tuning    | 94.56               | 32.70                 |
| HomLoRA r=5         | 92.57               | 80.51                 |
| HomLoRA r=50        | 70.57               | 307.96 (overfitting!) |
| Recon+SVD           | 63.28               | 323.89                |
| **HetLoRA (g=0.99)**| **94.23**           | **53.93**             |

Critical finding: Homogeneous high-rank (r=50) CATASTROPHICALLY overfits (perplexity 307 vs 32 for full FT). HetLoRA avoids this by combining high and low rank benefits.

**Limitations (stated by authors):**
- Assumes rank distribution across clients is independent of data distribution
- In reality, resource-rich devices (affluent areas) may have systematically different data patterns
- Only tested on Google's PaLM models, not open-source LLMs

### HeLoRA (ACM TOIT, 2025)

**Two Variants:**
1. **HeLoRA-Pad:** Uses context-based padding to align heterogeneous LoRA matrices for aggregation. Rank-based adaptive aggregation gives higher-rank clients more influence.
2. **HeLoRA-KD:** Uses knowledge distillation / deep mutual learning for heterogeneous aggregation -- entirely different approach that doesn't require matrix alignment.

### FLoRA (NeurIPS 2024, Sony Research)

**Key Innovation:** Mathematically proves that existing LoRA aggregation methods (simple averaging of A and B matrices) introduce aggregation noise. Proposes stacking-based aggregation that is noise-free.

**Experimental Setup:**
- Models: TinyLlama 1.1B, LLaMA 7B, Llama2 7B
- Datasets: Databricks-dolly-15k, Alpaca, Wizard, ShareGPT
- Metrics: MMLU (QA), MT-bench (chat)
- Homogeneous rank: 16
- Heterogeneous ranks: [64, 32, 16, 16, 8, 8, 4, 4, 4, 4] across 10 clients
- Learning rate: 0.0003, batch size: 128

**Key Results:**

| Setting       | Method   | TinyLlama MMLU | Llama MMLU |
|---------------|----------|----------------|------------|
| Homogeneous   | FLoRA    | 30.80          | 29.85      |
| Homogeneous   | FedIT    | 16.35          | 29.41      |
| Heterogeneous | FLoRA    | 18.45          | 29.54      |
| Heterogeneous | Zero-pad | 15.76          | 7.97       |

Zero-padding completely collapses on heterogeneous ranks (7.97 on Llama). FLoRA's stacking method is robust.

**References:**
- HetLoRA: arxiv.org/abs/2401.06432 (EMNLP 2024)
- HeLoRA: dl.acm.org/doi/10.1145/3723877 (ACM TOIT 2025)
- FLoRA: arxiv.org/abs/2409.05976 (NeurIPS 2024)
- Rank collapse prevention: arxiv.org/html/2602.13486

---

## 4. Battery-Aware Federated Learning Papers

### BEFL (Balancing Energy in FL, Dec 2024)

**What it optimizes:** Three objectives jointly:
1. Global model accuracy
2. Total energy consumption across all devices
3. Energy usage DISPARITY between devices (fairness)

**How it works:**
- Sequential Least Squares Programming (SLSQP) for communication resource allocation
- Heuristic client selection via cluster partitioning + utility-driven approach
- Offline imitation learning during pre-training
- Online ranking-based reinforcement learning for adaptive selection

**Results:**
- +1.6% global model accuracy
- -72.7% energy consumption variance (fairness improvement)
- -28.2% total energy consumption

**Reference:** arxiv.org/abs/2412.03950

### FedBacys (Battery-aware Cyclic Scheduling, Apr 2025)

**What it optimizes:** Energy efficiency through smart scheduling of client participation.

**How it works:**
1. Cluster clients by remaining battery levels
2. Schedule participation sequentially within clusters (cyclic)
3. Clients train just before their designated transmission time (no idle computation)
4. FedBacys-Odd variant: selective participation for even more savings

**Key properties:**
- First comprehensive evaluation of cyclic participation in energy-harvesting FL
- Unified scheduling incorporating both communication and computation costs
- Robust under non-IID data and infrequent charging

**Reference:** arxiv.org/abs/2504.12181

### EAFL (Energy-Aware FL, 2022)

**What it optimizes:** Joint minimization of time-to-accuracy + maximization of remaining battery levels.

**How it works:**
- Cherry-picks clients with higher battery levels for training
- Power-aware training algorithm for client selection

**Results:**
- Up to 85% improvement in testing model accuracy (vs naive selection)
- 2.45x reduction in client dropout

**Reference:** arxiv.org/abs/2208.04505

### LeanFed (Dec 2024)

**What it optimizes:** Device participation longevity on battery-constrained devices.

**How it works:**
- Dynamically adjusts the FRACTION of local data each device uses per round
- Devices with low battery train on less data (fewer local steps)
- Maximizes total device participation across all rounds

**Evaluation:** CIFAR-10 and CIFAR-100, various non-IID levels

**Reference:** arxiv.org/abs/2412.02289

### EnFed (Energy-aware Opportunistic FL, Dec 2024)

**What it optimizes:** Human Activity Recognition on resource-constrained devices with energy awareness.

**Reference:** arxiv.org/html/2412.00768

### Gap in Literature (YOUR OPPORTUNITY)
None of these papers combine battery-awareness with ADAPTIVE LoRA RANK selection. They adjust:
- Client selection (who participates)
- Data fraction (how much data to train on)
- Scheduling (when to participate)

But NOT the model complexity (LoRA rank) per client per round. This is the novel contribution of your paper.

---

## 5. Model Selection for Experiments

### Comparison of Candidate Models

| Property         | TinyLlama 1.1B    | Phi-2 2.7B        | Gemma-2B           |
|------------------|--------------------|--------------------|---------------------|
| Parameters       | 1.1B               | 2.7B               | 2.0B                |
| Hidden dim       | 2,048              | 2,560              | 2,048               |
| Layers           | 22                 | 32                 | 18                   |
| Attention        | GQA (4 KV heads)   | Full MHA (32 heads)| MQA (1 KV head)     |
| FP16 Memory      | ~2.2 GB            | ~5.4 GB            | ~4.0 GB              |
| Tokens/sec       | ~85                | ~58                | ~65                  |
| MMLU             | 25.3%              | 56.7%              | 42.3%                |
| HumanEval        | 6.5%               | 47.0%              | 10.8%                |
| License          | Apache 2.0         | MIT                | Terms of Use         |
| Architecture     | LLaMA-based        | Custom             | Custom               |

### Memory Requirements for LoRA Fine-Tuning

**TinyLlama 1.1B:**
- Full FT (FP16 + Adam): ~7.75 GB VRAM
- Full FT (INT4 + Adam): ~1.94 GB VRAM
- LoRA (FP16, r=16): ~3-4 GB VRAM estimated
- QLoRA (4-bit, r=16): ~1.5-2 GB VRAM estimated

**General Rule:** ~2 GB per 1B parameters for model weights (FP16) + ~1-3 GB overhead for LoRA, optimizer, activations depending on batch size

**For a 7B model reference:**
- Full FT: ~67 GB
- LoRA (16-bit): ~15 GB
- QLoRA (8-bit): ~9 GB
- QLoRA (4-bit): ~5 GB

### Recommendation: TinyLlama 1.1B as Primary Model

**Reasons:**
1. Smallest memory footprint -- most realistic for on-device simulation
2. LLaMA architecture -- widest ecosystem support (HuggingFace, PEFT, Flower)
3. Already used in FLoRA (NeurIPS 2024) -- enables direct comparison
4. Apache 2.0 license -- no restrictions
5. GQA architecture -- actually designed for memory-constrained deployment
6. Fastest inference throughput (85 tok/s)

**Secondary model:** Gemma-2B for generalizability experiments (different architecture, MQA)

**References:**
- TinyLlama: huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Architecture comparison: josedavidbaena.com/blog/tiny-language-models/tiny-llm-architecture-comparison
- Unsloth for efficient fine-tuning: github.com/unslothai/unsloth

---

## 6. Datasets for Federated Text Tasks

### Recommended Datasets

**Tier 1 -- Standard FL Benchmarks (for comparison with existing work):**

1. **GLUE/SuperGLUE Subsets:**
   - SST-2 (sentiment, binary, 67K training samples) -- most commonly used
   - MNLI (natural language inference, 393K samples, 3 classes)
   - QNLI (question NLI, 105K samples)
   - These are used in pFL-Bench and many FL papers

2. **Databricks-dolly-15k** -- Used in FLoRA. 15K instruction-following samples. Good for MMLU evaluation.

3. **Alpaca / Alpaca-GPT4** -- 52K instruction-tuning samples. Used in FLoRA and Mozilla Blueprint.

**Tier 2 -- Federated-Specific Datasets:**

4. **Reddit (comment summarization)** -- Used in HetLoRA. Naturally federated (each user = client). 298 users with 100+ samples each.

5. **Multi-Session Chat (MSC)** -- Used in HetLoRA. 100 users, dialogue data. Natural user-level federation.

6. **ShareGPT** -- Used in FLoRA for chat tasks. Multi-turn conversations.

**Tier 3 -- For Personalization Evaluation:**

7. **Natural Instructions** -- Large-scale instruction dataset with task-level splits for heterogeneity.

### Creating Non-IID Splits

**Dirichlet Distribution Method (Standard Approach):**
- Sample label proportions from Dir(alpha) for each client
- alpha=0.1: Extreme non-IID (most clients see only 1-2 labels)
- alpha=0.5: Moderate non-IID (standard in most papers)
- alpha=1.0: Mild non-IID
- alpha=100: Approximately IID

**Flower's Built-in Partitioner:**
```python
from flwr_datasets.partitioner import DirichletPartitioner
partitioner = DirichletPartitioner(
    num_partitions=50,
    partition_by="label",
    alpha=0.5,
    min_partition_size=100
)
```

**For Text/Instruction Data:**
- Partition by topic/category rather than label
- Use natural user-level splits when available (Reddit, MSC)
- For Alpaca: cluster by instruction type (open QA, classification, generation, etc.)

**Recommended experimental grid:**
- alpha in {0.1, 0.5, 1.0} to show robustness across non-IID levels
- Number of clients: 10, 50, 100

**References:**
- NIID-Bench: github.com/Xtra-Computing/NIID-Bench (ICDE 2022)
- pFL-Bench: arxiv.org/abs/2206.03655
- Flower DirichletPartitioner: flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.DirichletPartitioner.html
- GLUE benchmark: gluebenchmark.com

---

## 7. Realistic Battery Profiles for Phones

### Power Consumption Numbers

**Smartphone Baseline Power:**
- Average smartphone power consumption: ~1,970 mW (~2W) during normal use
- Screen-on idle: ~800-1200 mW
- Heavy CPU load: ~3000-5000 mW
- GPU/NPU ML inference: ~2000-4000 mW

**On-Device ML Energy Measurements:**
- Google Edge TPU (in Pixel phones): ~2W for 8-bit integer ML operations
- ML inference energy varies dramatically with resolution: 4.1% of total energy at 100x100 pixels to 33.9% at 600x600 pixels
- MIT research achieved 73% power reduction for neural network inference on phones

**Federated Learning Energy (from Google's measurements on Pixel 6):**
- Average user holds ~16 training examples
- 10 local epochs = 160 examples processed per device session
- ~186 kJ total smartphone energy for training component (across 62,500 device sessions)
- Per-device per-session: ~3 J for training on ~16 examples
- FL total (~567 kJ) is ~12x datacenter training (~47 kJ) for equivalent task

**Typical Battery Capacities:**
- Budget phone: 3,000-4,000 mAh (11-15 Wh)
- Mid-range: 4,000-5,000 mAh (15-19 Wh)
- Flagship: 5,000-6,000 mAh (19-23 Wh)
- At 2W ML training load: drains ~10-13% per hour on a 4500mAh battery

### Public Battery Trace Datasets

1. **GreenHub Dataset:**
   - 23+ million anonymous data samples
   - 700+ million data points on processes
   - 1,600+ device brands, 11,800+ smartphone models
   - 50+ Android versions
   - Access: SQL, REST API, CSV/Parquet dump
   - URL: greenhub.di.ubi.pt

2. **Kaggle: Mobile Battery with Time:**
   - Battery level traces over time
   - URL: kaggle.com/datasets/rahulgarg28/mobile-battery-with-time

3. **CALCE Battery Data (UMD):**
   - Li-ion battery discharge/charge cycles
   - State estimation, remaining useful life data
   - URL: calce.umd.edu/battery-data

4. **BatteryML (Microsoft Research):**
   - Open-source battery analysis platform

### Battery Drain Model for Simulation

```python
# Simple battery drain model for FL simulation
class BatteryModel:
    def __init__(self, capacity_mAh=4500, voltage=3.8):
        self.capacity_Wh = capacity_mAh * voltage / 1000  # ~17.1 Wh
        self.remaining_Wh = self.capacity_Wh
        self.idle_power_W = 0.5       # screen-off idle
        self.training_power_W = 3.0   # ML training load
        self.comm_power_W = 1.5       # uploading model updates

    def train_round(self, train_time_s, comm_time_s):
        energy_train = self.training_power_W * train_time_s / 3600
        energy_comm = self.comm_power_W * comm_time_s / 3600
        self.remaining_Wh -= (energy_train + energy_comm)
        return self.remaining_Wh / self.capacity_Wh  # remaining fraction

    def can_participate(self, threshold=0.15):
        return (self.remaining_Wh / self.capacity_Wh) > threshold
```

**References:**
- CACM energy study: cacm.acm.org/research/energy-and-emissions-of-machine-learning-on-smartphones-vs-the-cloud/
- Smartphone energy modeling: arxiv.org/pdf/2012.10246
- GreenHub: dl.acm.org/doi/abs/10.1007/s10664-020-09925-5
- MCM 2026 Problem A (battery drain modeling): immchallenge.org/mcm/2026_MCM_Problem_A.pdf

---

## 8. Baselines for Comparison

### Must-Have Baselines

| # | Method | Type | Reference |
|---|--------|------|-----------|
| 1 | **FedAvg + HomLoRA** | Uniform rank across all clients | McMahan et al., 2017 + Hu et al., 2021 |
| 2 | **FedProx + HomLoRA** | FedAvg with proximal regularization | Li et al., MLSYS 2020 |
| 3 | **HetLoRA** | Heterogeneous ranks, sparsity-weighted aggregation | Cho et al., EMNLP 2024 (arxiv 2401.06432) |
| 4 | **FLoRA** | Stacking-based heterogeneous aggregation | Wang et al., NeurIPS 2024 (arxiv 2409.05976) |
| 5 | **FlexLoRA** | SVD-based adaptive rank redistribution | Bai et al., NeurIPS 2024 |
| 6 | **FFA-LoRA** | Freeze-A, only train B matrix | Sun et al., 2024 |
| 7 | **FedSA-LoRA** | Share A matrices, keep B local | Yang et al., 2024 |
| 8 | **Random client selection** | Battery-unaware random selection | Standard baseline |
| 9 | **EAFL** | Battery-aware client selection (no rank adaptation) | Arouj et al., 2022 (arxiv 2208.04505) |
| 10 | **BEFL** | Energy-balancing client selection | arxiv 2412.03950 |

### Additional Worth Considering

| # | Method | Type | Reference |
|---|--------|------|-----------|
| 11 | **FedEx-LoRA** | Exact aggregation with residual error | arxiv 2410.09432 |
| 12 | **LoRA-FAIR** | Aggregation + initialization refinement | Bian et al., ICCV 2025 |
| 13 | **FedALT** | Adaptive local training + RoTW LoRA | arxiv 2503.11880 |
| 14 | **AFLoRA** | Adaptive resource-efficient FL | arxiv 2505.24773 |
| 15 | **LeanFed** | Adaptive data fraction per battery | arxiv 2412.02289 |
| 16 | **Centralized LoRA** | Upper bound (no federation) | - |
| 17 | **Local-only LoRA** | No aggregation, each client trains alone | - |

### Recommended Minimum Baseline Set (for paper)
1. FedAvg + HomLoRA (r=8) -- standard FL baseline
2. FedAvg + HomLoRA (r=16) -- higher capacity baseline
3. HetLoRA -- heterogeneous rank baseline (no battery awareness)
4. FLoRA -- SOTA heterogeneous aggregation
5. EAFL -- battery-aware baseline (no rank adaptation)
6. Random selection + HomLoRA -- naive baseline
7. Centralized LoRA -- upper bound
8. **Your method** -- battery-aware adaptive rank

---

## 9. Evaluation Metrics

### Primary Metrics

**Task Performance:**
- **Accuracy / F1-score** on held-out test set (per-client and global)
- **Perplexity** for language modeling tasks
- **MMLU score** for instruction-following (used by FLoRA)
- **MT-bench score** for chat quality (used by FLoRA)
- **RougeL** for summarization tasks (used by HetLoRA)

**Convergence:**
- **Rounds to target accuracy** (e.g., rounds to reach 90% of centralized performance)
- **Accuracy vs. communication rounds** curve
- **Total communication cost** (bytes transferred) to reach target accuracy

**Energy / Battery:**
- **Total energy consumed** across all clients (in Joules or Wh)
- **Average remaining battery** after N rounds
- **Client dropout rate** (fraction of clients that run out of battery)
- **Energy per accuracy point** (energy efficiency ratio)

### Fairness Metrics

- **Accuracy variance** across clients (lower = fairer)
- **Min-max accuracy gap** (worst client vs best client)
- **Jain's fairness index** on accuracy distribution
- **Energy variance** across clients (from BEFL)
- **Battery depletion fairness** -- variance in remaining battery levels

### Secondary Metrics

- **Per-round wall-clock time** (simulated)
- **Total trainable parameters per client** (shows rank adaptation)
- **Average LoRA rank assigned** over time
- **Rank distribution** per round (histogram)

### Recommended Evaluation Grid

| Metric | How to Report |
|--------|---------------|
| Global accuracy | Single number + convergence curve |
| Per-client accuracy | Mean +/- std, min, max |
| Rounds to 90% performance | Single number |
| Total energy (J) | Single number |
| Client dropout rate | Percentage |
| Accuracy fairness | Jain's index + variance |
| Energy fairness | Variance of remaining battery |
| Communication cost | Total MB transferred |

**References:**
- Fairness metrics: arxiv.org/pdf/2012.10069
- FairEnergy: arxiv.org/html/2511.15454
- Holistic FL evaluation: arxiv.org/abs/2402.02360
- pFL-Bench metrics: arxiv.org/abs/2206.03655

---

## 10. Simulating Battery / Device Heterogeneity Without Real Phones

### Approach 1: Flower Simulation Engine (Recommended)

Flower's Virtual Client Engine runs all clients in a single machine:
```python
# Configure heterogeneous resources per client
backend_config = {
    "client_resources": {
        "num_cpus": 2,
        "num_gpus": 0.1  # fractional GPU
    }
}
run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=50,
    backend_config=backend_config
)
```

### Approach 2: Simulated Battery State (What Most Papers Do)

Maintain a software battery model for each virtual client:

```python
import numpy as np

class DeviceFleet:
    def __init__(self, num_devices=50):
        # Heterogeneous battery capacities (mAh)
        self.capacities = np.random.choice(
            [3000, 4000, 4500, 5000, 6000],
            size=num_devices,
            p=[0.1, 0.25, 0.3, 0.25, 0.1]
        )
        # Device tiers: weak, medium, strong
        self.tiers = np.random.choice(
            ['weak', 'medium', 'strong'],
            size=num_devices,
            p=[0.3, 0.5, 0.2]
        )
        # Training power consumption (W) by tier
        self.power = {
            'weak': 2.0, 'medium': 3.0, 'strong': 4.5
        }
        # Training speed (relative) by tier
        self.speed = {
            'weak': 1.0, 'medium': 0.6, 'strong': 0.3  # seconds per step
        }
        # LoRA rank capacity by tier
        self.max_rank = {
            'weak': 8, 'medium': 16, 'strong': 32
        }
        # Initialize battery levels (random between 30-100%)
        self.battery_pct = np.random.uniform(0.3, 1.0, num_devices)

    def simulate_round(self, device_id, lora_rank, num_local_steps):
        tier = self.tiers[device_id]
        train_time = self.speed[tier] * num_local_steps * (lora_rank / 8)
        energy_Wh = self.power[tier] * train_time / 3600
        capacity_Wh = self.capacities[device_id] * 3.8 / 1000
        self.battery_pct[device_id] -= energy_Wh / capacity_Wh

        # Random recharging events
        if np.random.random() < 0.1:  # 10% chance of partial recharge
            self.battery_pct[device_id] = min(1.0,
                self.battery_pct[device_id] + np.random.uniform(0.1, 0.3))

        return self.battery_pct[device_id]
```

### Approach 3: Existing Simulation Frameworks

1. **LEAF Framework** -- Introduced datasets + non-IID partitioning for realistic FL benchmarks
2. **FedScale** -- Large-scale FL benchmarking platform with system heterogeneity simulation
3. **Plato** (github.com/TL-System/plato) -- Simulates client resources using logs collected from physical devices (network speed, memory logs)
4. **FLASH** -- Fork of LEAF with explicit device type assignment (weak/medium/strong training speeds, network conditions)
5. **FLTK** -- Heterogeneity-aware simulation with comparative evaluation

### Approach 4: Delay Injection (Simulating Compute Heterogeneity)

```python
# In Flower client, simulate device speed differences
import time

class SlowClient(FlowerClient):
    def fit(self, parameters, config):
        # Simulate device speed based on tier
        tier = config.get("device_tier", "medium")
        delay_factor = {"weak": 3.0, "medium": 1.0, "strong": 0.5}

        # Actual training
        result = super().fit(parameters, config)

        # Add simulated delay (for wall-clock time experiments)
        time.sleep(delay_factor[tier] * base_training_time)

        return result
```

### Approach 5: Use GreenHub Battery Traces

Load real battery traces from GreenHub (23M+ samples) and replay them:
- Map each virtual FL client to a real device trace
- Use actual discharge curves instead of linear models
- Accounts for screen-on time, background apps, charging events

### What Other Papers Do (Summary)

| Paper | Simulation Method |
|-------|-------------------|
| HetLoRA | Assign ranks randomly from distribution, no battery |
| FLoRA | Fixed heterogeneous rank assignment [64,32,16,...,4] |
| BEFL | Energy model with computation + communication costs |
| FedBacys | Battery model with energy harvesting, cyclic scheduling |
| EAFL | Battery level tracking, cherry-pick high-battery clients |
| LeanFed | Battery-constrained simulation, adaptive data fraction |
| AdaptiveFL | 3 device tiers (weak/medium/strong) |

### Recommended Simulation Design for Your Paper

1. Use Flower simulation engine with 50 virtual clients
2. 3 device tiers with realistic power profiles from literature
3. Battery model using GreenHub traces OR simple discharge model
4. Per-round: query battery level -> decide LoRA rank -> train -> drain battery -> (maybe) recharge
5. Report: accuracy, energy, fairness, dropout rate

**References:**
- Plato: github.com/TL-System/plato
- LEAF: leaf.cmu.edu
- FedScale: github.com/SymbioticLab/FedScale
- Flower simulation: flower.ai/docs/framework/
- FLASH: heterogeneity-aware FL platform
- AdaptiveFL: arxiv.org/html/2311.13166v2

---

## Summary: Proposed Experimental Design

### Model
- Primary: TinyLlama 1.1B (matches FLoRA, lightweight, Apache 2.0)
- Secondary: Gemma-2B (for generalizability)

### Datasets
- Alpaca-GPT4 (instruction tuning, MMLU eval) -- matches FLoRA/Mozilla baseline
- Reddit summarization (RougeL eval) -- matches HetLoRA
- Non-IID splits: Dirichlet alpha in {0.1, 0.5, 1.0}

### Client Configuration
- 50 virtual clients in Flower simulation
- 3 tiers: weak (30%), medium (50%), strong (20%)
- Battery: 3000-6000 mAh, random initial charge 30-100%
- Max LoRA ranks by tier: weak=8, medium=16, strong=32

### Your Novel Method
- Each round: read battery level -> assign LoRA rank proportional to remaining energy
- High battery (>70%): use maximum rank for tier
- Medium battery (30-70%): use half of max rank
- Low battery (15-30%): use minimum rank (r=2 or 4)
- Below 15%: skip round (preserve battery)
- Aggregation: FLoRA stacking (proven noise-free for heterogeneous ranks)

### Baselines (8 methods)
1. FedAvg + HomLoRA r=8
2. FedAvg + HomLoRA r=16
3. HetLoRA (heterogeneous, no battery awareness)
4. FLoRA (stacking, no battery awareness)
5. EAFL (battery-aware selection, no rank adaptation)
6. Random selection + HomLoRA
7. Centralized LoRA (upper bound)
8. Local-only LoRA (lower bound)

### Metrics
- Global accuracy (MMLU or RougeL)
- Per-client accuracy (mean, std, min, max)
- Rounds to 90% of centralized performance
- Total energy consumed (J)
- Client dropout rate
- Jain's fairness index (accuracy + energy)
- Communication cost (MB)

### Expected Contribution
First paper to combine battery-aware client management with adaptive LoRA rank selection in federated fine-tuning. The gap: existing work either adapts rank (HetLoRA, FLoRA) OR manages energy (BEFL, EAFL, LeanFed), but nobody does both simultaneously with rank as the control knob for energy management.
