"""
BatteryLoRA Configuration
=========================
All hyperparameters and experiment settings in one place.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the base language model."""
    name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    quantization: int = 4  # 0 = none, 4 = 4-bit, 8 = 8-bit
    gradient_checkpointing: bool = True
    max_seq_length: int = 512


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    # Available ranks for battery-aware selection
    available_ranks: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    max_rank: int = 32  # Global adapter rank on the server
    lora_alpha: int = 64  # Alpha scaling factor (typically 2x max rank)
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@dataclass
class BatteryConfig:
    """Configuration for battery simulation."""
    # Battery capacity in Wh (typical 4500mAh * 3.8V = 17.1 Wh)
    capacity_wh: float = 17.1
    # Reserve threshold — device won't train below this
    reserve_percent: float = 15.0
    # Dropout threshold — device drops out entirely below this
    dropout_percent: float = 10.0
    # Energy cost per training round by rank (Wh), estimated
    # These will be calibrated during experiments
    energy_per_round: dict = field(default_factory=lambda: {
        2: 0.08,
        4: 0.12,
        8: 0.20,
        16: 0.35,
        32: 0.60,
    })
    # Whether to use real battery traces from GreenHub or synthetic
    use_real_traces: bool = False


@dataclass
class DeviceTierConfig:
    """Configuration for simulated device tiers."""
    # Distribution of device tiers among clients
    # high = flagship phones, mid = mid-range, low = budget
    tier_distribution: dict = field(default_factory=lambda: {
        "high": 0.20,   # 20% flagship
        "mid": 0.50,    # 50% mid-range
        "low": 0.30,    # 30% budget
    })
    # Maximum rank each tier can handle (hardware limit)
    tier_max_rank: dict = field(default_factory=lambda: {
        "high": 32,
        "mid": 16,
        "low": 8,
    })


@dataclass
class TrainingConfig:
    """Configuration for local training on each client."""
    local_epochs: int = 3
    batch_size: int = 4
    learning_rate_max: float = 5e-4
    learning_rate_min: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_grad_norm: float = 1.0


@dataclass
class FederatedConfig:
    """Configuration for the federated learning process."""
    num_clients: int = 50
    clients_per_round: int = 10  # 20% participation rate
    num_rounds: int = 100
    # Data partitioning
    dataset_name: str = "vicgalle/alpaca-gpt4"
    dirichlet_alpha: float = 0.5  # Controls non-IID degree
    # Limit samples per client (0 = no limit, used for quick testing)
    max_samples_per_client: int = 0
    # Checkpointing
    save_every_rounds: int = 10
    save_path: str = "results/checkpoints"


@dataclass
class RankPolicyConfig:
    """Configuration for the battery-to-rank mapping policy."""
    # Which policy to use: "threshold", "continuous", "binary", "random", "fixed"
    policy_type: str = "threshold"
    # For "fixed" policy — what rank to use
    fixed_rank: int = 8
    # For "threshold" policy — battery thresholds
    thresholds: List[int] = field(default_factory=lambda: [80, 60, 40, 20])
    # For "continuous" policy — minimum future rounds to guarantee
    min_future_rounds: int = 5


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    device_tiers: DeviceTierConfig = field(default_factory=DeviceTierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    rank_policy: RankPolicyConfig = field(default_factory=RankPolicyConfig)
    # Experiment metadata
    experiment_name: str = "battery_lora_main"
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"


# ─── Preset Configurations for Different Experiments ───────────────────────

def get_e1_main_config(seed: int = 42) -> ExperimentConfig:
    """E1: Main comparison experiment."""
    cfg = ExperimentConfig()
    cfg.experiment_name = f"e1_main_seed{seed}"
    cfg.seed = seed
    return cfg


def get_e2_ablation_config(policy: str, seed: int = 42) -> ExperimentConfig:
    """E2: Ablation on rank policies."""
    cfg = ExperimentConfig()
    cfg.experiment_name = f"e2_ablation_{policy}_seed{seed}"
    cfg.rank_policy.policy_type = policy
    cfg.seed = seed
    return cfg


def get_e3_noniid_config(alpha: float, seed: int = 42) -> ExperimentConfig:
    """E3: Non-IID sensitivity."""
    cfg = ExperimentConfig()
    cfg.experiment_name = f"e3_noniid_alpha{alpha}_seed{seed}"
    cfg.federated.dirichlet_alpha = alpha
    cfg.seed = seed
    return cfg


def get_e4_scale_config(num_clients: int, seed: int = 42) -> ExperimentConfig:
    """E4: Scale sensitivity."""
    cfg = ExperimentConfig()
    cfg.experiment_name = f"e4_scale_{num_clients}clients_seed{seed}"
    cfg.federated.num_clients = num_clients
    cfg.federated.clients_per_round = max(3, num_clients // 5)
    cfg.seed = seed
    return cfg


# ─── Baseline Configurations ──────────────────────────────────────────────

def get_baseline_homolora_config(rank: int = 8, seed: int = 42) -> ExperimentConfig:
    """Baseline: FedAvg + HomLoRA (all clients use the same fixed rank)."""
    cfg = ExperimentConfig()
    cfg.experiment_name = f"baseline_homolora_r{rank}_seed{seed}"
    cfg.rank_policy.policy_type = "fixed"
    cfg.rank_policy.fixed_rank = rank
    cfg.seed = seed
    return cfg


def get_baseline_hetlora_config(seed: int = 42) -> ExperimentConfig:
    """Baseline: HetLoRA (rank based on static device tier, no battery)."""
    cfg = ExperimentConfig()
    cfg.experiment_name = f"baseline_hetlora_seed{seed}"
    cfg.rank_policy.policy_type = "static_tier"
    cfg.seed = seed
    return cfg


def get_baseline_local_only_config(seed: int = 42) -> ExperimentConfig:
    """Baseline: Local-only LoRA (no federation)."""
    cfg = ExperimentConfig()
    cfg.experiment_name = f"baseline_local_only_seed{seed}"
    cfg.federated.clients_per_round = 0  # No aggregation
    cfg.seed = seed
    return cfg
