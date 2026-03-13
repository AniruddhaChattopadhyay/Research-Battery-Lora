"""
BatteryLoRA Experiment Runner
==============================
Main entry point for running experiments.

Usage:
    # Run main experiment (BatteryLoRA with threshold policy)
    python run_experiment.py --experiment e1_main

    # Run a specific baseline
    python run_experiment.py --experiment baseline_homolora --rank 8

    # Run ablation on policy
    python run_experiment.py --experiment e2_ablation --policy continuous

    # Run non-IID sensitivity
    python run_experiment.py --experiment e3_noniid --alpha 0.1

    # Quick test with fewer rounds/clients
    python run_experiment.py --experiment e1_main --quick
"""

import argparse
import sys
import time

from config import (
    ExperimentConfig,
    get_e1_main_config,
    get_e2_ablation_config,
    get_e3_noniid_config,
    get_e4_scale_config,
    get_baseline_homolora_config,
    get_baseline_hetlora_config,
    get_baseline_local_only_config,
)
from server import BatteryLoRAServer


def apply_quick_mode(cfg: ExperimentConfig) -> ExperimentConfig:
    """Reduce experiment size for quick testing."""
    cfg.federated.num_clients = 10
    cfg.federated.clients_per_round = 3
    cfg.federated.num_rounds = 3
    cfg.training.local_epochs = 1
    cfg.training.batch_size = 8
    cfg.model.max_seq_length = 256
    cfg.federated.save_every_rounds = 3
    cfg.federated.max_samples_per_client = 50  # Only 50 examples per client
    return cfg


def main():
    parser = argparse.ArgumentParser(description="BatteryLoRA Experiments")

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "e1_main",
            "e2_ablation",
            "e3_noniid",
            "e4_scale",
            "baseline_homolora",
            "baseline_hetlora",
            "baseline_local",
        ],
        help="Which experiment to run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")

    # Experiment-specific args
    parser.add_argument("--rank", type=int, default=8, help="Fixed rank for HomLoRA baseline")
    parser.add_argument(
        "--policy",
        type=str,
        default="threshold",
        choices=["threshold", "continuous", "binary", "random", "fixed", "static_tier"],
        help="Rank policy for ablation",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID")
    parser.add_argument("--num-clients", type=int, default=50, help="Number of clients")

    args = parser.parse_args()

    # Create config based on experiment type
    if args.experiment == "e1_main":
        cfg = get_e1_main_config(seed=args.seed)
    elif args.experiment == "e2_ablation":
        cfg = get_e2_ablation_config(policy=args.policy, seed=args.seed)
    elif args.experiment == "e3_noniid":
        cfg = get_e3_noniid_config(alpha=args.alpha, seed=args.seed)
    elif args.experiment == "e4_scale":
        cfg = get_e4_scale_config(num_clients=args.num_clients, seed=args.seed)
    elif args.experiment == "baseline_homolora":
        cfg = get_baseline_homolora_config(rank=args.rank, seed=args.seed)
    elif args.experiment == "baseline_hetlora":
        cfg = get_baseline_hetlora_config(seed=args.seed)
    elif args.experiment == "baseline_local":
        cfg = get_baseline_local_only_config(seed=args.seed)
    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)

    # Apply quick mode if requested
    if args.quick:
        cfg = apply_quick_mode(cfg)
        print("QUICK MODE: Reduced clients/rounds for testing")

    # Print experiment info
    print(f"\nExperiment: {cfg.experiment_name}")
    print(f"Model: {cfg.model.name}")
    print(f"Clients: {cfg.federated.num_clients}")
    print(f"Rounds: {cfg.federated.num_rounds}")
    print(f"Policy: {cfg.rank_policy.policy_type}")
    print(f"Seed: {cfg.seed}")
    print()

    # Run
    start_time = time.time()
    server = BatteryLoRAServer(cfg)
    server.run_training()
    elapsed = time.time() - start_time

    print(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
