"""
BatteryLoRA Federated Server
==============================
Orchestrates the federated training process:
1. Selects clients for each round
2. Queries battery states and assigns ranks
3. Distributes sub-adapters at appropriate ranks
4. Collects and aggregates updates using FLoRA stacking
5. Tracks metrics across rounds
"""

import os
import json
import random
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from peft import get_peft_model_state_dict

from config import ExperimentConfig
from battery_simulator import BatterySimulator
from rank_policy import RankPolicy, create_rank_policy
from flora_aggregation import (
    aggregate_flora,
    extract_sub_adapter,
    compute_communication_cost,
)
from model_utils import load_base_model, prepare_base_for_lora, apply_lora


class BatteryLoRAServer:
    """
    Central server for BatteryLoRA federated training.

    This server coordinates the training process, managing:
    - Client selection
    - Battery-aware rank assignment
    - FLoRA aggregation of heterogeneous adapters
    - Metrics tracking
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Initialize battery simulator
        self.battery_sim = BatterySimulator(
            num_clients=cfg.federated.num_clients,
            battery_cfg=cfg.battery,
            tier_cfg=cfg.device_tiers,
            seed=cfg.seed,
        )

        # Initialize rank policy
        self.rank_policy = create_rank_policy(
            policy_cfg=cfg.rank_policy,
            tier_cfg=cfg.device_tiers,
            battery_capacity_wh=cfg.battery.capacity_wh,
            reserve_percent=cfg.battery.reserve_percent,
            energy_per_round=cfg.battery.energy_per_round,
            seed=cfg.seed,
        )

        # Initialize global model and extract initial LoRA state
        print("Loading base model and initializing global LoRA adapter...")
        self.base_model, self.tokenizer = load_base_model(cfg.model)
        self.base_model = prepare_base_for_lora(self.base_model)

        # Create initial LoRA at max rank to get parameter structure
        init_peft = apply_lora(self.base_model, cfg.lora, cfg.lora.max_rank)

        # Store global LoRA state as numpy arrays
        self.global_state: Dict[str, np.ndarray] = {
            k: v.cpu().float().numpy()
            for k, v in get_peft_model_state_dict(init_peft).items()
        }
        self.global_state_keys = list(self.global_state.keys())

        # Unload to get clean base model back for client use
        self.base_model = init_peft.unload()
        del init_peft

        # Metrics tracking
        self.round_metrics: List[Dict] = []
        self.communication_bytes_total = 0

    def select_clients(self, round_num: int) -> List[int]:
        """Select clients for this round (random subset of active clients)."""
        active = self.battery_sim.get_active_clients()
        eligible = [
            cid for cid in active
            if self.battery_sim.can_participate(cid)
        ]

        k = min(self.cfg.federated.clients_per_round, len(eligible))
        if k == 0:
            print(f"  WARNING: No eligible clients in round {round_num}!")
            return []

        return self.rng.sample(eligible, k)

    def get_client_rank(self, client_id: int) -> int:
        """Determine LoRA rank for a client based on its battery state."""
        device_state = self.battery_sim.get_device_state(client_id)
        return self.rank_policy.get_rank(device_state)

    def prepare_client_adapter(self, client_id: int, rank: int) -> List[np.ndarray]:
        """
        Extract a sub-adapter at the client's rank from the global adapter.
        Returns a list of numpy arrays (one per LoRA parameter).
        """
        sub_state = extract_sub_adapter(self.global_state, rank)
        # Track communication (download)
        self.communication_bytes_total += compute_communication_cost(sub_state)
        return [sub_state[k] for k in self.global_state_keys]

    def aggregate_round(
        self,
        client_results: List[Tuple[int, List[np.ndarray], int, int, Dict]],
    ):
        """
        Aggregate client updates using FLoRA stacking.

        Args:
            client_results: List of (client_id, parameters, num_samples, rank, metrics)
        """
        # Convert to format expected by aggregate_flora
        client_updates = []
        for client_id, params, num_samples, rank, metrics in client_results:
            # Reconstruct state dict from parameter list
            client_state = {
                k: v for k, v in zip(self.global_state_keys, params)
            }
            client_updates.append((client_state, num_samples, rank))

            # Track communication (upload)
            self.communication_bytes_total += compute_communication_cost(client_state)

        # FLoRA stacking aggregation
        self.global_state = aggregate_flora(
            self.global_state,
            client_updates,
            max_rank=self.cfg.lora.max_rank,
        )

    def run_training(self):
        """
        Main training loop.

        This is the core federated training process:
        1. For each round, select clients
        2. Assign ranks based on battery
        3. Distribute adapters
        4. Collect and aggregate updates
        5. Track metrics
        """
        print(f"\n{'='*60}")
        print(f"BatteryLoRA Training")
        print(f"Policy: {self.rank_policy.name}")
        print(f"Clients: {self.cfg.federated.num_clients}")
        print(f"Rounds: {self.cfg.federated.num_rounds}")
        print(f"{'='*60}\n")

        for round_num in range(1, self.cfg.federated.num_rounds + 1):
            round_info = self._run_single_round(round_num)
            self.round_metrics.append(round_info)

            # Print progress
            if round_num % 5 == 0 or round_num == 1:
                battery_stats = self.battery_sim.get_summary_stats()
                print(
                    f"Round {round_num:3d}/{self.cfg.federated.num_rounds} | "
                    f"Clients: {round_info['num_clients_trained']} | "
                    f"Ranks: {round_info['ranks_used']} | "
                    f"Avg Loss: {round_info['avg_train_loss']:.4f} | "
                    f"Active: {battery_stats['active_clients']}/{self.cfg.federated.num_clients} | "
                    f"Avg Battery: {battery_stats['avg_battery']:.1f}%"
                )

            # Save checkpoint
            if round_num % self.cfg.federated.save_every_rounds == 0:
                self._save_checkpoint(round_num)

        # Final summary
        self._print_final_summary()
        self._save_results()

    def _run_single_round(self, round_num: int) -> Dict:
        """Execute a single federated round."""
        # Environment changes (charging state toggles etc.)
        self.battery_sim.simulate_environment_changes(round_num)

        # Select clients
        selected = self.select_clients(round_num)

        if not selected:
            return {
                "round": round_num,
                "num_clients_trained": 0,
                "ranks_used": {},
                "avg_train_loss": float("inf"),
            }

        # Assign ranks and prepare adapters
        client_configs = []
        rank_counts = {}
        for cid in selected:
            rank = self.get_client_rank(cid)
            client_configs.append((cid, rank))
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        # Simulate client training
        # In the actual Flower simulation, this happens in parallel.
        # Here we do it sequentially for clarity.
        from client import BatteryLoRAClient

        client_results = []
        for cid, rank in client_configs:
            # Prepare adapter at client's rank
            adapter_params = self.prepare_client_adapter(cid, rank)

            # Create a fresh model with the right rank for this client
            peft_model = apply_lora(self.base_model, self.cfg.lora, rank)

            # Client trains locally
            client = BatteryLoRAClient(cid, self.cfg, rank)
            updated_params, num_samples, metrics = client.train(
                model=peft_model,
                tokenizer=self.tokenizer,
                parameters=adapter_params,
                current_round=round_num,
            )

            client_results.append((cid, updated_params, num_samples, rank, metrics))

            # Update battery after training
            self.battery_sim.update_after_training(cid, rank)

            # Unload adapter to free the base model for next client
            self.base_model = peft_model.unload()
            del peft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update idle clients' battery
        trained_ids = set(cid for cid, _ in client_configs)
        for cid in range(self.cfg.federated.num_clients):
            if cid not in trained_ids:
                self.battery_sim.update_idle_round(cid)

        # Aggregate
        self.aggregate_round(client_results)

        # Compute round metrics
        avg_loss = np.mean([m["train_loss"] for _, _, _, _, m in client_results])

        return {
            "round": round_num,
            "num_clients_trained": len(client_results),
            "ranks_used": rank_counts,
            "avg_train_loss": avg_loss,
            "client_details": [
                {
                    "client_id": cid,
                    "rank": rank,
                    "loss": m["train_loss"],
                    "samples": m["num_samples"],
                }
                for cid, _, _, rank, m in client_results
            ],
        }

    def _save_checkpoint(self, round_num: int):
        """Save the global LoRA adapter."""
        save_dir = os.path.join(
            self.cfg.federated.save_path,
            self.cfg.experiment_name,
            f"round_{round_num}",
        )
        os.makedirs(save_dir, exist_ok=True)

        # Save as numpy
        np.savez(
            os.path.join(save_dir, "global_lora.npz"),
            **self.global_state,
        )
        print(f"  Checkpoint saved: {save_dir}")

    def _save_results(self):
        """Save all metrics to JSON."""
        results_dir = os.path.join("results", self.cfg.experiment_name)
        os.makedirs(results_dir, exist_ok=True)

        # Round metrics
        with open(os.path.join(results_dir, "round_metrics.json"), "w") as f:
            json.dump(self.round_metrics, f, indent=2, default=str)

        # Battery stats
        battery_stats = self.battery_sim.get_summary_stats()
        with open(os.path.join(results_dir, "battery_stats.json"), "w") as f:
            json.dump(battery_stats, f, indent=2)

        # Per-device stats
        device_stats = {}
        for cid, dev in self.battery_sim.devices.items():
            device_stats[cid] = {
                "tier": dev.tier,
                "final_battery": dev.battery_percent,
                "is_active": dev.is_active,
                "total_energy_wh": dev.total_energy_consumed_wh,
                "rounds_participated": dev.rounds_participated,
                "battery_history": dev.battery_history,
                "rank_history": dev.rank_history,
            }
        with open(os.path.join(results_dir, "device_stats.json"), "w") as f:
            json.dump(device_stats, f, indent=2)

        # Summary
        summary = {
            "experiment": self.cfg.experiment_name,
            "policy": self.rank_policy.name,
            "num_clients": self.cfg.federated.num_clients,
            "num_rounds": self.cfg.federated.num_rounds,
            "total_communication_bytes": self.communication_bytes_total,
            "total_communication_mb": self.communication_bytes_total / (1024 * 1024),
            "battery_summary": battery_stats,
            "final_avg_loss": (
                self.round_metrics[-1]["avg_train_loss"]
                if self.round_metrics
                else None
            ),
        }
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {results_dir}/")

    def _print_final_summary(self):
        """Print a summary of the training run."""
        stats = self.battery_sim.get_summary_stats()
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Policy:                  {self.rank_policy.name}")
        print(f"Active clients:          {stats['active_clients']}/{self.cfg.federated.num_clients}")
        print(f"Dropout rate:            {stats['dropout_rate']:.1%}")
        print(f"Total energy (Wh):       {stats['total_energy_wh']:.2f}")
        print(f"Avg energy/client (Wh):  {stats['avg_energy_per_client']:.2f}")
        print(f"Energy std (Wh):         {stats['energy_std']:.2f}")
        print(f"Jain fairness index:     {stats['jain_fairness_index']:.4f}")
        print(f"Avg rounds participated: {stats['avg_rounds_participated']:.1f}")
        print(f"Total communication:     {self.communication_bytes_total / (1024*1024):.1f} MB")
        if self.round_metrics:
            print(f"Final avg train loss:    {self.round_metrics[-1]['avg_train_loss']:.4f}")
        print(f"{'='*60}\n")
