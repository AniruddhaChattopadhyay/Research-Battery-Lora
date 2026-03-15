"""
BatteryLoRA — Full Experiment Runner for A100
==============================================
Runs experiments for the NGEN-AI 2026 paper.

Each experiment is run individually and results are saved to Google Drive
after completion, so you can resume across Colab sessions.

Usage:
    python run_all.py --quick                          # Sanity test (~1 min)
    python run_all.py --run battery_lora               # Our method
    python run_all.py --run homolora_r8                # Baseline
    python run_all.py --run all                        # All 7 experiments
    python run_all.py --run all --seed 123             # Different seed
    python run_all.py --summary                        # Print results table
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time

import torch

# ── Setup ─────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DRIVE_RESULTS = '/content/drive/MyDrive/battery_lora_results'

from config import (
    ExperimentConfig,
    get_e1_main_config,
    get_e2_ablation_config,
    get_baseline_homolora_config,
    get_baseline_hetlora_config,
)
from server import BatteryLoRAServer


# ── Config ────────────────────────────────────────────────────

def apply_quick_mode(cfg):
    """Tiny config for sanity-checking (~1 min on A100)."""
    cfg.federated.num_clients = 10
    cfg.federated.clients_per_round = 3
    cfg.federated.num_rounds = 3
    cfg.training.local_epochs = 1
    cfg.training.batch_size = 8
    cfg.model.max_seq_length = 256
    cfg.federated.save_every_rounds = 3
    cfg.federated.max_samples_per_client = 50
    return cfg


def apply_paper_mode(cfg):
    """Paper-scale config for A100 (~1.5-2 hr per experiment).

    - 10 clients, 5/round = 50% participation (matches FLoRA setup)
    - 30 rounds is enough to see convergence differences
    - 200 samples/client x 2 epochs = 100 steps per client (~40s each)
    - Total: ~100 min per experiment
    """
    cfg.federated.num_clients = 10
    cfg.federated.clients_per_round = 5
    cfg.federated.num_rounds = 30
    cfg.training.local_epochs = 2
    cfg.training.batch_size = 4
    cfg.model.max_seq_length = 512
    cfg.federated.save_every_rounds = 10
    cfg.federated.max_samples_per_client = 200
    return cfg


# ── Experiment registry ───────────────────────────────────────
# Maps friendly name -> config factory


def _make_fixed_config(policy_type, ema_alpha, seed):
    """Create a config with the fixed BatteryLoRA improvements."""
    cfg = get_e1_main_config(seed=seed)
    cfg.rank_policy.policy_type = policy_type
    cfg.rank_policy.ema_alpha = ema_alpha
    cfg.experiment_name = f"e1_fixed_{policy_type.replace('_smoothed','_sm')}_ema{int(ema_alpha*100)}_seed{seed}"
    return apply_paper_mode(cfg)


EXPERIMENTS = {
    # E1: Main comparison (original)
    'battery_lora':   lambda seed: apply_paper_mode(get_e1_main_config(seed=seed)),
    'homolora_r8':    lambda seed: apply_paper_mode(get_baseline_homolora_config(rank=8, seed=seed)),
    'homolora_r32':   lambda seed: apply_paper_mode(get_baseline_homolora_config(rank=32, seed=seed)),
    'hetlora':        lambda seed: apply_paper_mode(get_baseline_hetlora_config(seed=seed)),
    # E2: Ablation (original)
    'ablation_binary':     lambda seed: apply_paper_mode(get_e2_ablation_config(policy='binary', seed=seed)),
    'ablation_random':     lambda seed: apply_paper_mode(get_e2_ablation_config(policy='random', seed=seed)),
    'ablation_continuous': lambda seed: apply_paper_mode(get_e2_ablation_config(policy='continuous', seed=seed)),
    # E3: Fixed BatteryLoRA variants (new)
    'fixed_threshold_smoothed':      lambda seed: _make_fixed_config('threshold_smoothed', ema_alpha=0.5, seed=seed),
    'fixed_continuous_smoothed':     lambda seed: _make_fixed_config('continuous_smoothed', ema_alpha=0.5, seed=seed),
    'fixed_threshold_ema_only':      lambda seed: _make_fixed_config('threshold', ema_alpha=0.5, seed=seed),
    'fixed_continuous_ema_only':     lambda seed: _make_fixed_config('continuous', ema_alpha=0.5, seed=seed),
}

# Recommended run order (most important first)
RUN_ORDER = [
    'battery_lora',        # Our method — run first
    'homolora_r8',         # Main baseline
    'homolora_r32',        # Quality upper bound
    'hetlora',             # Main competitor
    'ablation_binary',     # Ablation 1
    'ablation_random',     # Ablation 2
    'ablation_continuous', # Ablation 3
]


# ── Runner ────────────────────────────────────────────────────

def is_already_done(experiment_name):
    """Check if this experiment already has results (in Drive or local)."""
    for base in [DRIVE_RESULTS, RESULTS_DIR]:
        summary = os.path.join(base, experiment_name, 'summary.json')
        if os.path.exists(summary):
            return True
    return False


def run_single(cfg, skip_if_done=True):
    """Run a single experiment and save results to Drive."""

    if skip_if_done and is_already_done(cfg.experiment_name):
        print(f"\n  SKIPPING {cfg.experiment_name} — already completed (results in Drive)")
        print(f"  Delete from Drive to re-run.\n")
        return cfg.experiment_name

    print(f"\n{'='*70}")
    print(f"  STARTING: {cfg.experiment_name}")
    print(f"  Policy: {cfg.rank_policy.policy_type}")
    print(f"  Clients: {cfg.federated.num_clients}, "
          f"per round: {cfg.federated.clients_per_round}, "
          f"rounds: {cfg.federated.num_rounds}")
    print(f"  Local epochs: {cfg.training.local_epochs}, "
          f"samples/client: {cfg.federated.max_samples_per_client}")
    print(f"{'='*70}\n")

    start = time.time()
    server = BatteryLoRAServer(cfg)
    server.run_training()
    elapsed = time.time() - start

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    print(f"\nCompleted {cfg.experiment_name} in {h}h {m}m ({elapsed:.0f}s)")

    # Save to Google Drive
    results_dir = os.path.join(RESULTS_DIR, cfg.experiment_name)
    if os.path.exists('/content/drive/MyDrive'):
        os.makedirs(DRIVE_RESULTS, exist_ok=True)
        drive_dest = os.path.join(DRIVE_RESULTS, cfg.experiment_name)
        if os.path.exists(results_dir):
            shutil.copytree(results_dir, drive_dest, dirs_exist_ok=True)
            print(f"Results saved to Google Drive: {drive_dest}")
    else:
        print(f"Results saved locally: {results_dir}")
        print("WARNING: Google Drive not mounted — results may be lost if session ends!")

    # Cleanup GPU memory
    del server
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cfg.experiment_name


# ── Summary ───────────────────────────────────────────────────

def print_summary():
    """Print a comparison table of all completed experiments."""
    # Collect results from Drive first, then local
    all_results = {}
    for base in [RESULTS_DIR, DRIVE_RESULTS]:
        if not os.path.exists(base):
            continue
        for d in os.listdir(base):
            summary_path = os.path.join(base, d, 'summary.json')
            if os.path.exists(summary_path) and d not in all_results:
                all_results[d] = summary_path

    if not all_results:
        print("\nNo completed experiments found.")
        return

    print(f"\n{'='*100}")
    print(f"{'Experiment':<45} {'Loss':>8} {'Energy(Wh)':>11} {'E.Std':>8} "
          f"{'Jain':>8} {'Dropout':>8} {'Comm(MB)':>9}")
    print(f"{'-'*100}")

    for name in sorted(all_results.keys()):
        with open(all_results[name]) as f:
            s = json.load(f)
        bs = s.get('battery_summary', {})
        print(f"{name:<45} "
              f"{s.get('final_avg_loss', 0):>8.4f} "
              f"{bs.get('total_energy_wh', 0):>11.2f} "
              f"{bs.get('energy_std', 0):>8.2f} "
              f"{bs.get('jain_fairness_index', 0):>8.4f} "
              f"{bs.get('dropout_rate', 0):>7.1%} "
              f"{s.get('total_communication_mb', 0):>9.1f}")

    print(f"{'='*100}")
    print(f"\nTotal: {len(all_results)} experiments completed\n")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BatteryLoRA Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Individual experiments:
  --run battery_lora          Our method (threshold policy)
  --run homolora_r8           Baseline: all clients rank 8
  --run homolora_r32          Baseline: all clients rank 32
  --run hetlora               Baseline: static tier-based ranks
  --run ablation_binary       Ablation: battery > 50% → max, else 2
  --run ablation_random       Ablation: random rank each round
  --run ablation_continuous   Ablation: energy-budget-based
  --run all                   Run all 7 (skips already-completed)
        """)
    parser.add_argument('--quick', action='store_true',
                        help='Quick sanity test (~1 min)')
    parser.add_argument('--run', type=str,
                        help='Which experiment to run (see list above)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--summary', action='store_true',
                        help='Print results table for all completed experiments')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if results already exist')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.quick:
        print("\n=== QUICK SANITY TEST ===\n")
        cfg = apply_quick_mode(get_e1_main_config(seed=42))
        run_single(cfg, skip_if_done=False)
        print("\n=== Quick test PASSED! ===")
        return

    if not args.run:
        parser.print_help()
        print("\nAvailable experiments:", ', '.join(RUN_ORDER))
        return

    skip = not args.force

    if args.run == 'all':
        print(f"\nRunning all 7 experiments with seed={args.seed}")
        print(f"Already-completed experiments will be skipped.\n")
        total_start = time.time()
        completed = []

        for name in RUN_ORDER:
            cfg = EXPERIMENTS[name](args.seed)
            try:
                run_single(cfg, skip_if_done=skip)
                completed.append(name)
            except Exception as e:
                print(f"\nERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing to next experiment...\n")

        total = time.time() - total_start
        h = int(total // 3600)
        m = int((total % 3600) // 60)
        print(f"\nAll done! {len(completed)}/{len(RUN_ORDER)} experiments, "
              f"total time: {h}h {m}m")
        print_summary()

    elif args.run in EXPERIMENTS:
        cfg = EXPERIMENTS[args.run](args.seed)
        run_single(cfg, skip_if_done=skip)
        print_summary()

    else:
        print(f"Unknown experiment: {args.run}")
        print(f"Available: {', '.join(RUN_ORDER)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
