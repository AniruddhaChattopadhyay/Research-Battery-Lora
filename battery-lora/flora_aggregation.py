"""
FLoRA Stacking Aggregation
===========================
Implements the FLoRA (NeurIPS 2024) aggregation method for combining
LoRA adapters of different ranks.

The Problem:
  Client 1 returns rank-32 adapter (A: 2048x32, B: 32x2048)
  Client 2 returns rank-8  adapter (A: 2048x8,  B: 8x2048)
  Client 3 returns rank-2  adapter (A: 2048x2,  B: 2x2048)

  How do we average these? They have different shapes!

The FLoRA Solution:
  Treat each client's adapter as updating a SUBSPACE of a larger global adapter.
  - Client 2's rank-8 adapter updates the first 8 dimensions of the rank-32 global.
  - Client 3's rank-2 adapter updates the first 2 dimensions.
  - Dimensions updated by multiple clients are averaged.
  - Dimensions updated by no one keep their current global values.

This is mathematically correct (no noise injection) unlike zero-padding.

Reference: Wang et al., "FLoRA: Federated Fine-Tuning Large Language Models
with Heterogeneous Low-Rank Adaptations," NeurIPS 2024.
"""

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple


def aggregate_flora(
    global_state: Dict[str, np.ndarray],
    client_updates: List[Tuple[Dict[str, np.ndarray], int, int]],
    max_rank: int,
    ema_alpha: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Aggregate heterogeneous-rank LoRA adapters using FLoRA stacking.

    Args:
        global_state: Current global LoRA state dict (at max_rank).
                      Keys are like "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        client_updates: List of (state_dict, num_samples, client_rank) tuples.
                       Each state_dict has the same keys but potentially smaller tensors.
        max_rank: The global maximum rank (e.g., 32)
        ema_alpha: EMA blending factor. 0.0 = no EMA (original behavior),
                   0.5 = blend 50% old + 50% new. Prevents any single round
                   from drastically changing the global adapter.

    Returns:
        Updated global state dict with aggregated parameters.
    """
    if not client_updates:
        return global_state

    new_state = {}

    for key in global_state.keys():
        global_param = global_state[key]

        # Determine if this is a LoRA A or B matrix based on shape
        # LoRA A: (rank, in_features) — rank is the first dimension
        # LoRA B: (out_features, rank) — rank is the second dimension
        is_lora_a = "lora_A" in key
        is_lora_b = "lora_B" in key

        if not (is_lora_a or is_lora_b):
            # Not a LoRA parameter — just average normally
            weighted_sum = np.zeros_like(global_param)
            total_samples = 0
            for client_state, num_samples, rank in client_updates:
                if key in client_state:
                    weighted_sum += client_state[key] * num_samples
                    total_samples += num_samples
            if total_samples > 0:
                new_state[key] = weighted_sum / total_samples
            else:
                new_state[key] = global_param
            continue

        # ── FLoRA stacking for LoRA A and B matrices ──

        # Initialize accumulator and count at global shape
        accumulator = np.zeros_like(global_param, dtype=np.float64)
        weight_count = np.zeros_like(global_param, dtype=np.float64)

        for client_state, num_samples, client_rank in client_updates:
            if key not in client_state:
                continue

            client_param = client_state[key]

            if is_lora_a:
                # LoRA A shape: (rank, in_features)
                # Client has shape (client_rank, in_features)
                # Maps to first client_rank rows of global
                r = min(client_rank, client_param.shape[0])
                accumulator[:r, :] += client_param[:r, :] * num_samples
                weight_count[:r, :] += num_samples

            elif is_lora_b:
                # LoRA B shape: (out_features, rank)
                # Client has shape (out_features, client_rank)
                # Maps to first client_rank columns of global
                r = min(client_rank, client_param.shape[1])
                accumulator[:, :r] += client_param[:, :r] * num_samples
                weight_count[:, :r] += num_samples

        # Average where we have contributions, keep global where we don't
        mask = weight_count > 0
        result = np.copy(global_param).astype(np.float64)
        result[mask] = accumulator[mask] / weight_count[mask]

        # EMA blending: preserve some of the old global state to prevent
        # any single round from drastically overwriting learned parameters
        if ema_alpha > 0:
            result = ema_alpha * global_param.astype(np.float64) + (1 - ema_alpha) * result

        new_state[key] = result.astype(global_param.dtype)

    return new_state


def extract_sub_adapter(
    global_state: Dict[str, np.ndarray],
    target_rank: int,
) -> Dict[str, np.ndarray]:
    """
    Extract a sub-adapter of target_rank from the global adapter.

    This is what the server sends to a client that needs a lower rank.
    We take the first target_rank rows/columns of the A/B matrices.

    Args:
        global_state: Global LoRA state dict at max_rank
        target_rank: The rank the client needs

    Returns:
        State dict with smaller A/B matrices
    """
    sub_state = {}

    for key, param in global_state.items():
        if "lora_A" in key:
            # A: (rank, in_features) → take first target_rank rows
            sub_state[key] = param[:target_rank, :]
        elif "lora_B" in key:
            # B: (out_features, rank) → take first target_rank columns
            sub_state[key] = param[:, :target_rank]
        else:
            # Non-LoRA params: copy as-is
            sub_state[key] = param.copy()

    return sub_state


def compute_communication_cost(
    state_dict: Dict[str, np.ndarray],
) -> int:
    """
    Compute the total bytes that would be transmitted for a state dict.

    Args:
        state_dict: The LoRA parameters being sent

    Returns:
        Total bytes (assuming float16 transmission)
    """
    total_bytes = 0
    for param in state_dict.values():
        total_bytes += param.size * 2  # float16 = 2 bytes
    return total_bytes
