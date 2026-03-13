"""
Rank Policy Module
==================
Maps battery state + device tier → LoRA rank.

This is the core novel contribution of BatteryLoRA.
We implement multiple policies for ablation studies.
"""

import random
from abc import ABC, abstractmethod
from typing import List

from battery_simulator import DeviceState
from config import DeviceTierConfig, RankPolicyConfig


class RankPolicy(ABC):
    """Base class for rank selection policies."""

    @abstractmethod
    def get_rank(self, device: DeviceState) -> int:
        """Return the LoRA rank this device should use."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class BatteryThresholdPolicy(RankPolicy):
    """
    Our main method: threshold-based battery-to-rank mapping.

    Battery Level    High Tier    Mid Tier    Low Tier
    > 80%            32           16          8
    60-80%           16           8           4
    40-60%           8            4           4
    20-40%           4            2           2
    < 20%            2            2           2
    Charging         max_tier     max_tier    max_tier
    """

    RANK_TABLE = {
        # (tier, battery_band) -> rank
        ("high", 4): 32, ("mid", 4): 16, ("low", 4): 8,   # > 80%
        ("high", 3): 16, ("mid", 3): 8,  ("low", 3): 4,   # 60-80%
        ("high", 2): 8,  ("mid", 2): 4,  ("low", 2): 4,   # 40-60%
        ("high", 1): 4,  ("mid", 1): 2,  ("low", 1): 2,   # 20-40%
        ("high", 0): 2,  ("mid", 0): 2,  ("low", 0): 2,   # < 20%
    }

    def __init__(self, tier_cfg: DeviceTierConfig):
        self.tier_cfg = tier_cfg

    @property
    def name(self) -> str:
        return "threshold"

    def _get_battery_band(self, battery_percent: float) -> int:
        """Convert battery percentage to a band (0-4)."""
        if battery_percent > 80:
            return 4
        elif battery_percent > 60:
            return 3
        elif battery_percent > 40:
            return 2
        elif battery_percent > 20:
            return 1
        else:
            return 0

    def get_rank(self, device: DeviceState) -> int:
        if device.is_charging:
            return self.tier_cfg.tier_max_rank[device.tier]

        band = self._get_battery_band(device.battery_percent)
        return self.RANK_TABLE[(device.tier, band)]


class ContinuousPolicy(RankPolicy):
    """
    Energy-budget-based continuous rank selection.

    Estimates remaining available energy, then picks the highest rank
    that still allows at least `min_future_rounds` more rounds.
    """

    def __init__(
        self,
        tier_cfg: DeviceTierConfig,
        battery_capacity_wh: float = 17.1,
        reserve_percent: float = 15.0,
        min_future_rounds: int = 5,
        energy_per_round: dict = None,
    ):
        self.tier_cfg = tier_cfg
        self.battery_capacity_wh = battery_capacity_wh
        self.reserve_percent = reserve_percent
        self.min_future_rounds = min_future_rounds
        self.energy_per_round = energy_per_round or {
            2: 0.08, 4: 0.12, 8: 0.20, 16: 0.35, 32: 0.60
        }

    @property
    def name(self) -> str:
        return "continuous"

    def get_rank(self, device: DeviceState) -> int:
        if device.is_charging:
            return self.tier_cfg.tier_max_rank[device.tier]

        # Calculate available energy
        remaining_wh = self.battery_capacity_wh * (device.battery_percent / 100)
        reserve_wh = self.battery_capacity_wh * (self.reserve_percent / 100)
        available_wh = max(0, remaining_wh - reserve_wh)

        # Hardware limit for this tier
        max_rank = self.tier_cfg.tier_max_rank[device.tier]

        # Pick highest affordable rank
        for rank in [32, 16, 8, 4, 2]:
            if rank > max_rank:
                continue
            cost = self.energy_per_round[rank]
            if available_wh / cost >= self.min_future_rounds:
                return rank

        return 2  # Minimum rank as fallback


class BinaryPolicy(RankPolicy):
    """
    Simple binary policy for ablation.
    Battery > 50% → max rank for tier
    Battery <= 50% → rank 2
    """

    def __init__(self, tier_cfg: DeviceTierConfig):
        self.tier_cfg = tier_cfg

    @property
    def name(self) -> str:
        return "binary"

    def get_rank(self, device: DeviceState) -> int:
        if device.is_charging or device.battery_percent > 50:
            return self.tier_cfg.tier_max_rank[device.tier]
        return 2


class FixedRankPolicy(RankPolicy):
    """
    Baseline: all clients use the same fixed rank regardless of battery.
    This is what standard FedAvg + HomLoRA does.
    """

    def __init__(self, rank: int = 8):
        self.rank = rank

    @property
    def name(self) -> str:
        return f"fixed_r{self.rank}"

    def get_rank(self, device: DeviceState) -> int:
        return self.rank


class StaticTierPolicy(RankPolicy):
    """
    Baseline (HetLoRA-style): rank based on device tier only, ignores battery.
    """

    def __init__(self, tier_cfg: DeviceTierConfig):
        self.tier_cfg = tier_cfg

    @property
    def name(self) -> str:
        return "static_tier"

    def get_rank(self, device: DeviceState) -> int:
        return self.tier_cfg.tier_max_rank[device.tier]


class RandomPolicy(RankPolicy):
    """
    Ablation baseline: random rank each round.
    Controls for whether rank diversity alone (without battery awareness) helps.
    """

    def __init__(
        self,
        available_ranks: List[int] = None,
        tier_cfg: DeviceTierConfig = None,
        seed: int = 42,
    ):
        self.available_ranks = available_ranks or [2, 4, 8, 16, 32]
        self.tier_cfg = tier_cfg
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def get_rank(self, device: DeviceState) -> int:
        max_rank = 32
        if self.tier_cfg:
            max_rank = self.tier_cfg.tier_max_rank[device.tier]
        valid = [r for r in self.available_ranks if r <= max_rank]
        return self.rng.choice(valid)


# ─── Factory ──────────────────────────────────────────────────────────────

def create_rank_policy(
    policy_cfg: RankPolicyConfig,
    tier_cfg: DeviceTierConfig,
    battery_capacity_wh: float = 17.1,
    reserve_percent: float = 15.0,
    energy_per_round: dict = None,
    seed: int = 42,
) -> RankPolicy:
    """Create a rank policy from config."""
    if policy_cfg.policy_type == "threshold":
        return BatteryThresholdPolicy(tier_cfg)
    elif policy_cfg.policy_type == "continuous":
        return ContinuousPolicy(
            tier_cfg,
            battery_capacity_wh=battery_capacity_wh,
            reserve_percent=reserve_percent,
            min_future_rounds=policy_cfg.min_future_rounds,
            energy_per_round=energy_per_round,
        )
    elif policy_cfg.policy_type == "binary":
        return BinaryPolicy(tier_cfg)
    elif policy_cfg.policy_type == "fixed":
        return FixedRankPolicy(rank=policy_cfg.fixed_rank)
    elif policy_cfg.policy_type == "static_tier":
        return StaticTierPolicy(tier_cfg)
    elif policy_cfg.policy_type == "random":
        return RandomPolicy(tier_cfg=tier_cfg, seed=seed)
    else:
        raise ValueError(f"Unknown policy type: {policy_cfg.policy_type}")
