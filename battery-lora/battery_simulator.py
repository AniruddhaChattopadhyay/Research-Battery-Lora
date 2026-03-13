"""
Battery Simulator
=================
Simulates battery drain for each virtual client during federated training.

Each client has:
- A device tier (high / mid / low)
- A battery level (0-100%)
- A charging state (True/False)
- Energy costs that vary by LoRA rank

The simulator tracks battery over federated rounds and determines
when devices drop out due to low battery.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import BatteryConfig, DeviceTierConfig


@dataclass
class DeviceState:
    """State of a single simulated device."""
    client_id: int
    tier: str  # "high", "mid", "low"
    battery_percent: float  # 0.0 - 100.0
    is_charging: bool
    is_active: bool = True  # False = dropped out
    # Tracking
    total_energy_consumed_wh: float = 0.0
    rounds_participated: int = 0
    rounds_skipped: int = 0
    battery_history: List[float] = field(default_factory=list)
    rank_history: List[int] = field(default_factory=list)


class BatterySimulator:
    """
    Simulates battery behavior for a fleet of mobile devices
    participating in federated learning.

    Usage:
        sim = BatterySimulator(num_clients=50, battery_cfg=..., tier_cfg=...)
        for round in range(100):
            for client_id in selected_clients:
                state = sim.get_device_state(client_id)
                rank = policy.get_rank(state)
                # ... train with this rank ...
                sim.update_after_training(client_id, rank)
    """

    def __init__(
        self,
        num_clients: int,
        battery_cfg: BatteryConfig,
        tier_cfg: DeviceTierConfig,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        self.battery_cfg = battery_cfg
        self.tier_cfg = tier_cfg
        self.rng = random.Random(seed)

        # Initialize all devices
        self.devices: Dict[int, DeviceState] = {}
        self._initialize_devices()

    def _initialize_devices(self):
        """Create initial device states with random battery levels and tiers."""
        # Assign tiers based on distribution
        tiers = []
        for tier, fraction in self.tier_cfg.tier_distribution.items():
            count = int(self.num_clients * fraction)
            tiers.extend([tier] * count)
        # Fill remaining due to rounding
        while len(tiers) < self.num_clients:
            tiers.append("mid")
        self.rng.shuffle(tiers)

        for i in range(self.num_clients):
            # Random initial battery between 20% and 100%
            battery = self.rng.uniform(20.0, 100.0)
            # 30% of devices start charging
            is_charging = self.rng.random() < 0.30

            self.devices[i] = DeviceState(
                client_id=i,
                tier=tiers[i],
                battery_percent=battery,
                is_charging=is_charging,
                battery_history=[battery],
            )

    def get_device_state(self, client_id: int) -> DeviceState:
        """Get current state of a device."""
        return self.devices[client_id]

    def get_active_clients(self) -> List[int]:
        """Return IDs of all devices that haven't dropped out."""
        return [
            cid for cid, dev in self.devices.items()
            if dev.is_active
        ]

    def can_participate(self, client_id: int) -> bool:
        """Check if a device has enough battery to participate."""
        dev = self.devices[client_id]
        if not dev.is_active:
            return False
        if dev.is_charging:
            return True
        return dev.battery_percent > self.battery_cfg.reserve_percent

    def update_after_training(self, client_id: int, rank_used: int):
        """
        Update device state after one round of training.

        Args:
            client_id: Which device
            rank_used: The LoRA rank that was used (determines energy cost)
        """
        dev = self.devices[client_id]

        # Energy consumed this round
        energy_wh = self.battery_cfg.energy_per_round.get(rank_used, 0.2)

        # Update total energy tracking
        dev.total_energy_consumed_wh += energy_wh

        # Update battery
        if dev.is_charging:
            # Charging devices gain battery (net effect of charging - training)
            # Typical charger: ~10W, training: ~3W, so net +7W
            charge_rate_wh = 0.3  # Net gain per round while charging
            dev.battery_percent = min(
                100.0,
                dev.battery_percent + (charge_rate_wh / self.battery_cfg.capacity_wh) * 100
            )
        else:
            # Drain battery
            drain_percent = (energy_wh / self.battery_cfg.capacity_wh) * 100
            dev.battery_percent = max(0.0, dev.battery_percent - drain_percent)

        # Check for dropout
        if dev.battery_percent <= self.battery_cfg.dropout_percent and not dev.is_charging:
            dev.is_active = False

        # Record history
        dev.rounds_participated += 1
        dev.battery_history.append(dev.battery_percent)
        dev.rank_history.append(rank_used)

    def update_idle_round(self, client_id: int):
        """Update device state for a round where it was not selected."""
        dev = self.devices[client_id]

        # Idle drain (background processes)
        if not dev.is_charging:
            idle_drain = 0.01  # Very small idle drain
            drain_percent = (idle_drain / self.battery_cfg.capacity_wh) * 100
            dev.battery_percent = max(0.0, dev.battery_percent - drain_percent)
        else:
            # Charging while idle — faster charging
            charge_rate_wh = 0.5
            dev.battery_percent = min(
                100.0,
                dev.battery_percent + (charge_rate_wh / self.battery_cfg.capacity_wh) * 100
            )

        dev.battery_history.append(dev.battery_percent)
        dev.rounds_skipped += 1

    def simulate_environment_changes(self, round_num: int):
        """
        Simulate realistic environment changes each round.
        - Some devices start/stop charging
        - Battery levels drift naturally
        """
        for dev in self.devices.values():
            if not dev.is_active:
                continue

            # 5% chance of charging state change each round
            if self.rng.random() < 0.05:
                dev.is_charging = not dev.is_charging

    def get_summary_stats(self) -> Dict:
        """Get aggregate statistics across all devices."""
        active = [d for d in self.devices.values() if d.is_active]
        dropped = [d for d in self.devices.values() if not d.is_active]
        all_devs = list(self.devices.values())

        energies = [d.total_energy_consumed_wh for d in all_devs]
        batteries = [d.battery_percent for d in all_devs]

        # Jain's fairness index for energy consumption
        if sum(energies) > 0:
            n = len(energies)
            jain = (sum(energies) ** 2) / (n * sum(e ** 2 for e in energies))
        else:
            jain = 1.0

        return {
            "active_clients": len(active),
            "dropped_clients": len(dropped),
            "dropout_rate": len(dropped) / len(all_devs),
            "avg_battery": sum(batteries) / len(batteries),
            "min_battery": min(batteries),
            "total_energy_wh": sum(energies),
            "avg_energy_per_client": sum(energies) / len(all_devs),
            "energy_std": (
                sum((e - sum(energies) / len(energies)) ** 2 for e in energies)
                / len(energies)
            ) ** 0.5,
            "jain_fairness_index": jain,
            "avg_rounds_participated": sum(
                d.rounds_participated for d in all_devs
            ) / len(all_devs),
        }
