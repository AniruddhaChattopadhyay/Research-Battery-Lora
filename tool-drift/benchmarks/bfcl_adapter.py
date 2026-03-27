from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.common import load_json


def load_bfcl_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load a small BFCL subset from a pre-exported JSON file.

    Expected format:
    [
      {
        "id": "...",
        "prompt": "...",
        "tool": {...},
        "gold_call": {"name": "...", "arguments": {...}}
      }
    ]
    """
    data_cfg = config.get("data", {})
    subset_path = data_cfg.get("bfcl_subset_path")
    if not subset_path:
        raise NotImplementedError(
            "Real BFCL mode is not wired yet. Set data.bfcl_subset_path to a "
            "pre-exported JSON subset or run the script with --demo."
        )

    path = Path(subset_path)
    if not path.exists():
        raise FileNotFoundError(f"BFCL subset file not found: {path}")

    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("BFCL subset JSON must be a list of task objects")
    return payload
