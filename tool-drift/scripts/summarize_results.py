from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CURRENT = Path(__file__).resolve().parents[1]
import sys

if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))

from scripts.common import load_json


def summarize_file(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    summary = payload.get("summary", {})
    return {
        "file": str(path),
        "benchmark": summary.get("benchmark", "unknown"),
        "sample_count": summary.get("sample_count", 0),
        "original_score": summary.get("original_score", 0.0),
        "drifted_score": summary.get("drifted_score", 0.0),
        "repaired_score": summary.get("repaired_score", 0.0),
        "recovery_rate": summary.get("recovery_rate", 0.0),
        "error_breakdown": summary.get("error_breakdown", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    files = sorted(results_dir.rglob("*_results.json"))
    summary = [summarize_file(path) for path in files]
    text = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

