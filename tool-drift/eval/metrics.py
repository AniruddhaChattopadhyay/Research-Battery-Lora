from __future__ import annotations

import random
from statistics import mean
from typing import Any, Iterable


def accuracy(items: Iterable[bool]) -> float:
    values = list(items)
    return mean(values) if values else 0.0


def exact_match_rate(predictions: Iterable[Any], references: Iterable[Any]) -> float:
    preds = list(predictions)
    refs = list(references)
    if not preds:
        return 0.0
    return sum(pred == ref for pred, ref in zip(preds, refs)) / len(preds)


def recovery_rate(original: float, drifted: float, repaired: float) -> float:
    gap = original - drifted
    if gap <= 0:
        return 0.0
    return max(0.0, min(1.0, (repaired - drifted) / gap))


def bootstrap_ci(values: list[float], samples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    draws = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        draws.append(mean(draw))
    draws.sort()
    lower = draws[int((alpha / 2) * len(draws))]
    upper = draws[int((1 - alpha / 2) * len(draws)) - 1]
    return lower, upper


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": mean(values), "min": min(values), "max": max(values)}

