"""Compute UQ scores from saved raw outputs. CPU only — no model needed."""

import numpy as np
from scipy.optimize import minimize_scalar


def scores_msp(data: dict) -> np.ndarray:
    """Mean Softmax Probability — higher = more confident."""
    return data["max_probs"]


def scores_entropy(data: dict) -> np.ndarray:
    """Predictive entropy — normalized to [0, 1] where higher = more confident.

    Raw entropy is non-negative (0 = certain, large = uncertain).
    We normalize via: confidence = 1 - (entropy / max_entropy), clamped to [0, 1].
    """
    raw = data["mean_entropies"]
    if len(raw) == 0:
        return raw
    max_ent = np.max(raw) if np.max(raw) > 0 else 1.0
    normalized = 1.0 - (raw / max_ent)
    return np.clip(normalized, 0.0, 1.0)


def scores_log_likelihood(data: dict) -> np.ndarray:
    """Mean token log-likelihood — normalized to [0, 1] where higher = more confident.

    Raw log-likelihood is negative (0 = certain, large negative = uncertain).
    We normalize via: confidence = 1 - (|ll| / max_|ll|), clamped to [0, 1].
    """
    raw = data["mean_log_probs"]
    if len(raw) == 0:
        return raw
    abs_ll = np.abs(raw)
    max_abs = np.max(abs_ll) if np.max(abs_ll) > 0 else 1.0
    normalized = 1.0 - (abs_ll / max_abs)
    return np.clip(normalized, 0.0, 1.0)


def scores_verbalized(data: dict) -> np.ndarray:
    """Verbalized confidence — already in [0, 1], higher = more confident."""
    return data.get("verbalized_conf", np.full(len(data["is_correct"]), np.nan))


def scores_self_consistency(data: dict) -> np.ndarray:
    """Self-consistency agreement — already in [0, 1], higher = more confident."""
    return data.get("self_consistency", np.full(len(data["is_correct"]), np.nan))


def fit_temperature(confidences: np.ndarray, is_correct: np.ndarray, cal_fraction: float = 0.2) -> float:
    """Fit temperature scaling on a calibration split.

    Minimizes negative log-likelihood on the calibration set.
    Returns the optimal temperature T.
    """
    n = len(confidences)
    cal_n = max(int(n * cal_fraction), 10)
    cal_conf = confidences[:cal_n]
    cal_correct = is_correct[:cal_n].astype(float)

    def nll(T):
        # Scale confidences: new_conf = conf^(1/T) / (conf^(1/T) + (1-conf)^(1/T))
        # Simpler: treat conf as a probability, apply temp scaling in log space
        eps = 1e-8
        scaled = np.clip(cal_conf, eps, 1 - eps)
        log_odds = np.log(scaled / (1 - scaled)) / T
        p = 1 / (1 + np.exp(-log_odds))
        # NLL
        loss = -np.mean(
            cal_correct * np.log(p + eps) + (1 - cal_correct) * np.log(1 - p + eps)
        )
        return loss

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x


def apply_temperature(confidences: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to confidence values."""
    eps = 1e-8
    scaled = np.clip(confidences, eps, 1 - eps)
    log_odds = np.log(scaled / (1 - scaled)) / T
    return 1 / (1 + np.exp(-log_odds))


def compute_all_scores(data: dict) -> dict[str, np.ndarray]:
    """Compute all UQ scores from raw output data.

    Returns dict mapping method name -> confidence array (higher = more confident).
    """
    is_correct = data["is_correct"]

    scores = {
        "msp": scores_msp(data),
        "entropy": scores_entropy(data),
        "log_likelihood": scores_log_likelihood(data),
    }

    # Verbalized confidence (may have NaNs)
    verbal = scores_verbalized(data)
    if not np.all(np.isnan(verbal)):
        scores["verbalized"] = verbal

    # Self-consistency (may not be present)
    sc = scores_self_consistency(data)
    if not np.all(np.isnan(sc)):
        scores["self_consistency"] = sc

    # Temperature-scaled MSP
    msp = scores["msp"]
    valid_mask = np.isfinite(msp) & (msp > 0) & (msp < 1)
    if valid_mask.sum() > 20:
        T = fit_temperature(msp[valid_mask], is_correct[valid_mask])
        scores["temp_scaled_msp"] = apply_temperature(msp, T)
        scores["_temperature"] = np.array([T])  # save for reference
    else:
        scores["temp_scaled_msp"] = msp.copy()

    return scores
