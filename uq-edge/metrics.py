"""Calibration and uncertainty metrics. CPU only."""

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


def compute_ece(confidences: np.ndarray, is_correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Groups predictions into confidence bins, measures gap between
    average confidence and average accuracy per bin, weighted by bin size.
    """
    mask = np.isfinite(confidences)
    conf = confidences[mask]
    correct = is_correct[mask].astype(float)

    if len(conf) == 0:
        return float("nan")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = conf[in_bin].mean()
        ece += (in_bin.sum() / len(conf)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_auroc(confidences: np.ndarray, is_correct: np.ndarray) -> float:
    """AUROC for selective prediction — can the confidence separate correct from incorrect?"""
    mask = np.isfinite(confidences)
    conf = confidences[mask]
    correct = is_correct[mask].astype(int)

    if len(conf) < 2 or correct.sum() == 0 or correct.sum() == len(correct):
        return float("nan")

    try:
        return float(roc_auc_score(correct, conf))
    except ValueError:
        return float("nan")


def compute_brier(confidences: np.ndarray, is_correct: np.ndarray) -> float:
    """Brier score — mean squared error between confidence and outcome."""
    mask = np.isfinite(confidences)
    conf = np.clip(confidences[mask], 0, 1)
    correct = is_correct[mask].astype(float)

    if len(conf) == 0:
        return float("nan")

    return float(brier_score_loss(correct, conf))


def compute_reliability_diagram(
    confidences: np.ndarray, is_correct: np.ndarray, n_bins: int = 10,
) -> dict:
    """Compute data for a reliability diagram.

    Returns dict with bin_centers, bin_accuracies, bin_confidences, bin_counts.
    """
    mask = np.isfinite(confidences)
    conf = confidences[mask]
    correct = is_correct[mask].astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    accuracies = []
    mean_confs = []
    counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        count = in_bin.sum()
        counts.append(int(count))
        centers.append((lo + hi) / 2)
        if count > 0:
            accuracies.append(float(correct[in_bin].mean()))
            mean_confs.append(float(conf[in_bin].mean()))
        else:
            accuracies.append(float("nan"))
            mean_confs.append(float("nan"))

    return {
        "bin_centers": centers,
        "bin_accuracies": accuracies,
        "bin_confidences": mean_confs,
        "bin_counts": counts,
    }


def compute_all_metrics(
    scores: dict[str, np.ndarray],
    is_correct: np.ndarray,
) -> dict:
    """Compute all metrics for all UQ methods.

    Returns nested dict: {method: {ece, auroc, brier, reliability}}.
    """
    results = {}
    for method_name, conf in scores.items():
        if method_name.startswith("_"):
            continue  # skip internal keys like _temperature
        results[method_name] = {
            "ece": compute_ece(conf, is_correct),
            "auroc": compute_auroc(conf, is_correct),
            "brier": compute_brier(conf, is_correct),
            "reliability": compute_reliability_diagram(conf, is_correct),
        }
    return results


def compute_deltas(metrics: dict, baseline_metrics: dict) -> dict:
    """Compute metric deltas relative to a baseline (e.g., FP16)."""
    deltas = {}
    for method in metrics:
        if method not in baseline_metrics:
            continue
        deltas[method] = {}
        for metric_name in ("ece", "auroc", "brier"):
            base_val = baseline_metrics[method].get(metric_name)
            cur_val = metrics[method].get(metric_name)
            if base_val is not None and cur_val is not None:
                if not (np.isnan(base_val) or np.isnan(cur_val)):
                    deltas[method][f"{metric_name}_delta"] = cur_val - base_val
    return deltas
