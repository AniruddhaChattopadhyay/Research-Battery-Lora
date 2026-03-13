"""
Evaluation Module
=================
Evaluates the trained model on benchmarks and computes all metrics
needed for the paper.
"""

import json
import os
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

from config import ExperimentConfig


def load_model_from_checkpoint(
    cfg: ExperimentConfig,
    checkpoint_dir: str,
) -> PeftModel:
    """Load model with saved LoRA adapter from a checkpoint."""
    from model_utils import load_base_model, apply_lora

    model, tokenizer = load_base_model(cfg.model)
    peft_model = apply_lora(model, cfg.lora, cfg.lora.max_rank)

    # Load saved LoRA weights
    checkpoint = np.load(os.path.join(checkpoint_dir, "global_lora.npz"))
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    state_dict = OrderedDict()
    for key in get_peft_model_state_dict(peft_model).keys():
        if key in checkpoint:
            state_dict[key] = torch.tensor(checkpoint[key])

    set_peft_model_state_dict(peft_model, state_dict)
    return peft_model, tokenizer


def evaluate_generation(
    model,
    tokenizer,
    eval_examples: List[Dict],
    max_new_tokens: int = 128,
) -> Dict:
    """
    Evaluate model on instruction-following generation.
    Returns ROUGE-L scores.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Install rouge-score: pip install rouge-score")
        return {"rouge_l": 0.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    model.eval()

    scores = []
    for example in eval_examples:
        # Format prompt (without the response)
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        reference = example.get("output", "")

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Score
        score = scorer.score(reference, generated)
        scores.append(score["rougeL"].fmeasure)

    return {
        "rouge_l_mean": np.mean(scores),
        "rouge_l_std": np.std(scores),
        "rouge_l_median": np.median(scores),
        "num_evaluated": len(scores),
    }


def evaluate_perplexity(
    model,
    tokenizer,
    eval_texts: List[str],
    max_length: int = 512,
) -> Dict:
    """
    Evaluate model perplexity on a set of texts.
    Lower perplexity = better model.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in eval_texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
    }


def compute_efficiency_metrics(results_dir: str) -> Dict:
    """
    Compute efficiency metrics from saved experiment results.

    Returns energy, communication, fairness, and dropout metrics.
    """
    # Load saved data
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)

    with open(os.path.join(results_dir, "battery_stats.json")) as f:
        battery_stats = json.load(f)

    with open(os.path.join(results_dir, "device_stats.json")) as f:
        device_stats = json.load(f)

    with open(os.path.join(results_dir, "round_metrics.json")) as f:
        round_metrics = json.load(f)

    # Energy metrics
    energies = [d["total_energy_wh"] for d in device_stats.values()]
    n = len(energies)

    # Jain's fairness index
    if sum(energies) > 0:
        jain = (sum(energies) ** 2) / (n * sum(e ** 2 for e in energies))
    else:
        jain = 1.0

    # Gini coefficient
    sorted_e = sorted(energies)
    cumulative = np.cumsum(sorted_e)
    gini = (
        (2 * sum((i + 1) * e for i, e in enumerate(sorted_e)))
        / (n * sum(sorted_e))
        - (n + 1) / n
    ) if sum(sorted_e) > 0 else 0.0

    # Convergence speed (rounds to reach loss threshold)
    losses = [r["avg_train_loss"] for r in round_metrics if r["avg_train_loss"] < float("inf")]
    if losses:
        target_loss = losses[-1] * 1.1  # 10% above final loss
        rounds_to_target = next(
            (i + 1 for i, l in enumerate(losses) if l <= target_loss),
            len(losses),
        )
    else:
        rounds_to_target = -1

    # Rank distribution over time
    rank_distribution = {}
    for rm in round_metrics:
        for rank, count in rm.get("ranks_used", {}).items():
            rank = str(rank)
            rank_distribution[rank] = rank_distribution.get(rank, 0) + count

    return {
        "energy": {
            "total_wh": sum(energies),
            "mean_per_client_wh": np.mean(energies),
            "std_wh": np.std(energies),
            "min_wh": min(energies),
            "max_wh": max(energies),
        },
        "fairness": {
            "jain_index": jain,
            "gini_coefficient": gini,
        },
        "dropout": {
            "total_dropouts": sum(1 for d in device_stats.values() if not d["is_active"]),
            "dropout_rate": sum(1 for d in device_stats.values() if not d["is_active"]) / n,
        },
        "communication": {
            "total_mb": summary.get("total_communication_mb", 0),
        },
        "convergence": {
            "rounds_to_target": rounds_to_target,
            "final_loss": losses[-1] if losses else None,
        },
        "rank_distribution": rank_distribution,
    }


def compare_experiments(experiment_dirs: List[str]) -> Dict:
    """
    Compare metrics across multiple experiments (e.g., our method vs baselines).

    Args:
        experiment_dirs: List of result directory paths

    Returns:
        Comparison table data
    """
    comparison = {}

    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)

        try:
            metrics = compute_efficiency_metrics(exp_dir)
            with open(os.path.join(exp_dir, "summary.json")) as f:
                summary = json.load(f)

            comparison[exp_name] = {
                "final_loss": metrics["convergence"]["final_loss"],
                "total_energy_wh": metrics["energy"]["total_wh"],
                "energy_std_wh": metrics["energy"]["std_wh"],
                "jain_fairness": metrics["fairness"]["jain_index"],
                "dropout_rate": metrics["dropout"]["dropout_rate"],
                "communication_mb": metrics["communication"]["total_mb"],
                "rounds_to_converge": metrics["convergence"]["rounds_to_target"],
            }
        except FileNotFoundError as e:
            print(f"Skipping {exp_name}: {e}")

    return comparison
