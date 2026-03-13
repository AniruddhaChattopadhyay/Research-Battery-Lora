"""
Data Utilities
==============
Handles loading and partitioning the dataset for federated learning.

Key concept: In FL, each client has its own local dataset. We simulate
this by splitting a global dataset into N partitions.

Non-IID splits: In reality, each person's phone has different kinds of text.
We simulate this using Dirichlet distribution — lower alpha = more different
data across phones.
"""

from typing import Callable, Tuple

from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

from config import FederatedConfig


# Module-level cache to avoid re-downloading
_FDS_CACHE = {}


def format_alpaca_prompt(example: dict) -> str:
    """
    Format a single example in the Alpaca instruction-following template.

    The Alpaca format looks like:
        ### Instruction:
        <the task>

        ### Input:
        <optional context>

        ### Response:
        <the answer>
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )


def formatting_prompts_func(example: dict) -> str:
    """
    Formatting function for SFTTrainer.
    Called per-example (trl >= 0.29 calls this unbatched).
    """
    return format_alpaca_prompt(example)


def load_partitioned_data(
    partition_id: int,
    fed_cfg: FederatedConfig,
) -> Dataset:
    """
    Load a single client's partition of the dataset.

    Uses Dirichlet partitioning for non-IID splits. The alpha parameter
    controls how different each client's data is:
    - alpha=0.1  → very non-IID (each client has mostly one type of data)
    - alpha=0.5  → moderately non-IID (our default)
    - alpha=1.0  → mildly non-IID
    - alpha=100  → nearly IID (everyone has similar data)

    Args:
        partition_id: Which client's data to load (0 to num_clients-1)
        fed_cfg: Federated learning configuration

    Returns:
        HuggingFace Dataset for this partition
    """
    cache_key = (fed_cfg.dataset_name, fed_cfg.num_clients, fed_cfg.dirichlet_alpha)

    if cache_key not in _FDS_CACHE:
        # For Alpaca, we need a label column for Dirichlet partitioning.
        # Since it's an instruction dataset without explicit labels,
        # we use IID partitioning (standard in federated LoRA papers).
        # For classification tasks, we'd use DirichletPartitioner.
        partitioner = IidPartitioner(num_partitions=fed_cfg.num_clients)

        fds = FederatedDataset(
            dataset=fed_cfg.dataset_name,
            partitioners={"train": partitioner},
        )
        _FDS_CACHE[cache_key] = fds

    fds = _FDS_CACHE[cache_key]
    partition = fds.load_partition(partition_id, "train")

    # Limit samples per client if configured (for quick testing)
    if fed_cfg.max_samples_per_client > 0 and len(partition) > fed_cfg.max_samples_per_client:
        partition = partition.select(range(fed_cfg.max_samples_per_client))

    return partition


def load_eval_dataset(fed_cfg: FederatedConfig) -> Dataset:
    """
    Load a small evaluation dataset for server-side evaluation.

    We hold out a small portion of Alpaca for testing.
    """
    from datasets import load_dataset

    dataset = load_dataset(fed_cfg.dataset_name, split="train")
    # Use last 500 examples as eval set
    eval_dataset = dataset.select(range(len(dataset) - 500, len(dataset)))
    return eval_dataset


def get_dataset_stats(fed_cfg: FederatedConfig) -> dict:
    """Print stats about the dataset and partitioning."""
    from datasets import load_dataset

    dataset = load_dataset(fed_cfg.dataset_name, split="train")
    total_examples = len(dataset)
    avg_per_client = total_examples // fed_cfg.num_clients

    return {
        "dataset_name": fed_cfg.dataset_name,
        "total_examples": total_examples,
        "num_clients": fed_cfg.num_clients,
        "avg_examples_per_client": avg_per_client,
        "dirichlet_alpha": fed_cfg.dirichlet_alpha,
    }
