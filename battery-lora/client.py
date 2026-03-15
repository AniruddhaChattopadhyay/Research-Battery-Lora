"""
Flower Client for BatteryLoRA
==============================
Each virtual client represents one mobile device. During training:
1. Receives LoRA adapter weights from server (at its assigned rank)
2. Trains locally on its data partition for E epochs
3. Returns updated adapter weights

The key difference from standard federated LoRA:
- The client's LoRA rank can CHANGE each round based on battery state
- The client reports its battery state to the server
"""

import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple

from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from config import ExperimentConfig
from model_utils import load_base_model, get_data_collator
from data_utils import load_partitioned_data, formatting_prompts_func


class BatteryLoRAClient:
    """
    A federated learning client that adapts its LoRA rank based on battery.

    In the Flower simulation, each client instance is created fresh each round
    (Flower's virtual client engine manages this). So we store persistent state
    (battery, tier) externally in the BatterySimulator.
    """

    def __init__(
        self,
        client_id: int,
        cfg: ExperimentConfig,
        rank: int,
    ):
        self.client_id = client_id
        self.cfg = cfg
        self.rank = rank

    def get_local_dataset(self):
        """Load this client's data partition."""
        return load_partitioned_data(self.client_id, self.cfg.federated)

    def train(
        self,
        model,
        tokenizer,
        parameters: Dict[str, np.ndarray],
        current_round: int,
    ) -> Tuple[Dict[str, np.ndarray], int, Dict]:
        """
        Perform local training for one federated round.

        Args:
            model: PeftModel with LoRA at self.rank
            tokenizer: The tokenizer
            parameters: LoRA weights from server (at self.rank)
            current_round: Current federated round number

        Returns:
            (updated_parameters, num_samples, metrics)
        """
        # Load server parameters into model
        state_keys = list(get_peft_model_state_dict(model).keys())
        state_dict = OrderedDict()
        for key, param_array in zip(state_keys, parameters):
            state_dict[key] = torch.tensor(param_array)
        set_peft_model_state_dict(model, state_dict)

        # Load local data
        train_dataset = self.get_local_dataset()
        num_samples = len(train_dataset)

        # Cosine annealing learning rate
        lr = cosine_annealing(
            current_round,
            self.cfg.federated.num_rounds,
            self.cfg.training.learning_rate_max,
            self.cfg.training.learning_rate_min,
        )

        # Training arguments (TrainingArguments for trl<=0.28, SFTConfig for >=0.29)
        training_args = TrainingArguments(
            output_dir=f"./tmp/client_{self.client_id}",
            num_train_epochs=self.cfg.training.local_epochs,
            per_device_train_batch_size=self.cfg.training.batch_size,
            learning_rate=lr,
            weight_decay=self.cfg.training.weight_decay,
            max_grad_norm=self.cfg.training.max_grad_norm,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=True,
            bf16=torch.cuda.is_available(),
        )

        # Train — SFTTrainer handles tokenization and collation internally
        # trl<=0.28 uses `tokenizer=`, trl>=0.29 uses `processing_class=`
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            formatting_func=formatting_prompts_func,
            max_seq_length=self.cfg.model.max_seq_length,
        )

        results = trainer.train()

        # Extract updated LoRA parameters
        updated_params = [
            val.cpu().float().numpy()
            for val in get_peft_model_state_dict(model).values()
        ]

        metrics = {
            "train_loss": results.training_loss,
            "rank_used": self.rank,
            "num_samples": num_samples,
        }

        return updated_params, num_samples, metrics


def cosine_annealing(
    current_round: int,
    total_rounds: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """
    Cosine annealing learning rate schedule.
    Starts at lr_max, decays to lr_min over total_rounds.
    """
    import math
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * current_round / total_rounds)
    )
