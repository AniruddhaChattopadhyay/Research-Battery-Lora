"""
Model Utilities
===============
Handles loading TinyLlama with LoRA adapters at different ranks.

Key concept: The base model (TinyLlama 1.1B) is loaded ONCE and frozen.
Only the tiny LoRA adapter matrices are trained and exchanged in federation.
"""

import torch
from collections import OrderedDict
from typing import List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import DataCollatorForLanguageModeling

from config import ModelConfig, LoRAConfig


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator that masks everything except the response portion.
    The loss is only computed on tokens after the response template.
    """

    def __init__(self, response_template_ids, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template_ids = response_template_ids

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            labels = batch["labels"][i]
            input_ids = batch["input_ids"][i]
            response_start = self._find_response_start(input_ids)
            if response_start is not None:
                labels[:response_start] = -100
            else:
                # If template not found, mask everything (skip this example)
                labels[:] = -100
            batch["labels"][i] = labels
        return batch

    def _find_response_start(self, input_ids):
        """Find where the response template starts in input_ids."""
        template = self.response_template_ids
        template_len = len(template)
        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i : i + template_len].tolist() == template:
                return i + template_len
        return None


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(model_cfg: ModelConfig):
    """
    Load the base language model with optional quantization.

    Quantization reduces model size:
    - 4-bit: ~0.6 GB for TinyLlama (from ~2.2 GB)
    - 8-bit: ~1.1 GB for TinyLlama
    - none:  ~2.2 GB for TinyLlama (fp32) or ~4.4 GB

    Returns the model and tokenizer.
    """
    device = get_device()

    # Configure quantization
    if model_cfg.quantization == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.bfloat16
        device_map = "auto"
    elif model_cfg.quantization == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        quant_config = None
        dtype = torch.float32
        device_map = device if device != "cpu" else None

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quant_config,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_base_for_lora(model):
    """
    Prepare base model for LoRA training (call ONCE, not per client).
    Freezes base weights and enables gradient checkpointing.
    """
    return prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )


def apply_lora(model, lora_cfg: LoRAConfig, rank: int, adapter_name: str = "default"):
    """
    Wrap a base model with LoRA adapters at a specific rank.

    Args:
        model: Base model (already prepared with prepare_base_for_lora)
        lora_cfg: LoRA configuration
        rank: The LoRA rank to use (2, 4, 8, 16, or 32)
        adapter_name: Name for this adapter

    Returns:
        PeftModel with LoRA adapters
    """
    from peft import PeftModel

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # If model already has PEFT adapters, remove them first
    if isinstance(model, PeftModel):
        base = model.unload()
        peft_model = get_peft_model(base, peft_config)
    else:
        peft_model = get_peft_model(model, peft_config)

    peft_model.print_trainable_parameters()
    return peft_model


def get_lora_state_dict(model) -> OrderedDict:
    """Extract only the LoRA adapter weights from a PeftModel."""
    return get_peft_model_state_dict(model)


def set_lora_state_dict(model, state_dict: OrderedDict):
    """Load LoRA adapter weights into a PeftModel."""
    set_peft_model_state_dict(model, state_dict)


def lora_state_to_numpy(model) -> List:
    """Convert LoRA state dict to list of numpy arrays (for Flower)."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for val in state_dict.values()]


def lora_numpy_to_state(model, parameters: List) -> OrderedDict:
    """Convert list of numpy arrays back to a state dict."""
    keys = get_peft_model_state_dict(model).keys()
    return OrderedDict(
        {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    )


def get_data_collator(tokenizer):
    """
    Create a data collator that masks the instruction/input portion
    so the loss is only computed on the response.

    This is standard practice for instruction-following fine-tuning.
    """
    response_template = "\n### Response:"
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]  # Skip BOS tokens
    return DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )


def count_lora_parameters(rank: int, lora_cfg: LoRAConfig, model_cfg: ModelConfig) -> int:
    """
    Estimate the number of trainable LoRA parameters for a given rank.

    For TinyLlama with hidden_dim=2048:
    Each target module gets: 2 * hidden_dim * rank parameters
    (one for A matrix, one for B matrix)
    """
    # TinyLlama hidden dimensions
    hidden_dim = 2048
    num_layers = 22
    num_targets = len(lora_cfg.target_modules)

    params_per_module = 2 * hidden_dim * rank
    total = params_per_module * num_targets * num_layers

    return total


def print_rank_comparison(lora_cfg: LoRAConfig, model_cfg: ModelConfig):
    """Print a comparison table of different ranks."""
    print(f"\n{'Rank':<8} {'Parameters':<15} {'Size (MB)':<12} {'% of Base Model'}")
    print("-" * 55)
    base_params = 1_100_000_000  # TinyLlama 1.1B

    for rank in [2, 4, 8, 16, 32]:
        params = count_lora_parameters(rank, lora_cfg, model_cfg)
        size_mb = params * 2 / (1024 * 1024)  # fp16 = 2 bytes
        pct = params / base_params * 100
        print(f"{rank:<8} {params:<15,} {size_mb:<12.2f} {pct:.4f}%")
