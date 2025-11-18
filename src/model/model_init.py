from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
import torch

def load_model_and_tokenizer(model_name, use_peft=False, lora_config=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"   # GPU auto-placement
    )

    if use_peft:
        from .peft_lora_utils import apply_lora
        model = prepare_model_for_kbit_training(model)
        model = apply_lora(model, **lora_config)

    return model, tokenizer
