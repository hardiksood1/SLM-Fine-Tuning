from peft import LoraConfig, get_peft_model

def apply_lora(model, r=8, alpha=32, dropout=0.05):
    target_modules = [
        "attn.qkv_proj",   # QKV projection
        "attn.out_proj",   # Attention output
        "mlp.fc_in",       # MLP input
        "mlp.fc_out"       # MLP output
    ]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, config)
