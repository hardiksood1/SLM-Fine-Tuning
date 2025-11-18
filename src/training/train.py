# src/training/train.py

import torch
import os
import json
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.data.hf_dataset_loader import load_and_tokenize_datasets
from src.data.preprocessing import ConcatenatedDataset
from src.model.model_init import load_model_and_tokenizer
from src.utils.helpers import load_config

def main():
    cfg = load_config()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("final_model", exist_ok=True)

    # Load datasets config
    with open("data_configs/datasets.json") as f:
        dataset_configs = json.load(f)["datasets"]

    # Load model + tokenizer
    print("Loading model on GPU...")
    lora_cfg = cfg['model']['lora'] if cfg['model']['use_peft'] else None
    model, tokenizer = load_model_and_tokenizer(
        cfg['model']['name_or_path'],
        use_peft=cfg['model']['use_peft'],
        lora_config=lora_cfg
    )

    # 👉 FORCE MODEL TO GPU
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        raise RuntimeError("CUDA GPU not available!")

    # Tokenize dataset
    raw_tokens = load_and_tokenize_datasets(
        dataset_configs,
        split="train",
        seq_length=cfg['data']['seq_length'],
        tokenizer=tokenizer
    )

    train_dataset = ConcatenatedDataset(raw_tokens, cfg['data']['seq_length'])
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training arguments for GPU
    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg['training']['micro_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        num_train_epochs=cfg['training']['epochs'],
        learning_rate=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
        warmup_steps=cfg['training']['warmup_steps'],
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,

        # GPU Settings
        fp16=True,             # use fp16 for speed
        bf16=False,
        gradient_checkpointing=True,

        dataloader_num_workers=2,
        seed=cfg['training']['seed'],
        max_grad_norm=cfg['training']['max_grad_norm'],
        report_to=[],
        disable_tqdm=False,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    print("🚀 Starting training on GPU...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("final_model")
    tokenizer.save_pretrained("final_model")
    print("Training complete!")

if __name__ == "__main__":
    main()
