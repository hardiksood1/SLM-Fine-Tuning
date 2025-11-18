# SLM-LoRA Assignment

Fine-tune a small language model (SLM) using PEFT/LoRA on curated HF datasets with full W&B logging, evaluation, and LaTeX reporting.

## Reproducibility Checklist
- Model: `Salesforce/codegen-350M-multi` (commit: `main`)
- Datasets: See `data_configs/datasets.json` 
- Config: `configs/training_config.yaml`
- Seed: 42
- Hardware: Colab T4 (16GB)
- Final Checkpoint: [Link placeholder]

## Setup
```bash
pip install -r requirements.txt