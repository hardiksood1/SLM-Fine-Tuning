import os
import yaml
from datetime import datetime

def load_config(path="configs/training_config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_run_name(cfg):
    model = cfg['model']['name_or_path'].split('/')[-1]
    datasets = "+".join([d['id'].split('/')[-1] for d in cfg.get('datasets', [])])
    return f"{model}_lora_r{cfg['model']['lora']['r']}_{datasets[:30]}"