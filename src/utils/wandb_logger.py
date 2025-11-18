import wandb
from .helpers import get_run_name

def init_wandb(cfg, datasets_list):
    tags = [
        f"model:{cfg['model']['name_or_path'].split('/')[-1]}",
        f"datasets:{','.join([d['id'].split('/')[-1] for d in datasets_list[:3]])}",
        f"lora_r:{cfg['model']['lora']['r']}",
        f"lr:{cfg['training']['lr']}",
        "gpu:colab-t4"
    ]

    wandb.init(
        project=cfg['logging']['wandb_project'],
        entity=cfg['logging']['wandb_entity'],
        config=cfg,
        name=get_run_name(cfg),
        tags=tags
    )