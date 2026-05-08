import os
from datetime import datetime
from pathlib import Path


def make_run_name(stage: str, config_path: str) -> str:
    """Build run name: {stage}-{config_stem}-{YYYY-MM-DD_HH-MM}."""
    config_stem = Path(config_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{stage}-{config_stem}-{timestamp}"


def init_wandb(config, run_name: str, job_type: str, tags=None, group=None):
    """
    Initialise a W&B run and return the run object.
    Returns None if wandb is not installed.

    Entity resolution: WANDB_ENTITY env-var -> config.wandb.entity -> None
    Offline mode:      WANDB_MODE=offline -> config.wandb.offline=true -> online
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed - skipping W&B logging")
        return None

    from omegaconf import OmegaConf, DictConfig

    wandb_cfg = {}
    if isinstance(config, DictConfig) and "wandb" in config:
        wandb_cfg = OmegaConf.to_container(config.wandb, resolve=True)

    mode = os.environ.get("WANDB_MODE") or ("offline" if wandb_cfg.get("offline", False) else None)
    entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity") or None
    project = wandb_cfg.get("project", "cardiodit-dev")
    extra_tags = list(wandb_cfg.get("tags", []))
    all_tags = (tags or []) + extra_tags

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        job_type=job_type,
        group=group,
        tags=all_tags or None,
        config=OmegaConf.to_container(config, resolve=True),
        mode=mode,
        resume="allow",
    )
    return run
