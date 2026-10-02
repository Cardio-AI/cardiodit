import os
from datetime import datetime
from pathlib import Path


def make_run_name(stage: str, config_path: str) -> str:
    """Build run name: {stage}-{config_stem}-{YYYY-MM-DD_HH-MM}."""
    config_stem = Path(config_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{stage}-{config_stem}-{timestamp}"


def init_wandb(config, run_name: str, job_type: str, tags=None, group=None, run_id: str | None = None):
    """
    Initialise a W&B run and return the run object.
    Returns None if wandb is not installed.

    Entity resolution: WANDB_ENTITY env-var → config.wandb.entity → None (anonymous)
    Offline mode:      WANDB_MODE=offline  → config.wandb.offline=true → online

    If ``run_id`` is provided, wandb will resume that run (resume="allow"),
    keeping all step/metric history attached to the original run.
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping W&B logging")
        return None

    from omegaconf import OmegaConf, DictConfig

    wandb_cfg = {}
    if isinstance(config, DictConfig) and "wandb" in config:
        wandb_cfg = OmegaConf.to_container(config.wandb, resolve=True)

    mode = os.environ.get("WANDB_MODE") or ("offline" if wandb_cfg.get("offline", True) else None)
    entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity") or None
    project = wandb_cfg.get("project", "cardiodit-dev")
    extra_tags = list(wandb_cfg.get("tags", []))
    all_tags = (tags or []) + extra_tags

    log_dir = Path(os.environ.get("WANDB_DIR", str(Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")) / "wandb"))).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = {
        "dir": str(log_dir),
        "project": project,
        "entity": entity,
        "name": run_name,
        "job_type": job_type,
        "group": group,
        "tags": all_tags or None,
        "config": OmegaConf.to_container(config, resolve=True),
        "mode": mode,
    }
    try:
        return wandb.init(
            **init_kwargs,
            id=run_id,
            resume="allow",
        )
    except wandb.errors.CommError:
        if not run_id:
            raise
        # W&B permanently rejects an ID when its remote run was deleted.
        # A checkpoint remains valid in that situation, so detach the failed
        # backend and continue its history under a fresh tracking ID.
        print(
            f"W&B could not resume run {run_id!r}; retrying with a fresh run ID."
        )
        wandb.teardown()
        return wandb.init(
            **init_kwargs,
            id=None,
            resume=None,
        )


def load_resume_run_id(id_file: Path, *, has_checkpoint: bool) -> str | None:
    """Return a persisted W&B ID only when model state will also resume."""
    if not has_checkpoint or not id_file.is_file():
        return None
    run_id = id_file.read_text().strip()
    return run_id or None
