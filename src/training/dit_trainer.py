import random
import warnings
import numpy as np
import torch
from torch import amp
import torch.nn.functional as F
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
from src.models.ema import EMA
from src.config_validation import resolve_objective_loss
from src.data.temporal_alignment import load_descriptor_sidecar, pool_alpha_t
from src.utils.checkpointing import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    atomic_torch_save,
    gather_rng_states,
    restore_rng_state,
    write_checkpoint_pointer,
)
from src.utils.rng import isolated_rng
from src.utils.stage1_loading import load_stage1_strict


def _unsqueeze_right(x, ndim):
    """Add trailing singleton dimensions until ``x`` has ``ndim`` dims."""
    while x.ndim < ndim:
        x = x.unsqueeze(-1)
    return x


def _dist_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank():
    if not _dist_initialized():
        return 0
    try:
        return torch.distributed.get_rank()
    except (RuntimeError, ValueError):
        return 0


class DiTTrainer:

    def __init__(
        self,
        model,
        stage1,
        scheduler,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        device,
        run_dir: Path,
        config,
        is_main=True,
        start_epoch=0,
        best_loss=float("inf"),
        wandb_run=None,
        checkpoint=None,
        resolved_config=None,
        provenance=None,
        run_id=None,
        resume_overrides=None,
        schedule_contract=None,
    ):

        self.model = model
        self.stage1 = stage1
        self.scheduler = scheduler

        self.use_ema = config.training.get("use_ema", True)
        self.ema_decay = config.training.get("ema_decay", 0.9999)
        ema_warmup_steps = int(config.training.get("ema_warmup_steps", 0))

        if self.use_ema:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            self.ema = EMA(raw_model, decay=self.ema_decay, warmup_steps=ema_warmup_steps)
        else:
            self.ema = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.run_dir = run_dir
        self.is_main = is_main
        self.wandb_run = wandb_run
        self.resolved_config = resolved_config
        self.provenance = provenance
        self.run_id = run_id
        self.resume_overrides = list(resume_overrides or [])
        self.schedule_contract = schedule_contract
        self.log_wandb_artifacts = bool(config.get("wandb", {}).get("log_artifacts", True))

        self.n_epochs = config.training.n_epochs
        self.eval_freq = config.training.eval_freq
        target_updates = config.training.get("target_optimizer_updates", None)
        checkpoint_updates = config.training.get(
            "checkpoint_optimizer_updates", None
        )
        self.target_optimizer_updates = (
            int(target_updates) if target_updates is not None else None
        )
        self.checkpoint_optimizer_updates = (
            int(checkpoint_updates) if checkpoint_updates is not None else None
        )
        if (
            self.target_optimizer_updates is not None
            and self.target_optimizer_updates <= 0
        ):
            raise ValueError("training.target_optimizer_updates must be positive")
        if (
            self.checkpoint_optimizer_updates is not None
            and self.checkpoint_optimizer_updates <= 0
        ):
            raise ValueError(
                "training.checkpoint_optimizer_updates must be positive"
            )
        self.scale_factor = config.training.get("scale_factor", 1.0)
        self.normalize_latents = bool(config.training.get("normalize_latents", False))
        # Number of periodic checkpoints (epoch- or update-named) to keep on disk;
        # oldest are deleted as new ones are written. -1 disables rotation.
        self.keep_last_n_checkpoints = int(
            config.training.get("keep_last_n_checkpoints", 3)
        )

        self.min_snr_gamma = float(config.training.get("min_snr_gamma", 0.0))
        self.grad_accum_steps = max(1, int(config.training.get("grad_accum_steps", 1)))

        # Offset noise: adds a spatially-uniform per-channel noise component.
        # Only applied during DDPM training; 0.0 disables.
        self.offset_noise_strength = float(
            config.training.get("offset_noise_strength", 0.0)
        )

        # Self-conditioning: feed a previous x0 estimate back into the model.
        self.self_conditioning = config.training.get("self_conditioning", False)
        objective_explicit = any(
            config.training.get(key) is not None for key in ("objective_loss", "loss_type")
        )
        self.objective_loss = resolve_objective_loss(config.training)
        self.huber_beta = float(config.training.get("huber_beta", 1.0))
        if not objective_explicit:
            warnings.warn(
                "training.objective_loss is absent; preserving the historical "
                "Smooth-L1 objective as 'huber_legacy'. New runs should declare "
                "objective_loss explicitly (mse, l1, or huber_legacy).",
                RuntimeWarning,
                stacklevel=2,
            )

        # Missing means the historical unconditioned selection policy. New
        # self-conditioned runs should explicitly choose their scientific
        # primary rather than silently changing existing best-checkpoint rules.
        default_primary = "unconditioned"
        self.validation_primary_metric = str(
            config.training.get("validation_primary_metric", default_primary)
        ).lower()
        valid_primary = {"unconditioned"}
        if self.self_conditioning:
            valid_primary.add("self_conditioned")
        if self.validation_primary_metric not in valid_primary:
            raise ValueError(
                "training.validation_primary_metric must select an evaluated path; "
                f"valid choices are {sorted(valid_primary)}"
            )
        if self.self_conditioning and "validation_primary_metric" not in config.training:
            warnings.warn(
                "training.validation_primary_metric is absent; preserving the "
                "historical 'unconditioned' best-checkpoint metric while also "
                "logging 'self_conditioned'. New self-conditioned runs should "
                "declare their primary metric explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.validation_seed = int(config.training.get("validation_seed", 1729))
        self.sample_seed = int(config.training.get("sample_seed", 2718))

        self.start_epoch = start_epoch
        self.best_loss = best_loss
        progress = (checkpoint or {}).get("progress") or {}
        self._microbatch_step = int(progress.get("microbatch", 0))
        self._optimizer_updates = int(progress.get("optimizer_updates", 0))
        self._global_step = int(progress.get("global_step", self._optimizer_updates))
        self._examples_seen = int(progress.get("examples_seen", 0))
        self._tokens_seen = int(progress.get("tokens_seen", 0))
        self._batch_in_epoch = int(progress.get("batch_in_epoch", 0))
        self._resume_rng_bundle = (
            (checkpoint or {}).get("rng") if self._batch_in_epoch > 0 else None
        )
        self._last_epoch = int((checkpoint or {}).get("epoch", start_epoch - 1) or 0)

        # AMP dtype: bf16 default on Ada/Hopper — no GradScaler needed, more
        # numerically stable for v-prediction / flow-matching targets. fp16
        # retained as opt-in via training.amp_dtype: "fp16".
        amp_dtype_str = str(config.training.get("amp_dtype", "bf16")).lower()
        if amp_dtype_str in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
        elif amp_dtype_str in ("fp16", "float16", "half"):
            self.amp_dtype = torch.float16
        else:
            raise ValueError(f"Unknown training.amp_dtype '{amp_dtype_str}'")

        self.scaler = (
            amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None
        )

        self._is_flow_matching = (
            getattr(self.scheduler, "prediction_type", None) == "flow_matching"
        )

        # Cache scheduler config so _build_sample_scheduler can reuse it.
        from omegaconf import OmegaConf as _OC
        self._scheduler_cfg = (
            _OC.to_container(config.scheduler, resolve=True)
            if "scheduler" in config else {}
        )

        # Validation-time sample logging (wandb only).
        sample_log_cfg = config.training.get("sample_log", {})
        self._sample_log_enabled = bool(sample_log_cfg.get("enabled", False))
        self._sample_log_every = int(sample_log_cfg.get("every_n_evals", 1))
        self._sample_log_n_steps = int(sample_log_cfg.get("n_steps", 50))
        self._sample_log_scheduler = str(sample_log_cfg.get("scheduler", "ddim"))
        self._sample_log_n_samples = int(sample_log_cfg.get("n_samples", 1))
        self._sample_log_fps = int(sample_log_cfg.get("fps", 8))
        self._sample_log_eta = float(sample_log_cfg.get("eta", 0.0))
        self._sample_log_stage1_cfg = sample_log_cfg.get("stage1_cfg", None)
        self._sample_log_stage1_ckpt = sample_log_cfg.get("stage1_ckpt", None)
        self._sample_log_latent_shape = sample_log_cfg.get("latent_shape", None)
        self._sample_log_alpha_t_path = sample_log_cfg.get("alpha_t_path", None)
        self._sample_log_alpha_t = None
        self._sample_log_keep_stage1_on_gpu = bool(
            sample_log_cfg.get("keep_stage1_on_gpu", False)
        )
        self._stage1_eval = None  # lazy-loaded on first sample log

        # Per-channel latent normalisation (6D latents: B, C, D, H, W, T).
        self.latent_mean = None
        self.latent_std = None
        if self.normalize_latents:
            checkpoint_mean = (checkpoint or {}).get("latent_mean")
            checkpoint_std = (checkpoint or {}).get("latent_std")
            if (checkpoint_mean is None) != (checkpoint_std is None):
                raise RuntimeError(
                    "Checkpoint has incomplete latent normalization statistics; "
                    "both latent_mean and latent_std are required."
                )
            stats_path = config.training.get("latent_stats_path", None)
            if checkpoint_mean is not None:
                self.latent_mean = checkpoint_mean.to(self.device)
                self.latent_std = checkpoint_std.to(self.device)
                if self.is_main:
                    print("Loaded latent stats from checkpoint")
            elif stats_path and Path(stats_path).exists():
                stats = torch.load(stats_path, map_location="cpu")
                self.latent_mean = stats["mean"].to(self.device)
                self.latent_std = stats["std"].to(self.device)
                if self.is_main:
                    print("Loaded latent stats from", stats_path)
            else:
                if self.is_main:
                    print("Computing per-channel latent statistics…")
                self.latent_mean, self.latent_std = self._compute_latent_stats()

                if self.is_main:
                    mean_vals = self.latent_mean.view(-1).tolist()
                    std_vals = self.latent_std.view(-1).tolist()
                    print(f"  channel means : {[f'{v:.4f}' for v in mean_vals]}")
                    print(f"  channel stds  : {[f'{v:.4f}' for v in std_vals]}")
                    if stats_path:
                        stats_path = Path(stats_path)
                        stats_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_torch_save(
                            {
                                "mean": self.latent_mean.detach().cpu(),
                                "std": self.latent_std.detach().cpu(),
                            },
                            stats_path,
                        )

    def _draw_self_conditioning(self):
        """Draw one self-conditioning decision shared by every DDP rank."""
        if not self.self_conditioning:
            return False

        # Different ranks intentionally have different data/RNG streams, but
        # a conditional extra call through DDP must occur on every rank or on
        # none. Otherwise DDP buffer broadcasts and gradient all-reduces are
        # issued in different orders and the process group deadlocks.
        enabled = random.random() < 0.5 if _dist_rank() == 0 else False
        if not _dist_initialized():
            return enabled

        decision = torch.tensor(
            enabled,
            dtype=torch.uint8,
            device=self.device,
        )
        torch.distributed.broadcast(decision, src=0)
        return bool(decision.item())

    # ==========================================================
    # PUBLIC TRAIN LOOP
    # ==========================================================

    def train(self):
        if (
            self.target_optimizer_updates is not None
            and self._optimizer_updates >= self.target_optimizer_updates
        ):
            self._save_final_model()
            return

        for epoch in range(self.start_epoch, self.n_epochs):
            self._last_epoch = epoch

            batch_sampler = getattr(self.train_loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            elif hasattr(self.train_loader, "sampler") and isinstance(
                self.train_loader.sampler,
                torch.utils.data.DistributedSampler,
            ):
                self.train_loader.sampler.set_epoch(epoch)

            resume_batch = self._batch_in_epoch if epoch == self.start_epoch else 0
            reached_target = self._train_epoch(epoch, start_batch=resume_batch)

            if reached_target:
                break

            # Update-based checkpointing is exact and independent of loader size.
            # Keep epoch cadence only for explicitly legacy configurations.
            if (
                self.checkpoint_optimizer_updates is None
                and (epoch + 1) % self.eval_freq == 0
            ):

                val_loss = self._validate(epoch)
                # Every rank participates because checkpoint RNG capture is a
                # collective.  Only the main rank writes files.
                self._save_best_checkpoint(epoch, val_loss)
                self._save_periodic_checkpoint(epoch)

                if _dist_initialized():
                    torch.distributed.barrier()

            self._batch_in_epoch = 0

        if (
            self.target_optimizer_updates is not None
            and self._optimizer_updates < self.target_optimizer_updates
        ):
            raise RuntimeError(
                "training.n_epochs was exhausted before reaching "
                f"training.target_optimizer_updates={self.target_optimizer_updates}; "
                f"completed {self._optimizer_updates} optimizer updates"
            )
        self._save_final_model()

    # ==========================================================
    # TRAIN ONE EPOCH
    # ==========================================================

    def _train_epoch(self, epoch, start_batch=0):

        self.model.train()

        loader_length = len(self.train_loader)
        if start_batch < 0 or start_batch > loader_length:
            raise RuntimeError(
                f"Invalid resume batch {start_batch} for loader length {loader_length}"
            )
        iterator = iter(self.train_loader)
        for _ in range(start_batch):
            next(iterator)
        # Recreating and advancing a DataLoader iterator can consume the main
        # process RNG. Restore the state captured at the mid-epoch checkpoint
        # after positioning the iterator and before the next training draw.
        if self._resume_rng_bundle is not None:
            restore_rng_state(self._resume_rng_bundle, rank=_dist_rank())
            self._resume_rng_bundle = None
        pbar = tqdm(
            enumerate(iterator, start=start_batch),
            total=loader_length - start_batch,
            desc=f"Epoch {epoch}",
            disable=not self.is_main,
        )

        self.optimizer.zero_grad(set_to_none=True)
        accumulated_examples = 0
        accumulated_tokens = 0
        world_size = torch.distributed.get_world_size() if _dist_initialized() else 1

        for step, batch in pbar:
            self._microbatch_step += 1

            x = batch["image"]
            if hasattr(x, "as_tensor"):
                x = x.as_tensor()
            x = x.to(self.device, non_blocking=True)
            alpha_t = self._batch_alpha_t(batch)
            loss_mask = self._batch_loss_mask(batch)

            latents = self._encode_inputs(x)
            latents = latents * self.scale_factor
            batch_examples = int(latents.shape[0]) * world_size
            batch_tokens = self._count_tokens(latents) * world_size
            accumulated_examples += batch_examples
            accumulated_tokens += batch_tokens
            self._examples_seen += batch_examples
            self._tokens_seen += batch_tokens

            noise, t, noisy_latents, t_model = self._sample_t_and_noisy(latents)
            target = self._get_target(latents, noise, t)

            group_start = (step // self.grad_accum_steps) * self.grad_accum_steps
            group_size = min(
                self.grad_accum_steps, len(self.train_loader) - group_start
            )
            is_update_step = (step - group_start + 1) == group_size
            sync_context = (
                self.model.no_sync()
                if not is_update_step and hasattr(self.model, "no_sync")
                else nullcontext()
            )
            use_self_conditioning = self._draw_self_conditioning()

            with sync_context:
                # Self-conditioning: on ~50 % of steps, obtain a rough x0
                # estimate and feed it back into the gradient-carrying pass.
                x_self_cond = None
                if use_self_conditioning:
                    with torch.no_grad():
                        with amp.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype
                        ):
                            v_pred = self.model(
                                noisy_latents, t=t_model, y=None,
                                x_self_cond=None, alpha_t=alpha_t,
                            )
                        x_self_cond = self._x0_from_pred(
                            v_pred.detach(), noisy_latents, t
                        )

                with amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    noise_pred = self.model(
                        noisy_latents, t=t_model, y=None,
                        x_self_cond=x_self_cond, alpha_t=alpha_t,
                    )
                    loss = self._compute_loss(
                        noise_pred, target, t, loss_mask=loss_mask,
                    ) / group_size

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            update_succeeded = False
            effective_global_batch = accumulated_examples
            effective_global_tokens = accumulated_tokens
            if is_update_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    if not torch.isfinite(total_norm):
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        update_succeeded = True
                else:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    if not torch.isfinite(total_norm):
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.optimizer.step()
                        update_succeeded = True
                self.optimizer.zero_grad(set_to_none=True)

                if update_succeeded and self.ema:
                    self.ema.update()
                if update_succeeded:
                    self._optimizer_updates += 1
                    # W&B's global step is the count of completed optimizer
                    # updates and therefore remains monotonic across resume.
                    self._global_step = self._optimizer_updates
                    if self.lr_scheduler:
                        self.lr_scheduler.step()

            display_loss = loss.item() * group_size
            if self.is_main:
                pbar.set_postfix({"loss": f"{display_loss:.4f}"})

                if self.wandb_run and is_update_step and update_succeeded:
                    self.wandb_run.log({
                        "train/loss": display_loss,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/effective_global_batch": effective_global_batch,
                        "train/effective_global_tokens": effective_global_tokens,
                        "train/examples_seen": self._examples_seen,
                        "train/tokens_seen": self._tokens_seen,
                        "train/optimizer_updates": self._optimizer_updates,
                    }, step=self._global_step)
            if is_update_step:
                accumulated_examples = 0
                accumulated_tokens = 0
            self._batch_in_epoch = step + 1

            if update_succeeded:
                should_checkpoint = (
                    self.checkpoint_optimizer_updates is not None
                    and self._optimizer_updates % self.checkpoint_optimizer_updates == 0
                )
                if should_checkpoint:
                    val_loss = self._validate(epoch)
                    self._save_best_checkpoint(epoch, val_loss)
                    self._save_periodic_checkpoint(epoch)
                    if _dist_initialized():
                        torch.distributed.barrier()
                    # Epoch-based validation used to return directly to the
                    # outer loop. Update-based validation can occur mid-epoch,
                    # so explicitly restore training mode before continuing.
                    self.model.train()

                if (
                    self.target_optimizer_updates is not None
                    and self._optimizer_updates >= self.target_optimizer_updates
                ):
                    return True

        return False

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @torch.no_grad()
    def _validate(self, epoch):
        seed = getattr(self, "validation_seed", 1729) + _dist_rank()
        with isolated_rng(seed, self.device):
            return self._validate_isolated(epoch)

    @torch.no_grad()
    def _validate_isolated(self, epoch):
        torch.cuda.empty_cache()
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()

        try:
            self_conditioning = bool(getattr(self, "self_conditioning", False))
            metric_names = ["unconditioned"]
            if self_conditioning:
                metric_names.append("self_conditioned")
            loss_sums = {
                name: torch.zeros((), device=self.device) for name in metric_names
            }
            valid_count = torch.zeros((), device=self.device)
            n_examples = 0

            for batch in self.val_loader:

                x = batch["image"]
                if hasattr(x, "as_tensor"):
                    x = x.as_tensor()
                x = x.to(self.device, non_blocking=True)
                alpha_t = self._batch_alpha_t(batch)
                loss_mask = self._batch_loss_mask(batch)

                latents = self._encode_inputs(x)
                latents = latents * self.scale_factor

                noise, t, noisy_latents, t_model = self._sample_t_and_noisy(latents)
                target = self._get_target(latents, noise, t)

                with amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    unconditioned_pred = self.model(
                        noisy_latents, t=t_model, y=None,
                        x_self_cond=None, alpha_t=alpha_t,
                    )
                    predictions = {"unconditioned": unconditioned_pred}
                    if self_conditioning:
                        x_self_cond = self._x0_from_pred(
                            unconditioned_pred.detach(), noisy_latents, t
                        )
                        predictions["self_conditioned"] = self.model(
                            noisy_latents, t=t_model, y=None,
                            x_self_cond=x_self_cond, alpha_t=alpha_t,
                        )

                bs = latents.size(0)
                for name, prediction in predictions.items():
                    numerator, denominator = self._loss_sum_and_count(
                        prediction, target, t, loss_mask=loss_mask
                    )
                    loss_sums[name] += numerator.detach().float()
                    if name == metric_names[0]:
                        valid_count += denominator.detach().float()
                n_examples += bs

            if n_examples == 0 and not _dist_initialized():
                raise RuntimeError("Validation loader produced no batches.")

            payload = torch.stack(
                [loss_sums[name] for name in metric_names] + [valid_count]
            )
            if _dist_initialized():
                torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)

            n_total = payload[-1]
            metrics = {
                name: (payload[index] / n_total).item()
                for index, name in enumerate(metric_names)
            }
            primary_name = getattr(
                self, "validation_primary_metric", "unconditioned"
            )
            primary_loss = metrics[primary_name]

            if self.is_main:
                print(
                    f"Validation Loss ({primary_name}): {primary_loss:.6f}"
                )

                if self.wandb_run:
                    log = {f"val/loss_{name}": value for name, value in metrics.items()}
                    log["val/loss"] = primary_loss
                    self.wandb_run.log(log, step=self._global_step)

                if self._sample_log_enabled and (epoch + 1) % self._sample_log_every == 0:
                    self._log_sample(epoch)

        finally:
            if self.ema:
                self.ema.restore()

        return primary_loss

    # ==========================================================
    # VALIDATION-TIME SAMPLE LOGGING (wandb video)
    # ==========================================================

    def _load_stage1_eval(self):
        """Lazy-load and cache the VQ-GAN decoder for sample logging."""
        if self._stage1_eval is not None:
            return self._stage1_eval

        if not self._sample_log_stage1_cfg or not self._sample_log_stage1_ckpt:
            raise ValueError(
                "sample_log.enabled=True but sample_log.stage1_cfg / "
                "sample_log.stage1_ckpt are not set in the config."
            )

        model, _, _ = load_stage1_strict(
            self._sample_log_stage1_cfg,
            self._sample_log_stage1_ckpt,
            device="cpu",
        )
        self._stage1_eval = model
        return model

    def _sample_alpha_t(self, t_latent: int):
        if self._sample_log_alpha_t_path is None:
            return None
        if self._sample_log_alpha_t is None:
            alpha = load_descriptor_sidecar(self._sample_log_alpha_t_path)
            self._sample_log_alpha_t = pool_alpha_t(alpha, t_latent, method="linear")
        return self._sample_log_alpha_t.to(self.device).unsqueeze(0)

    def _build_sample_scheduler(self):
        """Build a fresh scheduler for sampling, mirroring the training schedule."""
        from src.models.ddimscheduler import DDIMScheduler
        from src.models.ddpmscheduler import DDPMScheduler
        from src.models.dpm_solver_scheduler import DPMSolverPPScheduler
        from src.models.flow_matching_scheduler import FlowMatchingScheduler

        kind = self._sample_log_scheduler.lower()
        cfg = dict(self._scheduler_cfg)

        if kind == "ddpm":
            return DDPMScheduler(**cfg)
        if kind == "ddim":
            return DDIMScheduler(**cfg)
        if kind == "dpm_pp":
            return DPMSolverPPScheduler(**cfg)
        if kind == "flow_matching":
            return FlowMatchingScheduler(**cfg)
        raise ValueError(f"Unknown sample_log.scheduler '{self._sample_log_scheduler}'")

    @torch.no_grad()
    def _decode_latent_4d(self, stage1, latent_4d):
        """(C, D, H, W, T) → (1, D, H_full, W_full, T_full), decoded slice-wise."""
        D = latent_4d.shape[1]
        slices = []
        for d in range(D):
            z = latent_4d[:, d, :, :, :].unsqueeze(0)  # (1, C, H, W, T)
            recon = stage1.decode_stage_2_outputs(z)   # (1, 1, H_full, W_full, T_full)
            slices.append(recon[0])
        return torch.stack(slices, dim=1)               # (1, D, H_full, W_full, T_full)

    @torch.no_grad()
    def _log_sample(self, epoch):
        seed = getattr(self, "sample_seed", 2718) + int(epoch)
        with isolated_rng(seed, self.device):
            return self._log_sample_isolated(epoch)

    @torch.no_grad()
    def _log_sample_isolated(self, epoch):
        """
        Sample one (or more) volumes through the EMA model, decode, log to wandb.

        Called inside _validate while EMA shadow weights are active. Memory budget
        is reduced by: small inference-step count, AMP autocast, B=1, optional
        offloading of stage1 to CPU between val intervals.
        """
        if not self.is_main or self.wandb_run is None:
            return

        try:
            from src.models.flow_matching_scheduler import FlowMatchingScheduler
            import wandb
        except ImportError:
            return

        torch.cuda.empty_cache()

        # Resolve latent shape from val batch if not configured (cached after first call).
        if self._sample_log_latent_shape is None:
            try:
                batch = next(iter(self.val_loader))
                x = batch["image"]
                if hasattr(x, "as_tensor"):
                    x = x.as_tensor()
                x = x.to(self.device, non_blocking=True)
                lat = self._encode_inputs(x)
                self._sample_log_latent_shape = list(lat.shape[1:])  # (C, D, H, W, T)
                del x, lat
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[sample_log] Could not infer latent shape: {e}")
                return
        latent_shape = self._sample_log_latent_shape

        C, D, H, W, T = latent_shape

        stage1 = self._load_stage1_eval().to(self.device)

        sched = self._build_sample_scheduler()
        sched.set_timesteps(self._sample_log_n_steps)
        is_flow_matching = isinstance(sched, FlowMatchingScheduler)

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self_cond_enabled = getattr(raw_model, "self_conditioning", False)
        sample_alpha_t = self._sample_alpha_t(T)

        videos = []
        for i in range(self._sample_log_n_samples):
            x = torch.randn(1, C, D, H, W, T, device=self.device)
            x_self_cond = None

            with amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                for t in sched.timesteps:
                    if is_flow_matching:
                        t_embed = t.item() * sched.num_train_timesteps
                        t_batch = torch.full(
                            (1,), t_embed, device=self.device, dtype=torch.float32
                        )
                    else:
                        t_batch = torch.full(
                            (1,), int(t), device=self.device, dtype=torch.long
                        )

                    noise_pred = raw_model(
                        x, t=t_batch, y=None, x_self_cond=x_self_cond,
                        alpha_t=sample_alpha_t,
                    )
                    x, x0_pred = sched.step(noise_pred, t, x)
                    if self_cond_enabled:
                        x_self_cond = x0_pred.detach()

            # Un-normalise + un-scale to align with VQ-GAN input distribution.
            x = x / self.scale_factor
            if self.latent_mean is not None:
                x = x * (self.latent_std + 1e-8) + self.latent_mean

            volume = self._decode_latent_4d(stage1, x[0]).float().cpu()  # (1, D, H, W, T)
            volume = volume.clamp(-1.0, 1.0).numpy()[0]                  # (D, H, W, T)

            # Mid-D slice as a video over time: (T, 1, H, W) uint8 in [0, 255].
            mid_d = volume.shape[0] // 2
            slc = volume[mid_d]                                          # (H, W, T)
            slc = (slc + 1.0) * 0.5
            slc = np.clip(slc * 255.0, 0, 255).astype(np.uint8)
            video = np.transpose(slc, (2, 0, 1))[:, None, :, :]          # (T, 1, H, W)
            video = np.repeat(video, 3, axis=1)                          # (T, 3, H, W) — gif writer needs RGB
            videos.append(video)

            del x, volume

        log_dict = {}
        for i, v in enumerate(videos):
            log_dict[f"val/sample_{i}"] = wandb.Video(v, fps=self._sample_log_fps, format="gif")
        self.wandb_run.log(log_dict, step=self._global_step)

        if not self._sample_log_keep_stage1_on_gpu:
            self._stage1_eval = self._stage1_eval.cpu()

        torch.cuda.empty_cache()

    # ==========================================================
    # HELPERS — timestep sampling and noise
    # ==========================================================

    def _sample_t_and_noisy(self, latents):
        B = latents.shape[0]

        if self._is_flow_matching:
            t = self.scheduler.sample_timesteps(B, device=self.device)
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)
            t_model = t * self.scheduler.num_train_timesteps
        else:
            t = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (B,),
                device=self.device,
            ).long()

            noise = torch.randn_like(latents)

            # Offset noise: adds a spatially-uniform per-channel perturbation.
            # Broadcast shape (B, C, 1, 1, 1, 1) covers all four 4D latent dims.
            if self.offset_noise_strength > 0.0:
                offset = torch.randn(
                    B, latents.shape[1], 1, 1, 1, 1, device=self.device
                )
                noise = noise + self.offset_noise_strength * offset

            noisy_latents = self.scheduler.add_noise(
                original_samples=latents, noise=noise, timesteps=t,
            )
            t_model = t

        return noise, t, noisy_latents, t_model

    # ==========================================================
    # HELPERS — training target
    # ==========================================================

    def _get_target(self, latents, noise, timesteps):
        pt = self.scheduler.prediction_type

        if pt == "flow_matching":
            return self.scheduler.get_velocity(latents, noise)
        elif pt == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)
        elif pt == "epsilon":
            return noise
        else:
            raise ValueError(f"Unknown prediction_type '{pt}'")

    # ==========================================================
    # HELPERS — x0 estimate from model prediction (self-conditioning)
    # ==========================================================

    def _x0_from_pred(self, pred, x_t, t):
        """
        Derive a clean-latent estimate x0 from the model's prediction.

        Used for self-conditioning.  Formula depends on prediction type.
        """
        if self._is_flow_matching:
            # x_t = (1-t)*x0 + t*eps  and  v = eps - x0
            # → x0 = x_t - t * v
            t_bc = _unsqueeze_right(t.float(), x_t.ndim)
            return x_t - t_bc * pred

        pt = self.scheduler.prediction_type
        acp = self.scheduler.alphas_cumprod.to(device=x_t.device)

        if pt == "v_prediction":
            sqrt_acp = _unsqueeze_right(acp[t].sqrt(), x_t.ndim)
            sqrt_1m = _unsqueeze_right((1.0 - acp[t]).sqrt(), x_t.ndim)
            return sqrt_acp * x_t - sqrt_1m * pred

        elif pt == "epsilon":
            sqrt_acp = _unsqueeze_right(acp[t].sqrt(), x_t.ndim)
            sqrt_1m = _unsqueeze_right((1.0 - acp[t]).sqrt(), x_t.ndim)
            return (x_t - sqrt_1m * pred) / sqrt_acp.clamp(min=1e-8)

        else:
            return pred  # "sample" prediction — model already predicts x0

    # ==========================================================
    # HELPERS — Min-SNR weighted loss
    # ==========================================================

    def _compute_loss(self, pred, target, t, loss_mask=None):
        element_loss = self._elementwise_loss(pred, target)
        if loss_mask is not None:
            mask = loss_mask.to(device=element_loss.device, dtype=element_loss.dtype)
            while mask.ndim < element_loss.ndim:
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(element_loss)
            batch_loss = element_loss.mul(mask).sum(dim=list(range(1, element_loss.ndim)))
            denom = mask.sum(dim=list(range(1, element_loss.ndim))).clamp_min(1.0)
            batch_loss = batch_loss / denom
        else:
            batch_loss = element_loss.mean(dim=list(range(1, element_loss.ndim)))

        if getattr(self, "min_snr_gamma", 0.0) > 0.0:
            weights = self._min_snr_weights(t)
            batch_loss = batch_loss * weights

        return batch_loss.mean()

    def _elementwise_loss(self, pred, target):
        objective = getattr(self, "objective_loss", "huber_legacy")
        if objective == "mse":
            return F.mse_loss(pred.float(), target.float(), reduction="none")
        elif objective == "l1":
            return F.l1_loss(pred.float(), target.float(), reduction="none")
        elif objective == "huber_legacy":
            return F.smooth_l1_loss(
                pred.float(), target.float(), reduction="none",
                beta=getattr(self, "huber_beta", 1.0),
            )
        else:
            raise RuntimeError(f"Unsupported objective loss {objective!r}")

    def _loss_sum_and_count(self, pred, target, t, loss_mask=None):
        """Exact validation numerator/denominator before distributed reduction."""
        element_loss = self._elementwise_loss(pred, target)
        if getattr(self, "min_snr_gamma", 0.0) > 0.0:
            weights = self._min_snr_weights(t)
            while weights.ndim < element_loss.ndim:
                weights = weights.unsqueeze(-1)
            element_loss = element_loss * weights
        if loss_mask is not None:
            mask = loss_mask.to(device=element_loss.device, dtype=element_loss.dtype)
            while mask.ndim < element_loss.ndim:
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(element_loss)
            return element_loss.mul(mask).sum(), mask.sum()
        return element_loss.sum(), torch.tensor(
            float(element_loss.numel()), device=element_loss.device
        )

    def _min_snr_weights(self, t):
        gamma = self.min_snr_gamma

        if self._is_flow_matching:
            t_safe = t.clamp(min=1e-5)
            snr = ((1.0 - t_safe) / t_safe) ** 2
        else:
            acp = self.scheduler.alphas_cumprod.to(device=t.device)
            snr = acp[t] / (1.0 - acp[t])

        if getattr(self.scheduler, "prediction_type", None) == "v_prediction":
            denom = snr + 1.0
        else:
            denom = snr.clamp(min=1e-8)

        return torch.clamp(snr, max=gamma) / denom.clamp(min=1e-8)

    # ==========================================================
    # HELPERS — encoding
    # ==========================================================

    def _raw_encode(self, x):
        """Encode x through stage1 without applying latent normalisation."""
        if self.stage1 is None:
            return x
        with torch.no_grad():
            return self.stage1.encode_stage_2_inputs(x)

    def _encode_inputs(self, x):
        """Encode and optionally apply per-channel normalisation."""
        latents = self._raw_encode(x)
        if self.latent_mean is not None:
            latents = (latents - self.latent_mean) / (self.latent_std + 1e-8)
        return latents

    def _batch_alpha_t(self, batch):
        alpha_t = batch.get("alpha_t")
        if alpha_t is None:
            return None
        if hasattr(alpha_t, "as_tensor"):
            alpha_t = alpha_t.as_tensor()
        return alpha_t.to(self.device, non_blocking=True).float()

    def _batch_loss_mask(self, batch):
        loss_mask = batch.get("loss_mask")
        if loss_mask is None:
            return None
        if hasattr(loss_mask, "as_tensor"):
            loss_mask = loss_mask.as_tensor()
        return loss_mask.to(self.device, non_blocking=True).float()

    def _count_tokens(self, latents):
        """Number of transformer tokens processed by this local batch."""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        patch_size = getattr(getattr(raw_model, "x_embedder", None), "patch_size", None)
        spatial = tuple(int(value) for value in latents.shape[-4:])
        if patch_size is None:
            tokens_per_example = int(np.prod(spatial))
        else:
            tokens_per_example = int(
                np.prod([dim // int(patch) for dim, patch in zip(spatial, patch_size)])
            )
        return int(latents.shape[0]) * tokens_per_example

    def load_ema_state(self, state_dict):
        if self.ema and state_dict is not None:
            self.ema.load_state_dict(state_dict)

    # ==========================================================
    # HELPERS — per-channel latent statistics (6D: B, C, D, H, W, T)
    # ==========================================================

    def _compute_latent_stats(self):
        """
        Two-pass computation of per-channel mean and std over the training set.

        Latents are 6D: (B, C, D, H, W, T).  Spatial dims 2–5 are flattened
        for per-channel statistics.  Returns (mean, std) each of shape
        (1, C, 1, 1, 1, 1).
        """
        with torch.no_grad():
            chan_sum = None
            chan_sq_sum = None
            n_elements = 0

            for batch in self.train_loader:
                x = batch["image"]
                if hasattr(x, "as_tensor"):
                    x = x.as_tensor()
                x = x.to(self.device, non_blocking=True)
                loss_mask = self._batch_loss_mask(batch)
                latents = self._raw_encode(x).double()
                B, C = latents.shape[:2]
                n_spatial = latents.shape[2] * latents.shape[3] * latents.shape[4]
                if loss_mask is not None:
                    mask = loss_mask.to(device=latents.device, dtype=latents.dtype)
                    while mask.ndim < latents.ndim:
                        mask = mask.unsqueeze(1)
                    batch_sum = (latents * mask).sum(dim=(0, 2, 3, 4, 5))
                    batch_sq_sum = (latents.square() * mask).sum(dim=(0, 2, 3, 4, 5))
                    n_elements += int(mask[:, 0, 0, 0, 0, :].sum().item()) * n_spatial
                else:
                    n_time = latents.shape[5]
                    flat = latents.view(B, C, n_spatial * n_time)
                    batch_sum = flat.sum(dim=(0, 2))
                    batch_sq_sum = flat.square().sum(dim=(0, 2))
                    n_elements += B * n_spatial * n_time
                chan_sum = batch_sum if chan_sum is None else chan_sum + batch_sum
                chan_sq_sum = (
                    batch_sq_sum if chan_sq_sum is None else chan_sq_sum + batch_sq_sum
                )

            if chan_sum is None or n_elements == 0:
                raise RuntimeError("Training loader produced no batches for latent stats.")

            payload = torch.cat([
                chan_sum,
                chan_sq_sum,
                torch.tensor([float(n_elements)], device=self.device),
            ])
            if _dist_initialized():
                torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)

            C = (payload.numel() - 1) // 2
            chan_sum = payload[:C]
            chan_sq_sum = payload[C:2 * C]
            n_elements = payload[-1].clamp_min(1.0)

            mean = chan_sum / n_elements
            var = chan_sq_sum / n_elements - mean.square()
            std = var.clamp_min(0.0).sqrt()

        return (
            mean.float().view(1, -1, 1, 1, 1, 1),
            std.float().view(1, -1, 1, 1, 1, 1),
        )

    # ==========================================================
    # CHECKPOINTING
    # ==========================================================

    def _save_best_checkpoint(self, epoch, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            checkpoint = self._build_checkpoint(epoch)
            if getattr(self, "is_main", True):
                atomic_torch_save(checkpoint, self.run_dir / "best_model.pth")
            if (
                getattr(self, "is_main", True)
                and self.wandb_run
                and self.log_wandb_artifacts
            ):
                self._upload_best_model_artifact(epoch, val_loss)

    def _save_periodic_checkpoint(self, epoch):
        ckpt = self._build_checkpoint(epoch)
        if self.is_main:
            if self.checkpoint_optimizer_updates is not None:
                destination = (
                    self.run_dir
                    / f"checkpoint_update_{self._optimizer_updates}.pth"
                )
            else:
                destination = self.run_dir / f"checkpoint_epoch_{epoch}.pth"
            atomic_torch_save(ckpt, destination)
            write_checkpoint_pointer(self.run_dir, destination)
            self._rotate_periodic_checkpoints()

    def _rotate_periodic_checkpoints(self):
        keep = self.keep_last_n_checkpoints
        if keep < 0:
            return
        ckpts = sorted(
            [
                *self.run_dir.glob("checkpoint_epoch_*.pth"),
                *self.run_dir.glob("checkpoint_update_*.pth"),
            ],
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        # The pointer must always retain one canonical payload.  A configured
        # value of zero therefore means "pointer target only", not zero files.
        effective_keep = max(1, keep)
        for old in ckpts[:-effective_keep]:
            try:
                old.unlink()
            except OSError:
                pass

    def _save_final_model(self):
        checkpoint = self._build_checkpoint(self._last_epoch)
        if self.is_main:
            destination = self.run_dir / "final_model.pth"
            atomic_torch_save(checkpoint, destination)
            write_checkpoint_pointer(self.run_dir, destination)

    def _build_checkpoint(self, epoch):
        rng = gather_rng_states()
        ckpt = {
            "checkpoint_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "checkpoint_kind": "dit",
            "epoch": epoch,
            "best_loss": self.best_loss,
            "model": self._get_model_state(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "ema": self.ema.state_dict() if self.ema else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "scale_factor": float(self.scale_factor),
            "normalize_latents": bool(self.normalize_latents),
            "objective_loss": getattr(self, "objective_loss", "huber_legacy"),
            "huber_beta": float(getattr(self, "huber_beta", 1.0)),
            "validation_primary_metric": getattr(
                self, "validation_primary_metric", "unconditioned"
            ),
            "schedule_unit": "optimizer_update",
            "schedule_contract": getattr(self, "schedule_contract", None),
            "progress": {
                "epoch": int(epoch),
                "batch_in_epoch": getattr(self, "_batch_in_epoch", 0),
                "microbatch": self._microbatch_step,
                "optimizer_updates": self._optimizer_updates,
                "global_step": self._global_step,
                "examples_seen": getattr(self, "_examples_seen", 0),
                "tokens_seen": getattr(self, "_tokens_seen", 0),
            },
            "metrics": {"best": self.best_loss},
            "rng": rng,
            "resolved_config": self.resolved_config,
            "run_id": self.run_id,
            "provenance": self.provenance,
            "resume_overrides": list(self.resume_overrides),
        }
        # Persist latent normalisation stats so they are available at inference.
        if self.latent_mean is not None:
            ckpt["latent_mean"] = self.latent_mean.cpu()
            ckpt["latent_std"] = self.latent_std.cpu()
        else:
            ckpt["latent_mean"] = None
            ckpt["latent_std"] = None
        ckpt["availability"] = {
            field: ckpt.get(field) is not None
            for field in (
                "model", "ema", "optimizer", "lr_scheduler", "scaler",
                "latent_mean", "latent_std", "rng", "resolved_config",
                "run_id", "provenance",
            )
        }
        return ckpt

    def _upload_best_model_artifact(self, epoch, val_loss):
        import wandb
        artifact = wandb.Artifact(
            name="dit-best-model",
            type="model",
            metadata={"epoch": epoch, "val_loss": round(val_loss, 6)},
        )
        artifact.add_file(str(self.run_dir / "best_model.pth"))
        self.wandb_run.log_artifact(artifact)

    def _get_model_state(self):
        return (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )
