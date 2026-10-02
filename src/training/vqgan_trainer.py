import torch
from pathlib import Path
from collections import defaultdict
from torchvision.utils import make_grid
from torch import amp
from tqdm import tqdm
from contextlib import contextmanager
from src.utils.checkpointing import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    atomic_torch_save,
    gather_rng_states,
    write_checkpoint_pointer,
)
from src.utils.rng import isolated_rng


def _dist_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@contextmanager
def freeze_module_parameters(module):
    """Temporarily freeze parameters while retaining input-gradient flow."""
    parameters = list(module.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)


def _batch_temporal_mask(batch, device):
    mask = batch.get("temporal_valid_mask", batch.get("loss_mask"))
    return None if mask is None else mask.to(device)


class VQGANTrainer:

    def __init__(
        self,
        model,
        discriminator,
        loss_fn,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
        train_loader,
        val_loader,
        device,
        run_dir: Path,
        config,
        writer_train=None,
        writer_val=None,
        is_main=True,
        start_epoch=0,
        best_loss=float("inf"),
        wandb_run=None,
        checkpoint=None,
        resume_overrides=None,
        resolved_config=None,
        provenance=None,
        run_id=None,
        schedule_contract=None,
    ):

        self.model = model
        self.discriminator = discriminator
        self.loss_fn = loss_fn

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.run_dir = run_dir
        self.writer_train = writer_train
        self.writer_val = writer_val
        self.is_main = is_main
        self.wandb_run = wandb_run
        self.log_wandb_artifacts = bool(config.get("wandb", {}).get("log_artifacts", True))

        self.n_epochs = config.training.n_epochs
        self.eval_freq = config.training.eval_freq

        self.max_adv_weight = config.losses.adv_weight
        self.perceptual_weight = config.losses.perceptual_weight
        self.adv_warmup_epochs = config.losses.adv_warmup

        self.scaler_g = amp.GradScaler("cuda")
        self.scaler_d = amp.GradScaler("cuda")

        self.start_epoch = start_epoch
        self.best_loss = best_loss
        progress = (checkpoint or {}).get("progress") or {}
        self._optimizer_updates = int(progress.get("optimizer_updates", 0))
        self._discriminator_updates = int(progress.get("discriminator_updates", 0))
        self._global_step = int(progress.get("global_step", self._optimizer_updates))
        self._examples_seen = int(progress.get("examples_seen", 0))
        self._tokens_seen = int(progress.get("tokens_seen", 0))
        self.validation_seed = int(config.training.get("validation_seed", 1729))
        self.resume_overrides = list(resume_overrides or [])
        self.resolved_config = resolved_config
        self.provenance = provenance
        self.run_id = run_id
        self.schedule_contract = schedule_contract

    # ==========================================================
    # PUBLIC TRAIN LOOP
    # ==========================================================

    def train(self):
        val_loss = self._validate(self.start_epoch)
        if self.is_main:
            print(f"epoch {self.start_epoch} initial val loss: {val_loss:.4f}")

        for epoch in range(self.start_epoch, self.n_epochs):

            if hasattr(self.train_loader, "sampler") and isinstance(
                self.train_loader.sampler, torch.utils.data.DistributedSampler
            ):
                self.train_loader.sampler.set_epoch(epoch)

            adv_weight = self._get_adv_weight(epoch)
            self._train_epoch(epoch, adv_weight)

            if (epoch + 1) % self.eval_freq == 0:
                val_loss = self._validate(epoch)
                if self.is_main:
                    print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
                # Every rank must enter checkpoint construction because RNG
                # capture uses an all-gather; only rank zero writes files.
                self._save_checkpoint(epoch, val_loss)

                if _dist_initialized():
                    torch.distributed.barrier()

        self._save_final_model(self.n_epochs)

    # ==========================================================
    # TRAIN ONE EPOCH
    # ==========================================================

    def _train_epoch(self, epoch, adv_weight):
        self.model.train()
        self.discriminator.train()

        epoch_perplexity = 0.0
        epoch_used_codes = 0.0
        num_batches = 0
        quantizer = self.model.module.quantizer if hasattr(self.model, "module") else self.model.quantizer

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.is_main)

        for step, batch in enumerate(pbar):
            images = batch["image"]
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            images = images.to(self.device)
            temporal_mask = _batch_temporal_mask(batch, self.device)

            # ---- Generator step ----
            self.optimizer_g.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda"):
                reconstruction, quantization_loss, indices = self.model(images)
                epoch_perplexity += quantizer.perplexity.item()
                epoch_used_codes += indices.unique().numel()
                num_batches += 1
                with freeze_module_parameters(self.discriminator):
                    loss, losses = self.loss_fn.generator_loss(
                        self.discriminator,
                        adv_weight,
                        images,
                        reconstruction,
                        quantization_loss,
                        temporal_mask=temporal_mask,
                    )

            self.scaler_g.scale(loss).backward()
            self.scaler_g.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
            if self.scheduler_g:
                self.scheduler_g.step()

            # ---- Discriminator step ----
            d_losses = {}
            if adv_weight > 0:
                self.optimizer_d.zero_grad(set_to_none=True)
                with amp.autocast(device_type="cuda"):
                    d_loss, d_losses = self.loss_fn.discriminator_loss(
                        self.discriminator,
                        adv_weight,
                        images,
                        reconstruction.detach(),
                        temporal_mask=temporal_mask,
                    )
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
                self._discriminator_updates += 1
                if self.scheduler_d:
                    self.scheduler_d.step()

            self._optimizer_updates += 1
            self._global_step = self._optimizer_updates
            world_size = torch.distributed.get_world_size() if _dist_initialized() else 1
            effective_global_batch = int(images.shape[0]) * world_size
            self._examples_seen += effective_global_batch
            batch_tokens = int(indices.numel()) * world_size
            self._tokens_seen += batch_tokens

            if self.is_main:
                logs = {**losses, **d_losses}
                logs["total_g_loss"] = loss.item()
                for k, v in logs.items():
                    if self.writer_train is not None:
                        self.writer_train.add_scalar(k, v, self._global_step)
                pbar.set_postfix({"loss": f"{losses['l1_loss']:.4f}"})

                if self.wandb_run:
                    wlogs = {f"train/{k}": v for k, v in logs.items()}
                    wlogs["train/lr_g"] = self.optimizer_g.param_groups[0]["lr"]
                    wlogs.update({
                        "train/effective_global_batch": effective_global_batch,
                        "train/effective_global_tokens": batch_tokens,
                        "train/examples_seen": self._examples_seen,
                        "train/tokens_seen": self._tokens_seen,
                        "train/optimizer_updates": self._optimizer_updates,
                    })
                    self.wandb_run.log(wlogs, step=self._global_step)

        if self.is_main and num_batches > 0:
            avg_perplexity = epoch_perplexity / num_batches
            avg_used_codes = epoch_used_codes / num_batches
            usage_ratio = avg_used_codes / quantizer.quantizer.num_embeddings
            self.writer_train.add_scalar("codebook/perplexity", avg_perplexity, epoch)
            self.writer_train.add_scalar("codebook/used_codes", avg_used_codes, epoch)
            self.writer_train.add_scalar("codebook/usage_ratio", usage_ratio, epoch)

            if self.wandb_run:
                self.wandb_run.log({
                    "train/codebook/perplexity": avg_perplexity,
                    "train/codebook/used_codes": avg_used_codes,
                    "train/codebook/usage_ratio": usage_ratio,
                }, step=self._global_step)

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @torch.no_grad()
    def _validate(self, epoch):
        rank = 0
        if _dist_initialized():
            try:
                rank = torch.distributed.get_rank()
            except (RuntimeError, ValueError):
                rank = 0
        with isolated_rng(getattr(self, "validation_seed", 1729) + rank, self.device):
            return self._validate_isolated(epoch)

    @torch.no_grad()
    def _validate_isolated(self, epoch):
        torch.cuda.empty_cache()
        self.model.eval()
        self.discriminator.eval()

        total_losses = defaultdict(float)
        n = 0
        sample_images, sample_recons = None, None
        epoch_perplexity = 0.0
        epoch_used_codes = 0.0
        num_batches = 0
        quantizer = self.model.module.quantizer if hasattr(self.model, "module") else self.model.quantizer

        for batch_idx, batch in enumerate(self.val_loader):
            images = batch["image"]
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            images = images.to(self.device)
            temporal_mask = _batch_temporal_mask(batch, self.device)

            with amp.autocast(device_type="cuda"):
                reconstruction, quantization_loss, indices = self.model(images)
                adv_weight = self._get_adv_weight(epoch)
                _, losses = self.loss_fn.generator_loss(
                    self.discriminator,
                    adv_weight,
                    images,
                    reconstruction,
                    quantization_loss,
                    temporal_mask=temporal_mask,
                )

            epoch_perplexity += quantizer.perplexity.item()
            epoch_used_codes += indices.unique().numel()
            num_batches += 1

            bs = images.shape[0]
            for k, v in losses.items():
                total_losses[k] += v.detach().float() * bs
            n += bs

            if batch_idx == 0 and self.is_main:
                sample_images = images[:4].detach().cpu()
                sample_recons = reconstruction[:4].detach().cpu()

        loss_keys = sorted(total_losses)
        if n == 0:
            raise RuntimeError("Validation loader produced no batches.")

        payload = torch.stack(
            [total_losses[k].to(self.device) for k in loss_keys]
            + [
                torch.tensor(float(n), device=self.device),
                torch.tensor(float(epoch_perplexity), device=self.device),
                torch.tensor(float(epoch_used_codes), device=self.device),
                torch.tensor(float(num_batches), device=self.device),
            ]
        )
        if _dist_initialized():
            torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)

        loss_sums = payload[:len(loss_keys)]
        n_total = float(payload[len(loss_keys)].item())
        epoch_perplexity = float(payload[len(loss_keys) + 1].item())
        epoch_used_codes = float(payload[len(loss_keys) + 2].item())
        num_batches = int(payload[len(loss_keys) + 3].item())
        total_losses = {
            k: (loss_sums[i] / n_total).item()
            for i, k in enumerate(loss_keys)
        }

        tb_step = epoch * len(self.train_loader)
        if self.is_main and self.writer_val is not None:
            for k, v in total_losses.items():
                self.writer_val.add_scalar(k, v, tb_step)

        if self.is_main and num_batches > 0:
            avg_perplexity = epoch_perplexity / num_batches
            avg_used_codes = epoch_used_codes / num_batches
            usage_ratio = avg_used_codes / quantizer.quantizer.num_embeddings
            if self.writer_val is not None:
                self.writer_val.add_scalar("codebook/perplexity", avg_perplexity, epoch)
                self.writer_val.add_scalar("codebook/used_codes", avg_used_codes, epoch)
                self.writer_val.add_scalar("codebook/usage_ratio", usage_ratio, epoch)

            if self.wandb_run and self._global_step > 0:
                wlogs = {f"val/{k}": v for k, v in total_losses.items()}
                wlogs.update({
                    "val/codebook/perplexity": avg_perplexity,
                    "val/codebook/used_codes": avg_used_codes,
                    "val/codebook/usage_ratio": usage_ratio,
                })
                self.wandb_run.log(wlogs, step=self._global_step)

        if self.is_main and sample_images is not None:
            self._log_reconstructions(sample_images, sample_recons, epoch)

        if self.is_main:
            print(f"AUTORESEARCH_METRIC:{total_losses['l1_loss']:.6f}", flush=True)
            if "perceptual_loss" in total_losses:
                print(f"AUTORESEARCH_PERCEPTUAL:{total_losses['perceptual_loss']:.6f}", flush=True)

        return total_losses["l1_loss"]

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _log_reconstructions(self, images, recons, epoch, n_images=4):
        center = images.shape[-1] // 2  # center frame along T
        images_slice = images[:n_images, :, :, :, center]
        recons_slice = recons[:n_images, :, :, :, center]

        images_grid = make_grid(images_slice, normalize=True, scale_each=True)
        recons_grid = make_grid(recons_slice, normalize=True, scale_each=True)

        self.writer_val.add_image("images/ground_truth", images_grid, epoch)
        self.writer_val.add_image("images/reconstruction", recons_grid, epoch)

        if self.wandb_run and self._global_step > 0:
            import wandb
            self.wandb_run.log({
                "val/images/ground_truth": wandb.Image(images_grid.permute(1, 2, 0).numpy()),
                "val/images/reconstruction": wandb.Image(recons_grid.permute(1, 2, 0).numpy()),
            }, step=self._global_step)

    def _get_adv_weight(self, epoch):
        if epoch < self.adv_warmup_epochs:
            return self.max_adv_weight * (epoch / self.adv_warmup_epochs)
        return self.max_adv_weight

    def _save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss

        ckpt = self._build_checkpoint(epoch)
        if not getattr(self, "is_main", True):
            return
        destination = self.run_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        atomic_torch_save(ckpt, destination)
        write_checkpoint_pointer(self.run_dir, destination)

        if is_best:
            atomic_torch_save(ckpt, self.run_dir / "best_model.pth")
            print(f"New best val loss: {val_loss:.4f}")
            if self.wandb_run and self.log_wandb_artifacts:
                self._upload_best_model_artifact(epoch, val_loss)

    def _build_checkpoint(self, epoch):
        rng = gather_rng_states()
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_disc = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator

        ckpt = {
            "checkpoint_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "checkpoint_kind": "vqgan",
            "epoch": epoch + 1,
            "best_loss": self.best_loss,
            "model": raw_model.state_dict(),
            "discriminator": raw_disc.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict() if self.scheduler_g else None,
            "scheduler_d": self.scheduler_d.state_dict() if self.scheduler_d else None,
            "scaler_g": self.scaler_g.state_dict() if self.scaler_g else None,
            "scaler_d": self.scaler_d.state_dict() if self.scaler_d else None,
            "rng": rng,
            "schedule_unit": "optimizer_update",
            "schedule_contract": getattr(self, "schedule_contract", None),
            "progress": {
                "optimizer_updates": getattr(self, "_optimizer_updates", 0),
                "discriminator_updates": getattr(
                    self, "_discriminator_updates", 0
                ),
                "global_step": getattr(self, "_global_step", 0),
                "examples_seen": getattr(self, "_examples_seen", 0),
                "tokens_seen": getattr(self, "_tokens_seen", 0),
            },
            "resume_overrides": list(getattr(self, "resume_overrides", [])),
            "resolved_config": getattr(self, "resolved_config", None),
            "run_id": getattr(self, "run_id", None),
            "provenance": getattr(self, "provenance", None),
        }
        if hasattr(raw_model, "quantizer") and hasattr(raw_model.quantizer, "quantizer"):
            ckpt["ema_cluster_size"] = raw_model.quantizer.quantizer.ema_cluster_size
            ckpt["ema_w"] = raw_model.quantizer.quantizer.ema_w
        ckpt["availability"] = {
            field: ckpt.get(field) is not None
            for field in (
                "model", "discriminator", "optimizer_g", "optimizer_d",
                "scheduler_g", "scheduler_d", "scaler_g", "scaler_d",
                "rng", "resolved_config", "run_id", "provenance",
                "ema_cluster_size", "ema_w",
            )
        }
        return ckpt

    def _save_final_model(self, completed_epoch):
        checkpoint = self._build_checkpoint(int(completed_epoch) - 1)
        if self.is_main:
            destination = self.run_dir / "final_model.pth"
            atomic_torch_save(checkpoint, destination)
            write_checkpoint_pointer(self.run_dir, destination)

    def _upload_best_model_artifact(self, epoch, val_loss):
        import wandb
        artifact = wandb.Artifact(
            name="vqgan-best-model",
            type="model",
            metadata={"epoch": epoch + 1, "val_l1_loss": round(float(val_loss), 6)},
        )
        artifact.add_file(str(self.run_dir / "best_model.pth"))
        self.wandb_run.log_artifact(artifact)
