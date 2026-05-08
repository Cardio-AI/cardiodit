import torch
from pathlib import Path
from collections import defaultdict
from torchvision.utils import make_grid
from torch import amp
import torch.nn.functional as F
from tqdm import tqdm


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

        self.n_epochs = config.training.n_epochs
        self.eval_freq = config.training.eval_freq

        self.max_adv_weight = config.losses.adv_weight
        self.perceptual_weight = config.losses.perceptual_weight
        self.adv_warmup_epochs = config.losses.adv_warmup

        self.scaler_g = amp.GradScaler("cuda")
        self.scaler_d = amp.GradScaler("cuda")

        self.start_epoch = start_epoch
        self.best_loss = best_loss
        self._global_step = 0

    # ==========================================================
    # PUBLIC TRAIN LOOP
    # ==========================================================

    def train(self):
        if self.is_main:
            val_loss = self._validate(self.start_epoch)
            print(f"epoch {self.start_epoch} initial val loss: {val_loss:.4f}")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        for epoch in range(self.start_epoch, self.n_epochs):

            if hasattr(self.train_loader, "sampler") and isinstance(
                self.train_loader.sampler, torch.utils.data.DistributedSampler
            ):
                self.train_loader.sampler.set_epoch(epoch)

            adv_weight = self._get_adv_weight(epoch)
            self._train_epoch(epoch, adv_weight)

            if self.scheduler_g:
                self.scheduler_g.step()
            if self.scheduler_d and adv_weight > 0:
                self.scheduler_d.step()

            if (epoch + 1) % self.eval_freq == 0 and self.is_main:
                val_loss = self._validate(epoch)
                print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
                self._save_checkpoint(epoch, val_loss)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

        if self.is_main:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            torch.save(raw_model.state_dict(), self.run_dir / "final_model.pth")

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

            # ---- Generator step ----
            self.optimizer_g.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda"):
                reconstruction, quantization_loss, indices = self.model(images)
                epoch_perplexity += quantizer.perplexity.item()
                epoch_used_codes += indices.unique().numel()
                num_batches += 1
                loss, losses = self.loss_fn.generator_loss(
                    self.discriminator, adv_weight, images, reconstruction, quantization_loss,
                )

            self.scaler_g.scale(loss).backward()
            self.scaler_g.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()

            # ---- Discriminator step ----
            d_losses = {}
            if adv_weight > 0:
                self.optimizer_d.zero_grad(set_to_none=True)
                with amp.autocast(device_type="cuda"):
                    d_loss, d_losses = self.loss_fn.discriminator_loss(
                        self.discriminator, adv_weight, images, reconstruction.detach(),
                    )
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()

            global_step = epoch * len(self.train_loader) + step
            self._global_step = global_step

            if self.is_main:
                logs = {**losses, **d_losses}
                logs["total_g_loss"] = loss.item()
                for k, v in logs.items():
                    self.writer_train.add_scalar(k, v, global_step)
                pbar.set_postfix({"loss": f"{losses['l1_loss']:.4f}"})

                if self.wandb_run:
                    wlogs = {f"train/{k}": v for k, v in logs.items()}
                    wlogs["train/lr_g"] = self.optimizer_g.param_groups[0]["lr"]
                    self.wandb_run.log(wlogs, step=global_step)

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

            with amp.autocast(device_type="cuda"):
                reconstruction, quantization_loss, indices = self.model(images)
                adv_weight = self._get_adv_weight(epoch)
                _, losses = self.loss_fn.generator_loss(
                    self.discriminator, adv_weight, images, reconstruction, quantization_loss,
                )

            epoch_perplexity += quantizer.perplexity.item()
            epoch_used_codes += indices.unique().numel()
            num_batches += 1

            bs = images.shape[0]
            for k, v in losses.items():
                total_losses[k] += v * bs
            n += bs

            if batch_idx == 0 and self.is_main:
                sample_images = images[:4].detach().cpu()
                sample_recons = reconstruction[:4].detach().cpu()

        tb_step = epoch * len(self.train_loader)
        for k in total_losses:
            total_losses[k] /= n
            if self.is_main:
                self.writer_val.add_scalar(k, total_losses[k], tb_step)

        if self.is_main and num_batches > 0:
            avg_perplexity = epoch_perplexity / num_batches
            avg_used_codes = epoch_used_codes / num_batches
            usage_ratio = avg_used_codes / quantizer.quantizer.num_embeddings
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
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_disc = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator

        ckpt = {
            "epoch": epoch + 1,
            "best_loss": self.best_loss,
            "model": raw_model.state_dict(),
            "discriminator": raw_disc.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }
        if hasattr(raw_model, "quantizer") and hasattr(raw_model.quantizer, "quantizer"):
            ckpt["ema_cluster_size"] = raw_model.quantizer.quantizer.ema_cluster_size
            ckpt["ema_w"] = raw_model.quantizer.quantizer.ema_w

        #torch.save(ckpt, self.run_dir / f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(ckpt, self.run_dir / "last_checkpoint.pth")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(ckpt, self.run_dir / "best_model.pth")
            print(f"New best val loss: {val_loss:.4f}")
            if self.wandb_run:
                self._upload_best_model_artifact(epoch, val_loss)

    def _upload_best_model_artifact(self, epoch, val_loss):
        import wandb
        artifact = wandb.Artifact(
            name="vqgan-best-model",
            type="model",
            metadata={"epoch": epoch + 1, "val_l1_loss": round(float(val_loss), 6)},
        )
        artifact.add_file(str(self.run_dir / "best_model.pth"))
        self.wandb_run.log_artifact(artifact)
