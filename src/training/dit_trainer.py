from pathlib import Path

import torch
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm

from src.models.ema import EMA


class DiTTrainer:
    def __init__(
        self,
        model,
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
    ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = run_dir
        self.config = config
        self.is_main = is_main
        self.wandb_run = wandb_run

        self.n_epochs = int(config.training.n_epochs)
        self.eval_freq = int(config.training.eval_freq)
        self.scale_factor = float(config.training.get("scale_factor", 1.0))
        self.grad_accum_steps = max(1, int(config.training.get("grad_accum_steps", 1)))

        self.use_ema = bool(config.training.get("use_ema", True))
        self.ema_decay = float(config.training.get("ema_decay", 0.9999))
        if self.use_ema:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            self.ema = EMA(raw_model, decay=self.ema_decay)
        else:
            self.ema = None

        self.start_epoch = int(start_epoch)
        self.best_loss = best_loss
        self.global_step = self.start_epoch * len(self.train_loader)
        self.scaler = amp.GradScaler("cuda", enabled=self.device.type == "cuda")

    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            if hasattr(self.train_loader, "sampler") and isinstance(
                self.train_loader.sampler,
                torch.utils.data.DistributedSampler,
            ):
                self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (epoch + 1) % self.eval_freq == 0:
                if self.is_main:
                    val_loss = self._validate(epoch)
                    self._save_best_checkpoint(epoch, val_loss)
                    self._save_checkpoint(epoch)

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()

        if self.is_main:
            torch.save(self._get_model_state(), self.run_dir / "final_model.pth")

    def _train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.is_main)
        for step, batch in enumerate(pbar):
            latents = self._get_batch_tensor(batch) * self.scale_factor
            noise, timesteps, noisy_latents = self._sample_noisy_latents(latents)
            target = self._target(latents, noise, timesteps).detach()

            with amp.autocast(
                device_type=self.device.type, enabled=self.device.type == "cuda"
            ):
                pred = self.model(noisy_latents, t=timesteps, y=None)
                loss = self._loss(pred, target) / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            is_update_step = (
                (step + 1) % self.grad_accum_steps == 0
                or (step + 1) == len(self.train_loader)
            )
            if is_update_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None and self.is_main:
                    self.ema.update()

            display_loss = loss.item() * self.grad_accum_steps
            if self.is_main:
                pbar.set_postfix({"loss": f"{display_loss:.4f}"})
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "train/loss": display_loss,
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self.global_step,
                    )

            self.global_step += 1

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        try:
            total_loss = 0.0
            total_count = 0

            for batch in self.val_loader:
                latents = self._get_batch_tensor(batch) * self.scale_factor
                noise, timesteps, noisy_latents = self._sample_noisy_latents(latents)
                target = self._target(latents, noise, timesteps).detach()

                with amp.autocast(
                    device_type=self.device.type, enabled=self.device.type == "cuda"
                ):
                    pred = self.model(noisy_latents, t=timesteps, y=None)
                    loss = self._loss(pred, target)

                batch_size = latents.shape[0]
                total_loss += loss.item() * batch_size
                total_count += batch_size

            val_loss = total_loss / max(total_count, 1)
            print(f"Validation Loss: {val_loss:.6f}")

            if self.wandb_run is not None:
                self.wandb_run.log({"val/loss": val_loss}, step=self.global_step)
            return val_loss
        finally:
            if self.ema is not None:
                self.ema.restore()

    def _get_batch_tensor(self, batch):
        x = batch["image"]
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        return x.to(self.device, non_blocking=True)

    def _sample_noisy_latents(self, latents):
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(
            original_samples=latents,
            noise=noise,
            timesteps=timesteps,
        )
        return noise, timesteps, noisy_latents

    def _target(self, latents, noise, timesteps):
        if self.scheduler.prediction_type == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)
        if self.scheduler.prediction_type == "epsilon":
            return noise
        raise ValueError(f"Unsupported prediction type: {self.scheduler.prediction_type}")

    def _loss(self, pred, target):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if raw_model.learn_sigma:
            pred, log_sigma = torch.chunk(pred, 2, dim=1)
            log_sigma = torch.clamp(log_sigma, min=-20.0, max=5.0)
            return (
                torch.exp(-log_sigma) * (pred.float() - target.float()) ** 2
                + log_sigma
            ).mean()
        return F.smooth_l1_loss(pred.float(), target.float())

    def load_ema_state(self, state_dict):
        if self.ema is not None and state_dict is not None:
            self.ema.load_state_dict(state_dict)

    def _save_checkpoint(self, epoch):
        torch.save(self._build_checkpoint(epoch), self.run_dir / "last_checkpoint.pth")

    def _save_best_checkpoint(self, epoch, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(self._build_checkpoint(epoch), self.run_dir / "best_model.pth")
            print(f"New best val loss: {val_loss:.6f}")

    def _build_checkpoint(self, epoch):
        return {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "model": self._get_model_state(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
            if self.lr_scheduler is not None
            else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def _get_model_state(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        return raw_model.state_dict()
