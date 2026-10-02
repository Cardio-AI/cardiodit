import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import PatchAdversarialLoss, PerceptualLoss, JukeboxLoss


def _normalize_temporal_mask(mask: torch.Tensor | None, reference: torch.Tensor):
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=reference.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2 or mask.shape[-1] != reference.shape[-1]:
        raise ValueError(
            "temporal mask must have shape (B,T) matching inputs; "
            f"got {tuple(mask.shape)} for {tuple(reference.shape)}"
        )
    if mask.shape[0] == 1 and reference.shape[0] != 1:
        mask = mask.expand(reference.shape[0], -1)
    if mask.shape[0] != reference.shape[0]:
        raise ValueError("temporal mask batch dimension does not match inputs")
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("every sample must contain at least one valid temporal frame")
    return mask.bool()


def masked_l1_loss(
    reconstructions: torch.Tensor,
    images: torch.Tensor,
    temporal_mask: torch.Tensor | None,
) -> torch.Tensor:
    """L1 mean over valid temporal voxels only."""
    mask = _normalize_temporal_mask(temporal_mask, images)
    if mask is None:
        return F.l1_loss(reconstructions, images)
    view_shape = (images.shape[0],) + (1,) * (images.ndim - 2) + (images.shape[-1],)
    expanded = mask.view(view_shape).expand_as(images)
    return (reconstructions - images).abs().masked_select(expanded).mean()


def _masked_pair_loss(loss_module, predictions, targets, temporal_mask):
    """Run a volume loss on each sample after removing invalid frames."""
    mask = _normalize_temporal_mask(temporal_mask, targets)
    if mask is None:
        return loss_module(predictions.float(), targets.float()).mean()
    values = []
    for sample_idx in range(targets.shape[0]):
        valid = mask[sample_idx]
        values.append(
            loss_module(
                predictions[sample_idx:sample_idx + 1, ..., valid].float(),
                targets[sample_idx:sample_idx + 1, ..., valid].float(),
            ).mean()
        )
    return torch.stack(values).mean()


def _last_logits(discriminator_output):
    if isinstance(discriminator_output, (list, tuple)):
        return discriminator_output[-1]
    return discriminator_output


class VQGANLoss(nn.Module):
    """
    Combined loss for VQGAN training: reconstruction + perceptual + adversarial + jukebox.

    Call as:
        loss, logs = loss_fn.generator_loss(discriminator, adv_weight, images, recon, q_loss)
        loss, logs = loss_fn.discriminator_loss(discriminator, adv_weight, images, recon)
    """

    def __init__(
        self,
        perceptual_weight: float = 1.0,
        jukebox_weight: float = 1.0,
        patch_adv_params: dict | None = None,
        jukebox_params: dict | None = None,
        perceptual_params: dict | None = None,
    ):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.jukebox_weight = jukebox_weight

        # Initialize MONAI loss modules as submodules so .to(device) propagates.
        self.perceptual_loss = PerceptualLoss(**(perceptual_params or {}))
        self.adversarial_loss = PatchAdversarialLoss()
        self.jukebox_loss = JukeboxLoss(**(jukebox_params or {}))

    def generator_loss(
        self,
        discriminator,
        adv_weight,
        images: torch.Tensor,
        reconstructions: torch.Tensor,
        quantization_loss: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
    ):
        """
        Compute the generator loss combining:
        L1 reconstruction + perceptual + spectral + patch adversarial + jukebox loss
        """

        # Reconstruction L1
        l1 = masked_l1_loss(reconstructions, images, temporal_mask)

        # Perceptual and spectral losses operate on volumes with invalid frames
        # removed, so repeated cyclic padding cannot affect their reductions.
        p = _masked_pair_loss(
            self.perceptual_loss, reconstructions, images, temporal_mask
        )

        # Adversarial loss (patch-based)
        if adv_weight > 0:
            mask = _normalize_temporal_mask(temporal_mask, reconstructions)
            fake_batches = (
                [reconstructions]
                if mask is None
                else [
                    reconstructions[i:i + 1, ..., mask[i]]
                    for i in range(reconstructions.shape[0])
                ]
            )
            g_adv = torch.stack([
                self.adversarial_loss(
                    _last_logits(discriminator(fake.contiguous())),
                    target_is_real=True,
                    for_discriminator=False,
                ).mean()
                for fake in fake_batches
            ]).mean()
        else:
            g_adv = torch.zeros_like(l1)

        # Jukebox loss
        j_loss = (
            _masked_pair_loss(self.jukebox_loss, reconstructions, images, temporal_mask)
            if self.jukebox_weight > 0
            else torch.zeros_like(l1)
        )

        total_loss = (
            l1.float()
            + quantization_loss.float()
            + self.perceptual_weight * p
            + adv_weight * g_adv.float()
            + self.jukebox_weight * j_loss
        )

        logs = {
            "l1_loss": l1.mean(),
            "quantization_loss": quantization_loss.mean(),
            "perceptual_loss": p.mean(),
            "g_adv_loss": g_adv.mean(),
            "jukebox_loss": j_loss.mean(),
        }

        return total_loss.mean(), logs


    def discriminator_loss(
        self,
        discriminator,
        adv_weight,
        images: torch.Tensor,
        reconstructions: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
    ):
        """
        Patch adversarial discriminator loss for real and fake images
        """

        mask = _normalize_temporal_mask(temporal_mask, images)
        pairs = (
            [(images, reconstructions)]
            if mask is None
            else [
                (images[i:i + 1, ..., mask[i]], reconstructions[i:i + 1, ..., mask[i]])
                for i in range(images.shape[0])
            ]
        )
        per_sample = []
        for real, fake in pairs:
            logits_fake = _last_logits(discriminator(fake.detach().contiguous()))
            loss_fake = self.adversarial_loss(
                logits_fake, target_is_real=False, for_discriminator=True
            )
            logits_real = _last_logits(discriminator(real.detach().contiguous()))
            loss_real = self.adversarial_loss(
                logits_real, target_is_real=True, for_discriminator=True
            )
            per_sample.append(0.5 * (loss_fake + loss_real).mean())
        d_loss = torch.stack(per_sample).mean()
        return d_loss, {"d_loss": d_loss}
