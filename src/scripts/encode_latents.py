"""
Encode 4D CMR volumes into spatiotemporal VQ-GAN latents.

The VQ-GAN is a 3D model operating on individual 2D+t slices (H, W, T).
Each full CMR volume (D, H, W, T) is encoded slice-by-slice along the depth
axis and the resulting latents are stacked to form the 4D latent tensor
(C, D, H/f, W/f, T/ft) saved as a .pt file.
"""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,
    CenterSpatialCropd,
    SpatialPadd,
)
from omegaconf import OmegaConf

from src.models.vqvae import VQVAE
from src.data.dataloading import PermuteDimensionsd, CyclicPadTimed, UnsqueezeChanneld  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Encode 4D CMR volumes into VQ-GAN latents")
    parser.add_argument("--csv", required=True, help="CSV with 'image' column (paths to 4D .nii.gz files)")
    parser.add_argument("--output_dir", required=True, help="Directory to store .pt latent files")
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to VQ-GAN checkpoint")
    parser.add_argument("--config", required=True, help="Path to VQ-GAN config yaml")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[224, 224, 32],
                        metavar=("H", "W", "T"), help="Crop size for each 2D+t slice")
    parser.add_argument("--target_frames", type=int, default=32,
                        help="Temporal frames after cyclic padding")
    parser.add_argument("--target_z", type=int, default=10,
                        help="Z slices after center crop + black padding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dim_perm", type=int, nargs="+", default=[1, 2, 3, 0],
        metavar="I",
        help="Permutation applied to the 4-D tensor (T,H,W,D) produced by EnsureChannelFirstd. "
             "Default (1,2,3,0) reorders to (H,W,D,T). A channel dim is then inserted automatically. "
             "Override only if your NIfTI axis order differs.",
    )
    return parser.parse_args()


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = VQVAE(**config.model.params)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    skipped = [k for k in state_dict if k not in model_keys]
    if skipped:
        print(f"Skipped {len(skipped)} keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    model.load_state_dict(filtered, strict=False)
    return model.eval().requires_grad_(False).to(device)


def build_transforms(roi_size, target_frames, target_z=10, dim_perm=(1, 2, 3, 0)):
    """
    Preprocessing for a full 4D CMR volume.

    After LoadImaged + EnsureChannelFirstd the default CMR tensor is
    expected to be (T, H, W, D). ``dim_perm`` reorders it to (H, W, D, T)
    before a channel dimension is inserted.
    """
    H, W, T = roi_size
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),                              # (T,H,W,D)
        PermuteDimensionsd(keys=["image"], perm=tuple(dim_perm)),         # (H,W,D,T)
        UnsqueezeChanneld(keys=["image"], dim=0),                         # (1,H,W,D,T)
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        CyclicPadTimed(keys=["image"], target_frames=target_frames),      # pad/crop T
        CenterSpatialCropd(keys=["image"], roi_size=(H, W, target_z, -1)),  # crop H,W,D
        SpatialPadd(keys=["image"], spatial_size=(H, W, target_z, -1), constant_values=-1.0),  # pad H,W,D
        ToTensord(keys=["image"]),
    ])


@torch.no_grad()
def encode_volume(model, volume_4d, device):
    """
    Encode a full (1, H, W, D, T) CMR volume slice-by-slice along D.

    Returns a tensor of shape (C, D, h, w, t) where h=H/f, w=W/f, t=T/ft.
    """
    # volume_4d: (1, H, W, D, T)
    D = volume_4d.shape[3]
    latents = []

    for d in range(D):
        # Extract slice: (1, H, W, T) -> add batch -> (1, 1, H, W, T)
        slc = volume_4d[:, :, :, d, :].unsqueeze(0).to(device)
        z = model.encode_stage_2_inputs(slc, quantized=True)  # (1, C, h, w, t)
        latents.append(z[0].cpu())  # (C, h, w, t)

    return torch.stack(latents, dim=1)  # (C, D, h, w, t)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.config, args.vqvae_ckpt, device)
    transforms = build_transforms(args.roi_size, args.target_frames, args.target_z, args.dim_perm)

    df = pd.read_csv(args.csv)
    image_paths = [str(row["image"]) for _, row in df.iterrows()]

    output_rows = []
    for img_path in image_paths:
        stem = Path(img_path).stem.replace(".nii", "")
        out_path = output_dir / f"{stem}.pt"

        if out_path.exists():
            print(f"Skipping {out_path} (exists)")
        else:
            data = transforms({"image": img_path})
            volume = data["image"].unsqueeze(0)          # (1, 1, H, W, D, T) or similar
            # Rearrange to (1, 1, H, W, D, T) if needed; adjust for your data layout
            latent = encode_volume(model, volume[0], device)   # (C, D, h, w, t)
            torch.save(latent, out_path)
            print(f"Saved {out_path}  shape={list(latent.shape)}")

        output_rows.append(str(out_path))

    out_csv = output_dir / "latents.csv"
    pd.DataFrame({"image": output_rows}).to_csv(out_csv, index=False)
    print(f"\nSaved CSV -> {out_csv}  ({len(output_rows)} entries)")


if __name__ == "__main__":
    main()
