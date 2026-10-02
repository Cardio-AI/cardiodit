# CardioDiT

Official implementation of [CardioDiT: Latent Diffusion Transformers for 4D Cardiac MRI Synthesis](https://arxiv.org/abs/2603.25194), with the subsequent development features consolidated into this repository.

Stage 1 applies a 2D+t VQ-GAN independently to anatomical depth slices. Stage 2 models the assembled `(C,D,H,W,T)` latent with one 4D transformer. The code includes diffusion and flow matching, self-conditioning, Q/K normalization, anisotropic positional embeddings, temporal/4D RoPE, variable-length shape buckets, EMA, complete checkpoint resume, and DDPM/DDIM/DPM-Solver++/flow sampling. Experimental temporal preprocessing is described in [docs/temporal_alignment.md](docs/temporal_alignment.md).

## Installation

Use Python 3.10 or newer in a dedicated environment. Install the PyTorch and torchvision builds appropriate for your CPU/CUDA system, then install the runtime requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, `conda env create -f environment.yml` creates the environment. PyTorch SDPA is supported; xformers is optional. The requirement ranges describe the imported dependency surface, not a tested promise for every version combination. The consolidation checks used Python 3.10, PyTorch 2.10 and MONAI 1.5.2. Keep scientific run configurations and package versions with your outputs.

## External data and outputs

Keep datasets, manifests, latent tensors, checkpoints, generated images, metrics and caches **outside this checkout**. Set one external root; auxiliary tools default to `~/CardioDiT_runs` when it is unset:

```bash
export CARDIODIT_RUNS_DIR="$HOME/CardioDiT_runs"
export WANDB_MODE=disabled
export WANDB_DIR="$CARDIODIT_RUNS_DIR/wandb"
export XDG_CACHE_HOME="$CARDIODIT_RUNS_DIR/cache"
export MPLCONFIGDIR="$CARDIODIT_RUNS_DIR/cache/matplotlib"
mkdir -p "$CARDIODIT_RUNS_DIR" "$WANDB_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
```

`$CARDIODIT_RUNS_DIR/manifests/train.csv` and `val.csv` must contain an `image` column. Paths may be absolute or relative to the manifest. Supply your own access-authorized cine NIfTI files and splits; no patient data, CSV manifests or weights are distributed here. Verify axes, qform/sform, crop coverage and acquisition timing for your data. Spatial and temporal dimensions are not interchangeable.

The scientific configuration registry is [configs/advertised_runs.yaml](configs/advertised_runs.yaml). It lists the evolution source's remediated VQ-GAN and diffusion configurations. Other configurations preserve historical experiments; they are not all recommended settings for new runs. The compact paper configs have been expanded into the current schema without changing their architecture. Historical epoch schedules require the explicit training flag `--allow_legacy_epoch_schedule`.

## Train and encode Stage 1

Choose a VQ-GAN family before encoding. This example uses spatial and temporal downsampling with cyclic padding to a divisible length. The native-time `ds4xy_noT` and `ds8xy_noT` families preserve temporal length while still using temporal convolutions.

```bash
export S1=S1_ds4_all_dims_paddiv_v2_remediated
python src/scripts/train_vqgan.py \
  --config "configs/stage1/$S1.yaml" \
  --training_ids "$CARDIODIT_RUNS_DIR/manifests/train.csv" \
  --validation_ids "$CARDIODIT_RUNS_DIR/manifests/val.csv" \
  --cache_dir "$CARDIODIT_RUNS_DIR/cache/$S1" \
  --output_dir "$CARDIODIT_RUNS_DIR/outputs/stage1" --run_name "$S1"

for split in train val; do
  python src/scripts/encode_latents.py \
    --csv "$CARDIODIT_RUNS_DIR/manifests/$split.csv" \
    --output_dir "$CARDIODIT_RUNS_DIR/latents/$S1/$split" \
    --vqvae_ckpt "$CARDIODIT_RUNS_DIR/outputs/stage1/$S1/final_model.pth" \
    --config "configs/stage1/$S1.yaml" --device cuda
done
```

For local multi-GPU training, replace `python` with `torchrun --standalone --nproc_per_node=2`. Source code supports CPU execution; large production configurations are intended for GPUs. Stage-1 DDP requires the config's `model.params.ddp_sync: true` to synchronize the EMA codebook.

The main encoder records Stage-1/config/source identities, geometry, temporal/depth maps, validity masks, quantization and scaling in each latent file. Re-encoding verifies existing files; replacing stale files requires `--force`. Tensor-only historical latents require `training.allow_legacy_latents: true` and must not be mixed with contracted latents.

## Train Stage 2

```bash
export DIT=S1_ds4_all_dims_paddiv_ddpm_rope4d_selfcond_v2_remediated
python src/scripts/train_dit.py \
  --config "configs/dit/native_padded/$DIT.yaml" \
  --training_ids "$CARDIODIT_RUNS_DIR/latents/$S1/train/latents.csv" \
  --validation_ids "$CARDIODIT_RUNS_DIR/latents/$S1/val/latents.csv" \
  --stage1_cfg "configs/stage1/$S1.yaml" \
  --stage1_ckpt "$CARDIODIT_RUNS_DIR/outputs/stage1/$S1/final_model.pth" \
  --output_dir "$CARDIODIT_RUNS_DIR/outputs/dit" --run_name "$DIT"
```

Reusing the same output directory/run name resumes from its complete checkpoint. Remediated configurations use optimizer-update counts and channel normalization with global scale 1.0. The saved state includes optimizer, scheduler, scaler, EMA, random state and mid-epoch position.

Flow-matching configurations are under `configs/dit/fixed/` and `configs/dit/native_padded/`; the older controlled ablations are under `configs/transformer/test_configs/flow_matching/`. Select a configuration whose latent shape, codec, normalization and temporal policy match your encoded data. These preserve their original objective choices, including historical Huber loss; choosing MSE is a scientific experiment, not an automatic conversion. Self-conditioning is enabled by matching the model and training flags. Historical configurations combining normalization with a nonunit scale are intentionally rejected for new runs.

## Sample and evaluate

```bash
python src/scripts/sample_dit.py \
  --stage1_cfg "configs/stage1/$S1.yaml" \
  --stage1_ckpt "$CARDIODIT_RUNS_DIR/outputs/stage1/$S1/final_model.pth" \
  --diff_cfg "configs/dit/native_padded/$DIT.yaml" \
  --diff_ckpt "$CARDIODIT_RUNS_DIR/outputs/dit/$DIT/last_checkpoint.json" \
  --weights ema --scheduler auto --timesteps 300 --seed 42 \
  --decoder_mode both --geometry canonical_synthetic --spacing 10 1.7 1.7 1 \
  --output_dir "$CARDIODIT_RUNS_DIR/samples/$DIT"
```

`auto` selects the objective's sampler. `--decoder_mode both` reports direct and codebook-quantized decoding separately. Independent sample seeds, input hashes, sampling settings and completion manifests are saved with outputs. Native variable-length generation needs `variable_shape`, a sufficient `max_input_size`, and a compatible positional mode. `--T_image N --vqgan_temporal_stride S` rounds the latent length to the temporal patch size and trims decoded outputs to exactly N frames; it does not extend the trained maximum.

Output spacing is always supplied in `(D,H,W,T)` order. Optional `--output_axes hwd`, `--flip_axes` and `--foreground_crop` preserve world coordinates through the corresponding affine transformation. Canonical synthetic geometry does not establish patient geometry. Template geometry requires an explicit matching template.

Stage-1 reconstruction evaluation is available through `src/scripts/evaluate_stage1_vqgan.py`; comparison/plotting and scale-factor utilities are also included. The TempAlign FID/FVD/phase/temporal-consistency scripts remain clearly marked research skeletons and are not validated generation metrics.

See [docs/CONSOLIDATION.md](docs/CONSOLIDATION.md) for source provenance, compatibility boundaries, validation and remaining limitations. Retained configurations and feature code are not evidence of model quality or checkpoint interchangeability.
