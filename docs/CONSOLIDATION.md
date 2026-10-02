# Consolidation provenance and validation

Consolidated on 2026-10-02 into the existing `Cardio-AI/cardiodit` repository, based on public `main` commit `abf3c97a010735d27a4f38dccf24eba440a800bb`. One shared `src.models` package implements the model families. Source trees were inspected as working files, including modified, untracked and ignored source/config files; copying only Git HEAD would have lost fixes.

## Source map

| Source snapshot | Contribution | Destination |
| --- | --- | --- |
| Public CardioDiT, `abf3c97` | Original model lineage, license and paper configurations | Existing repository history; original compact config files expanded at their existing paths |
| CardioDiT local development, `2d850a036c5e4db2b535d589557be45fd3d05535` plus working changes | Earlier diffusion/FM, self-conditioning, positional and sampling work | Superseded core included through evolution; local config differences normalize to identical configurations after path cleanup |
| CardioDiT MNM2 evolution, `7eb9716f988f6fe3fa95896de115f8e790d27fc7` plus working changes | Current VQ-GAN, transformer, objectives, checkpointing, latent contracts, sampling and Stage-1 evaluation | `src/`, `configs/dit`, `configs/stage1`, `configs/transformer`, reusable `scripts/` |
| CardioDiT-TempAlign, `170c05ba3e28fa74b789713f52b3c2ef2cc705cc` plus working changes | Image-space alignment, descriptor/DTW tools, temporal ablations, L1/custom-beta losses, export options | Shared utilities/scripts, `encode_aligned_latents.py`, `configs/temporal_alignment`, `configs/stage1/tempalign_best.yaml`, `scripts/temporal_alignment` |
| Evolution resolved retraining configurations, including uncommitted edits | Exact 300k-update historical retraining variants | `configs/experiments/retrain_300k/` |

The evolution working changes to `src/models/ddpmscheduler.py`, `src/scripts/sample_dit.py`, `src/utils/sample_integrity.py`, and `scripts/sample_evolution_dit_models.py` are included. This preserves the zero-terminal-SNR cumulative-alpha ratio fix and the newer sampling/completion implementation. Both untracked `patch2x2_t5` flow configurations are retained. Names do not establish numerical settings: some historical `patch2x2_t1` filenames actually configure a temporal patch of two.

Machine-specific HPC packaging/submission wrappers, datasets, CSV manifests, generated figures, logs, checkpoints, output documents and developer tests are excluded from this repository. The local `prepare_mnm2_evolution.py` source remains for experiment/configuration reproducibility, with explicit external roots and the current Python executable. Existing source snapshots and excluded artifacts are preserved by the consolidation archive; this repository is not their only surviving copy. Original source file contents and `/mnt/sds` were preserved unchanged. Legacy local checkouts are retained in the external consolidation archive.

## Integration changes

- Preserved current evolution model/trainer/sampler behavior; reused TempAlign's unique preprocessing without replacing the newer shared models.
- Added explicit `interpolated_rescaled` positional mode to preserve TempAlign's fixed training-time temporal range; evolution's `interpolated` semantics remain available.
- Mapped historical L1/L2/Huber loss names explicitly, retained `huber_beta`, and rejected conflicting aliases. Checkpoints record the selected objective and beta.
- Preserved HWD export, flips and foreground crop through the main sampler while correctly transforming NIfTI affines. These transformations are explicit options and part of completion identity.
- Fixed mixed all-valid/padded latent bucket collation by always returning a temporal loss mask.
- Made requested image lengths round up to a valid temporal patch shape and trimmed decoded outputs to the requested frame count. Maximum model shape remains enforced.
- Updated the scale-factor utility to accept both contracted and explicitly historical tensor latents, and resolve CSV paths relative to their manifest.
- Moved auxiliary default output/cache/log roots outside the checkout; removed author-specific paths/entities; configurations default to offline W&B and the quick start disables it.
- Expanded the public compact paper configuration schema using the public loader's original defaults. This preserves architectural/numerical choices rather than silently substituting the newer experiment configurations.

## Local queue wrappers

`scripts/run_new_dit_synthesis_local.sh` retains the evolution source's local
checkpoint queue, process locking, per-run batch sizes, per-GPU workers, and
completion/resume policy. Set `CARDIODIT_RUNS_DIR`, `GPU_IDS`, and optionally
`PYTHON_BIN`, `RUNS_ROOT`, `OUTPUT_ROOT`, `N_SAMPLES`, or `RECHECK_ALL` before
invocation. `DRY_RUN=1` delegates only dry-run sampler commands and does not
require GPU access. Its historical queue selects checkpoints every 50,000
updates and family-specific batch sizes; review those choices for your model.

`scripts/run_decoder_quant_ablation_queue.sh` retains optional upstream-marker
waiting before paired decoder-quantization sampling. Set `UPSTREAM_LOG` and
`UPSTREAM_MARKER` to enable waiting; an empty log path starts immediately.
`STAGE1_CKPT`, `PYTHON`, `DEVICES`, `OUTPUT_ROOT`, `POLL_SECONDS` and sampling
counts are configurable. `DRY_RUN=1` prints the command without waiting or
sampling. Both wrappers keep outputs outside the checkout and preserve caller
CUDA visibility. They do not choose which occupied GPUs may be used.

## Validation performed

Validation used an existing local Python 3.10/PyTorch 2.10/MONAI 1.5.2 environment without changing it. Temporary checks and artifacts remain outside Git.

- Evolution CPU regression suite: **181 passed, 1 skipped**. Two copied test expectations were updated locally for deliberate consolidation behavior: Smooth-L1 is now a supported TempAlign alias; checkpoint defaults now resolve below the external runtime root. The original missing-CWD config test was rerun from the checkout.
- Existing Stage-1-to-DiT end-to-end test: **passed**, including tiny codec/model updates, encoding, checkpoint save/resume and sampling in a clean process.
- TempAlign temporal/descriptor checks: **40 passed**; numerical MSE/L1/custom-Huber alias and conflict checks passed.
- Synthetic integration: **16 cases passed on CPU and on one NVIDIA RTX 4090 GPU**. Covered diffusion and FM, self-conditioning enabled/disabled, four positional modes, two temporal lengths, finite forward/backward, real parameter updates, sampling, and tiny VQ-VAE encoding/decoding. These are functionality checks, not training-quality measurements.
- Targeted consolidation checks cover mixed-validity collation, HWD/flip/crop world-coordinate preservation, image-length rounding, and rescaled positional endpoints. A moving-foreground export regression verifies that full cine and individual frames share one crop, spatial shape, affine, and voxel values in both output layouts.
- Python syntax, YAML interpolation, shell syntax and delivery inventory checks passed. No datasets, CSVs, checkpoints, generated outputs or personal machine runtime paths are included. Both local queue wrappers passed shell syntax and external-directory dry-run checks; no sampling jobs were launched by those checks.

The historical cluster-packaging tests were not run against the omitted machine-specific packager. The two-GPU NCCL release gate was not run; the available GPU check used one otherwise idle GPU. Installing the declared dependency ranges in a fresh environment and long scientific training/evaluation were outside this source-consolidation validation.

## Remaining scientific and compatibility boundaries

- Retained features/configurations do not establish better generation quality, convergence, motion realism or clinical validity. Real datasets/weights were not loaded for these checks.
- Different VQ-GANs have different channels, spatial/temporal strides and quantization semantics. A matching config/checkpoint/latent contract is required; architectural resemblance is not compatibility evidence.
- Temporal loss masks do not make attention padding-invariant. The inherited model does not propagate support masks through global attention, and padded depth handling/statistics need their own scientific decision.
- Historical TempAlign's integrated interpolation encoder writes tensor-only latents and skips existing outputs without complete provenance validation. Its normalization/cropping order differs from preprocessing NIfTI before the contracted encoder. Both workflows are preserved and explicitly distinguished.
- Several historical configurations exceed practical attention memory or combine rejected normalization/scale conventions; they remain experiment records. Set resource limits deliberately and use remediated configs for the documented diffusion workflow.
- Historical DDIM endpoint choices, descriptor/preprocessing identity assumptions and normalization weighting remain inherited limitations for dedicated scientific review. The consolidation does not silently claim these resolved.
- FID/FVD/phase/temporal-consistency evaluators and motion-field interpolation from TempAlign are unfinished scaffolds; see `temporal_alignment.md`.
- Operational config edits change config hashes. Existing checkpoints require explicit identity review; `--allow_checkpoint_mismatch` is an audited override, not permission to ignore tensor/model incompatibility. Saved positional modes and preprocessing policies must still match.
