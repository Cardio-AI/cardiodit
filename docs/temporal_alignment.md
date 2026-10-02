# Temporal alignment experiments

The code from CardioDiT-TempAlign is available within this repository. It uses the
same `src.models` package as the main CardioDiT pipeline. Historical experiment
configurations are under `configs/temporal_alignment`; historical local queues
are under `scripts/temporal_alignment`. These configurations retain their original
scientific parameter values, including estimated latent scales. Recompute those
scales for your own dataset and checkpoint. They are experiment templates, not
recommended hyperparameters or evidence of improvement.

Set `CARDIODIT_RUNS_DIR` to an external directory. It defaults to
`~/CardioDiT_runs`. CSV manifests, descriptor sidecars, templates, latents,
checkpoints, logs, samples, and figures belong there or in another external path.
Pass your own input manifests and sidecar directories explicitly. No dataset,
checkpoint, descriptor sidecar, or CSV is distributed with the repository.

## Available implementations

- `src/utils/temporal_align.py`: cyclic repetition, linear interpolation, Fourier
  resampling, ED/MS/ES/PF/MD piecewise interpolation, and constrained descriptor
  DTW with optional keyframe anchoring and template refinement.
- `src/utils/descriptor_io.py`: validated one-dimensional `.pt`/`.npy` descriptor
  loading and supported sidecar layouts.
- `src/scripts/build_dtw_template.py`: training-set global or pathology-specific
  templates, including a global fallback for label-specific templates.
- `src/scripts/apply_dtw_preprocessing.py`: materialize DTW-aligned NIfTI images
  from a native-image manifest. Its output includes external image CSVs that can
  be passed to the main `encode_latents.py`.
- `src/scripts/export_preprocessing_examples.py`: export image-space comparisons,
  difference maps and GIFs for the implemented preprocessing methods.
- `src/scripts/plot_keyframe_distributions.py`, `plot_exp_losses.py`, and
  `make_temporal_alignment_figures.py`: analysis and illustration scripts.
  `plot_exp_losses.py --experiment NAME RUN_ID RUN_DIR` accepts your local W&B
  runs; repeat this argument for each experiment.

The retained `motionfield_warp` function and `eval_fid.py`, `eval_fvd.py`,
`eval_phase.py`, `eval_temporal_consistency.py`, and `eval_run.py` are unfinished
research scaffolds. Motion-field interpolation and the four evaluation metrics
raise `NotImplementedError`; they are not validated evaluation capabilities.
The evaluation orchestrator may generate samples before reaching those stubs.

## Encoding and latent provenance

Use `src/scripts/encode_latents.py` for the current self-describing latent format
and strict checkpoint/config loading. DTW preprocessing can be materialized first
with `apply_dtw_preprocessing.py`, then passed into this main encoder with an
explicit temporal policy. The contract identifies those aligned images as its
source; retain the external preprocessing manifest/templates alongside them.

`src/scripts/encode_aligned_latents.py` preserves the earlier integrated
normalize/crop-then-align workflow, including linear, Fourier, piecewise, global
DTW, and pathology-specific DTW modes. It writes **legacy bare-tensor latents**;
set `training.allow_legacy_latents: true` explicitly to consume them. That opt-in
is already declared in the historical temporal-alignment configs. The encoder
uses the shared strict Stage-1 loader, but its existing-file skip behavior remains
historical and does not validate provenance. Use a fresh output directory when
changing inputs, alignment options, or checkpoints.

The two encoding workflows are not numerically interchangeable: image intensity
normalization before interpolation can differ from normalization after
interpolation. This consolidation preserves both workflows without inventing
fractional temporal provenance for the current integer-index latent contract.
It does not establish compatibility with every historical DiT checkpoint.

## Training and export settings

The trainer accepts `training.objective_loss` or the older `training.loss_type`:
MSE/L2, L1/MAE, and Smooth-L1/Huber (`huber_legacy`). `training.huber_beta` controls
Smooth-L1 beta and is recorded in checkpoints. Conflicting loss names fail early.

The historical temporal interpolation experiment selects
`model.params.pos_embed_mode: interpolated_rescaled`, preserving the training
sequence's temporal positional range while changing the number of time tokens.
The main pipeline's existing `interpolated` mode remains available separately.
Variable-length configs use shape buckets and an explicit maximum shape; increase
`max_input_size` deliberately if your latent sequences exceed the template.

Sampling supports `--output_axes hwd`, explicit `--flip_axes`, and optional
`--foreground_crop` with threshold/fraction settings. `--spacing` always receives
D H W T values, including when exporting H W D arrays. The default main sampling
path keeps D H W arrays and does not crop or flip.

The queue scripts preserve particular historical run names/checkpoint epochs.
Set `PYTHON` and `CARDIODIT_RUNS_DIR`, inspect the configured paths and resource
requirements, and adapt their run lists before use. They launch work only when
explicitly invoked; they were not run during consolidation.
