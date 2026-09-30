# Pipeline reference

This directory is a frozen, sanitized snapshot of the research pipeline used to generate the distributed aggregate results. It is included for methodological inspection and requires the separately licensed imagery, masks, labels, pretrained backbones, and computing environment for an end-to-end rerun.

## Main entry points

- `scripts/patches/make_patches.py`: patch construction and spatial split planning.
- `scripts/analysis/audit_patch_split_overlap.py`: reciprocal footprint-buffer and overlap audit.
- `scripts/train/train_segformer.py`, `train_suite.py`, and `train_tabular.py`: SegFormer, U-Net, and Random Forest training.
- `scripts/analysis/evaluate_model_family_unique_pixels.py`: common unique-pixel model-family evaluation.
- `scripts/analysis/evaluate_final_ablation_unique_pixels.py`: acquisition-domain ablation evaluation.
- `scripts/analysis/run_m3_v3_full_scene_inference.py`: full-scene inference orchestration.
- `scripts/analysis/temporal_comparability_controls.py`: persistent-pixel and relative-shift controls.
- `scripts/analysis/hybrid_core_footprint_pipeline.py`: diagnostic-core and mapped-footprint structural scenario.
- `scripts/reporting/build_final_release_manifest.py`: frozen result manifest assembly.

## Paths and environment

Private absolute paths were replaced with `${PROJECT_ROOT}`, `${LICENSED_DATA_ROOT}`, and `${LICENSED_EXPORT_ROOT}`. Configure equivalent local roots before attempting a licensed rerun. The captured Python environment is in `../../environment/requirements_full_pipeline.txt`.

The portable repository checks do not execute this full pipeline; they validate the release boundary and rebuild the consolidated numerical summary from the distributed aggregate tables.
