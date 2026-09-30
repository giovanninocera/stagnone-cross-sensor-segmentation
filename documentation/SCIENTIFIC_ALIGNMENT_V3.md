# Scientific alignment v3 - non-structural audit

Generated from frozen project artifacts. The structural pipeline and manuscript
DOCX were not modified.

## Decisions

- Final M3 patch table: 6,000 patches; 3,240 train, 1,001 validation and 1,759 test.
- Frozen test support: 2,005,445 unique labeled pixels in 207 windows.
- Complete M1/M2/M3 matrix: seven branches, five seeds each, all on identical support.
- Ensemble calibration: recomputed on held-out unique pixels; the previous full-reference calibration CSV is not valid as a test estimate.
- Field cross-check: REMOVE_FROM_INFERENTIAL_MANUSCRIPT; optional descriptive supplement only.
- QuickBird 2003: the April 2006 PIF radiometric anchor must be disclosed.
- Scene-exclusion wording: label-held-out with target-scene normalization, not true sensor/date hold-out.

## Machine-verifiable outputs

- `patch_branch_counts.csv`
- `patch_m3_by_scene.csv`
- `frozen_test_support.csv`
- `ablation_pooled.csv`
- `ablation_m3_by_scene.csv`
- `ablation_m3_pairwise_bootstrap.csv`
- `ablation_scene_label_holdout.csv`
- `calibration_all_seeds_heldout.csv`
- `calibration_ensemble_heldout.csv`
- `calibration_reliability_bins.csv`
- `calibration_spatial_bootstrap.csv`
- `calibration_reliability_heldout.png`
- `field_crosscheck_final_product.csv`
- `field_crosscheck_summary.csv`
- `methods_verified_parameters.json`
- `MANUSCRIPT_READY_TEXT.md`

## Unresolved author-supplied information

- reference-mask interpreter identities and blinding protocol
- a signed/immutable raw record for the claimed 10 August 2023 field survey
- independent empirical co-registration RMSE for the three final acquisitions
