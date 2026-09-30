# Final scene ablation v3 - common-support results

Completed: 4 July 2026

## Evaluation design

- Seven branches: three M1, three M2 and one M3.
- Five seeds per branch: 123, 231, 312, 423 and 531.
- Total calibrated SegFormer checkpoints: 35.
- Common deterministic support: 207 windows, 69 per scene.
- Window selection is independent of the target class.
- Evaluation uses unique global pixels after averaging overlapping-window
  probabilities.
- Cross-split audit: zero overlaps against 29,752 train/validation windows.
- Unique labelled pixels:
  - 2003: 768,261;
  - 2016: 541,175;
  - 2023: 696,009.
- Primary metric: minimum IoU between VEG and NONVEG.
- Uncertainty: five-seed dispersion and 5,000-draw paired hierarchical
  bootstrap over seeds and 256-pixel spatial cells.

The metrics quantify agreement with supervised image-derived references. They
are not independent ecological accuracy estimates.

## Pooled results across all three scenes

| Branch | Mean minimum IoU | Seed SD |
|---|---:|---:|
| M3-2003/2016/2023 | **0.9564** | **0.0057** |
| M2-2003/2023 | 0.8968 | 0.0590 |
| M1-2023 | 0.8785 | 0.0066 |
| M2-2003/2016 | 0.8331 | 0.0495 |
| M2-2016/2023 | 0.8325 | 0.0171 |
| M1-2016 | 0.7779 | 0.0196 |
| M1-2003 | 0.5038 | 0.0308 |

M3 is both the strongest and the most stable model on the pooled support.

## M3 by scene

| Scene | Mean minimum IoU | Seed SD |
|---|---:|---:|
| 2003 QuickBird | 0.9533 | 0.0179 |
| 2016 WorldView-2 | 0.9404 | 0.0068 |
| 2023 Pleiades | 0.9715 | 0.0072 |
| All scenes | **0.9564** | **0.0057** |

## Does M3 sacrifice within-scene specialization?

The paired bootstrap compares M3 with the M1 specialist trained on the
corresponding scene.

| Scene | Comparison | Mean delta in minimum IoU | 95% interval |
|---|---|---:|---:|
| 2003 | M3 minus M1-2003 | -0.0097 | -0.0403 to 0.0248 |
| 2016 | M3 minus M1-2016 | -0.0003 | -0.0223 to 0.0252 |
| 2023 | M3 minus M1-2023 | -0.0068 | -0.0160 to 0.0000 |

M3 essentially retains specialist performance on 2003 and 2016. The
2023-only model has a small advantage on its own image-derived reference, as
expected for a specialist trained and evaluated on the same scene family.

## Paired pooled advantage of M3

| Competitor | Mean delta in minimum IoU | 95% interval | P(delta > 0) |
|---|---:|---:|---:|
| M1-2003 | +0.4567 | +0.4044 to +0.5077 | 1.000 |
| M1-2016 | +0.1802 | +0.1485 to +0.2174 | 1.000 |
| M1-2023 | +0.0787 | +0.0628 to +0.0965 | 1.000 |
| M2-2003/2016 | +0.1232 | +0.0700 to +0.1788 | 1.000 |
| M2-2003/2023 | +0.0602 | +0.0232 to +0.1187 | 1.000 |
| M2-2016/2023 | +0.1251 | +0.1005 to +0.1520 | 1.000 |

All pooled intervals are entirely above zero.

## Scientific interpretation

1. The three-scene design is justified by cross-scene robustness, not by
   claiming that M3 must outperform a specialist on that specialist's own
   acquisition.
2. Single-scene models transfer poorly, especially M1-2003.
3. Two-scene models improve transfer but remain less stable or leave one
   acquisition underrepresented.
4. M3 is the only branch that maintains high agreement on all three sensors
   with low seed variability.
5. The 2023 value remains reference agreement. Independent field observations
   must be reported separately.

## Release decision

M3-2003/2016/2023 with five seeds is the selected SegFormer configuration for
full-scene inference and downstream products, subject to:

- corrected RF and U-Net comparisons;
- full-scene seam and grid-shift checks;
- calibration and field-point audits;
- downstream roughness sensitivity analysis.
