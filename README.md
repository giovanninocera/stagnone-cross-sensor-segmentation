# Stagnone cross-sensor submerged-vegetation segmentation

[![Release verification](https://github.com/giovanninocera/stagnone-cross-sensor-segmentation/actions/workflows/verify.yml/badge.svg)](https://github.com/giovanninocera/stagnone-cross-sensor-segmentation/actions/workflows/verify.yml)

Code, frozen configurations, aggregate results, and provenance records supporting the manuscript **“Cross-Sensor Segmentation of Optically Detectable Submerged Vegetation under Acquisition Heterogeneity”**.

This repository documents a leakage-controlled comparison of Random Forest, U-Net, and SegFormer across QuickBird-2 (2003), WorldView-2 (2016), and Pléiades-1A (2023) acquisitions of the Stagnone di Marsala lagoon. It also contains the aggregate outputs of the separate diagnostic-core and mapped-footprint structural-screening workflow.

## Scope

The public release supports:

1. inspection of the training, inference, evaluation, calibration, and sensitivity-analysis implementation;
2. verification of all distributed files through SHA-256 checksums;
3. reconstruction of the consolidated manuscript-facing numerical summary from aggregate CSV and JSON files;
4. inspection of the frozen configurations, split parameters, and provenance manifests.

It does **not** contain commercial satellite pixels or coordinate-bearing geospatial products and therefore cannot perform end-to-end retraining without separately licensed inputs. See [`LIMITATIONS_AND_EXCLUSIONS.md`](LIMITATIONS_AND_EXCLUSIONS.md).

## Quick verification

Only Python 3.9 or newer is required for the portable checks:

```bash
python scripts/verify_release.py
python scripts/reproduce_summary.py --check
```

To write the consolidated numerical summary:

```bash
python scripts/reproduce_summary.py --output reproduced_summary.json
```

The expected output is [`expected_summary.json`](expected_summary.json).

## Frozen headline results

| Result | Frozen value |
|---|---:|
| Common unique-pixel test support | 2,005,445 pixels |
| SegFormer pooled minimum-IoU | 0.9564 +/- 0.0057 |
| U-Net pooled minimum-IoU | 0.8777 +/- 0.0285 |
| Random Forest pooled minimum-IoU | 0.7475 +/- 0.1088 |
| Selected 2023 structural objects | 598 |
| Selected 2023 mapped support | 48.9456 ha |

The structural product is an exploratory screening/context layer. It is **not** a calibrated hydraulic-roughness coefficient, geomorphic-change map, species map, or biomass estimate.

## Repository layout

- `configs/`: frozen model and structural-screening configurations.
- `documentation/`: concise methodological and result documentation.
- `environment/`: pinned full-pipeline Python environment.
- `manifests/current/`: sanitized provenance and result manifests. Local absolute paths are replaced by symbolic roots.
- `scripts/reproduce_summary.py`: dependency-free numerical summary builder.
- `scripts/verify_release.py`: integrity, licensing-boundary, and secret/path audit.
- `scripts/pipeline_reference/`: research pipeline source used for the analyses. Licensed inputs are required for full execution.
- `tables/`: aggregate model, calibration, temporal-control, sensitivity, and structural-screening results.
- `INVENTORY.csv` and `SHA256SUMS.txt`: release inventory and checksums.

## Data and licensing boundary

QuickBird-2, WorldView-2, and Pléiades-1A imagery is subject to third-party licensing and is not redistributed. The release also excludes imagery-derived patches, masks, raster predictions, geospatial vectors, field coordinates, model checkpoints, and QGIS projects. The commercial imagery must be obtained from the respective providers under the applicable licence.

Source code is licensed under the MIT License. Aggregate tables, documentation, and repository-authored metadata are licensed under CC BY 4.0. Third-party rights remain unaffected; see [`NOTICE.md`](NOTICE.md).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A version-specific Zenodo DOI will be added after the GitHub release is archived.

## Authors

- Giovanni Andrea Nocera
- Antonio Mederos-Barrera
- Dionisio Rodríguez-Esparragón
- Giuseppe Ciraolo
- Antonino Maltese

## Release

Version `1.0.0`, prepared on 2026-09-30. Repository: https://github.com/giovanninocera/stagnone-cross-sensor-segmentation
