# Limitations and exclusions

## Deliberately excluded material

This public release contains no:

- commercial QuickBird-2, WorldView-2, or Pléiades-1A source imagery;
- RGB/NIR crops, thumbnails, quicklooks, figures, or rendered maps carrying source pixels;
- source or derived rasters, including reflectance, probability, class, mask, uncertainty, structural-index, or registration-diagnostic layers;
- vector products, object footprints, shapefiles, GeoPackages, GeoJSON, QGIS projects, or coordinate-bearing field tables;
- model weights, checkpoints, serialized estimators, patch stores, NumPy arrays, or cached tensors;
- manuscript, template, presentation, or submission files.

## Reproducibility boundary

The release reproduces and verifies the consolidated numerical summary from the distributed aggregate tables and manifests. Full preprocessing, training, inference, and pixel-level evaluation require the separately licensed acquisitions, masks and labels, pretrained-backbone access, the documented software environment, and suitable GPU resources.

The pipeline-reference scripts are provided for methodological inspection. Symbolic roots such as `${PROJECT_ROOT}`, `${LICENSED_DATA_ROOT}`, and `${LICENSED_EXPORT_ROOT}` mark inputs or workspaces that are not distributed.

The detailed field-point table is excluded because it contains coordinates and point-level predictions. Its distributed summary is descriptive only: the opportunistic survey is not treated as design-based validation.

## Scientific interpretation limits

- Cross-acquisition performance does not establish deployment-blind generalization to an unseen sensor or site.
- Relative-shift tests quantify sensitivity to positional disagreement; they are not measured physical displacement.
- The structural-context layer is an exploratory screening product, not a calibrated hydraulic or geomorphic quantity.
- The fixed 2003 geometry supports persistence screening and cannot detect newly formed, migrated, or expanded structures.
