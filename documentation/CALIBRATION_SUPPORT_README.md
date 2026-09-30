# Calibration support correction

`calibration_metrics_heldout_unique_pixels.csv` is the authoritative calibration table for manuscript inference. It evaluates the final five-seed probability mean on the frozen union of 207 class-independent held-out windows, scoring each of 2,005,445 labeled global pixels once.

The legacy `calibration_metrics.csv` sampled all labeled pixels in the full reference rasters. Its values are descriptive full-reference diagnostics and must not be reported as held-out/test calibration.
