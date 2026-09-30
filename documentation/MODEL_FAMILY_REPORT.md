# Final five-seed model-family comparison

All models use the corrected v3 M3 data allocation and the same frozen,
deduplicated spatial support. Thresholds are selected only on validation.

| Model | Mean minimum IoU | Seed SD |
|---|---:|---:|
| SegFormer MiT-B2 | 0.9564 | 0.0057 |
| U-Net ResNet-34 | 0.8777 | 0.0285 |
| Random Forest | 0.7475 | 0.1088 |

## Paired spatial bootstrap

- SegFormer minus U-Net ResNet-34: +0.0796 (95% CI +0.0518 to +0.1116).
- SegFormer minus Random Forest: +0.2105 (95% CI +0.1221 to +0.3108).

These values quantify agreement with image-derived references, not
independent ecological accuracy.
