# SCOPE Protocol Notes

This file records the protocol assumptions used by the public SCOPE release.

## Core Rule

The SCOPE selector uses support-set diagnostics only. It must not read validation/test accuracy, best-accuracy tables, oracle routes, or result files when selecting a route.

## VTAB-1K Protocol

- Backbone: ViT-B/16 initialized from the same ImageNet-21k checkpoint family.
- Data split: official VTAB-1K split converted to the loader used by the training script.
- Normalization: the same normalization must be used across all compared methods.
- Training policy: the same epoch budget, optimizer family, augmentation policy, and data split should be used for all compared methods.
- Route selection: support diagnostics are computed before SCOPE training and do not use validation/test metrics.

## FGVC Protocol

- Datasets: CUB-200-2011, NABirds, Oxford Flowers-102, Stanford Cars, and Stanford Dogs.
- Split and augmentation: the same split and augmentation protocol should be used across all compared methods.
- Route selection: frozen support diagnostics or a pre-registered held-out rule should be used.

## Not Included

This release does not include datasets, pretrained checkpoints, third-party baseline repositories, full raw training logs, or intermediate checkpoints. Those artifacts are large and/or governed by separate licenses.
