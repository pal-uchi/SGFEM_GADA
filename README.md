# SGFEM_GADA
[Improved Head and Data Augmentation to Reduce Artifacts at Grid Boundaries in Object Detection](https://search.ieice.org/bin/summary.php?id=e107-d_1_115)

## Introduction
We found that when the class score of the same detected object fluctuates periodically and drops abruptly when the detection is performed while shifting the image in the One-stage method.
This phenomenon did not occur with the Two-stage method.
We propose the following two methods to address this phenomenon.
- Sub-Grid Feature Extraction Module (SGFEM)
- Grid-Aware Data Augmentation (GADA)

<img src=https://github.com/pal-uchi/SGFEM_GADA/assets/29569950/1478cb89-ab93-40bb-b0f6-badc0d1ed102
 width=300>


## Implementation
This repository contains only the custom parts from [FCOS](https://arxiv.org/abs/1904.01355) implemented by [mmdetection](https://github.com/open-mmlab/mmdetection).
