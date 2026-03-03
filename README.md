## Small Underwater Objects 3D Point Cloud (SOUP) Dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18844894.svg)](https://doi.org/10.5281/zenodo.18844894)

This repository provides the **object recognition benchmark code and label files** used to evaluate the **SUOP dataset** (Zenodo DOI: 10.5281/zenodo.18475883).

## Overview

* **Purpose**: Provide the exact **code + labels** used for benchmarking object detection/recognition on the SUOP dataset.
* **Included**:

  * **PointNet++**: 3D point cloud-based object recognition code.
  * **YOLOv8 (Ultralytics)**: 2D image-based detection code and the corresponding **bounding-box label (.txt) files** used for training. 
* **Not included**:

  * Raw dataset files and generated images are not distributed here. Images can be reproduced using the same generation pipeline used during benchmarking (e.g., `png_make.py`) if the SUOP dataset is available locally.

## What this repository is for

This repository is intended to help others **reproduce the benchmark results** (training/inference pipeline) on the SUOP dataset using the provided implementation and label

## 📁 Dataset Structure
```
object_detection/
├── PointNet++/
│   └── object_detection_code/
│       ├── model.py
│       ├── dataset.py
│       ├── train.py
│       └── object_detection.py
├── YOLOv8/
│   ├── object_detection_code/
│   │   ├── data.yaml
│   │   ├── ply_change.py
│   │   ├── png_make.py
│   │   ├── train.py
│   │   └── object_detection.py
│   └── bbox_labels/
│       ├── chair/
│       │   ├── chair_range_3m/ (case_XXX.txt ...)
│       │   ├── chair_range_6m/ (case_XXX.txt ...)
│       │   └── chair_range_10m/ (case_XXX.txt ...)
│       ├── drum/ ...
│       ├── dummy/ ...
│       ├── net/ ...
│       └── tire/ ...
```
