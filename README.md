# Synthetic Shape Detection

Course project for `CENG428 Neural Networks`.

This repository uses `MS COCO 2017` as the natural-image background dataset and adds synthetic shapes on top of COCO images. The current main workflow is notebook-based and lives in [notebook.ipynb](/home/neo/Desktop/CENG%203.%20SENE%20HEPS%C4%B0/CENG428%20NEURAL%20NETWORKS/Synthetic-Shape-Detection/notebook.ipynb).

## Repository Contents

- [notebook.ipynb](/home/neo/Desktop/CENG%203.%20SENE%20HEPS%C4%B0/CENG428%20NEURAL%20NETWORKS/Synthetic-Shape-Detection/notebook.ipynb): main development notebook
- [train.py](/home/neo/Desktop/CENG%203.%20SENE%20HEPS%C4%B0/CENG428%20NEURAL%20NETWORKS/Synthetic-Shape-Detection/train.py): training pipeline skeleton
- [src/dataset.py](/home/neo/Desktop/CENG%203.%20SENE%20HEPS%C4%B0/CENG428%20NEURAL%20NETWORKS/Synthetic-Shape-Detection/src/dataset.py): dataset utilities
- [ASSIGNMENT.md](/home/neo/Desktop/CENG%203.%20SENE%20HEPS%C4%B0/CENG428%20NEURAL%20NETWORKS/Synthetic-Shape-Detection/ASSIGNMENT.md): preserved assignment text
- [REPORT.md](/home/neo/Desktop/CENG%203.%20SENE%20HEPS%C4%B0/CENG428%20NEURAL%20NETWORKS/Synthetic-Shape-Detection/REPORT.md): draft report template

## Expected Data Layout

The COCO dataset is not included in this repository. Place it in the following structure:

```text
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

## Environment

Python `3.10+` is expected.

Main dependencies are defined in [pyproject.toml](/home/neo/Desktop/CENG%203.%20SENE%20HEPS%C4%B0/CENG428%20NEURAL%20NETWORKS/Synthetic-Shape-Detection/pyproject.toml):

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `Pillow`
- `opencv-python`
- `pycocotools`
- `tqdm`
- `jupyter`

## How To Run

If you use `uv`:

```bash
uv sync
uv run jupyter notebook
```

If you use `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
jupyter notebook
```

Then open [notebook.ipynb](/home/neo/Desktop/CENG%203.%20SENE%20HEPS%C4%B0/CENG428%20NEURAL%20NETWORKS/Synthetic-Shape-Detection/notebook.ipynb) and run the cells in order.

## Current Workflow

The notebook is organized around the assignment steps:

1. Load `CocoDetection` bases from COCO.
2. Build fixed train, validation, and test splits.
3. Generate synthetic shapes on COCO backgrounds.
4. Create detection targets automatically.
5. Train a CNN-based object detector.
6. Compare against a simple baseline.
7. Evaluate with detection metrics such as `Precision@0.5`, `Recall@0.5`, `F1@0.5`, and mean IoU.
8. Visualize predictions and analyze failure cases.

## Planned Experiments

The current notebook includes placeholders for these experiments:

1. High opacity vs low opacity
2. Easy shapes vs hard shapes
3. Small training set vs larger training set
4. With data augmentation vs without data augmentation

## Outputs

During development and evaluation, results may be saved under:

- `results/` for metrics
- `figures/` for generated examples and prediction visualizations

## Notes

- Do not submit the COCO dataset itself.
- Validation and test synthetic shapes should be generated deterministically using the assignment seed rule.
- The main solution is intended to be `object detection`, not classification-only.
