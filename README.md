# Synthetic Shape Detection

Course project for `CENG428 Neural Networks`.

This repository uses `MS COCO 2017` as the natural-image background dataset and adds synthetic shapes on top of COCO images. The current main workflow is notebook-based and lives in [notebook.ipynb](notebook.ipynb).

For submission-oriented runs, the project also includes a 3-notebook split:

- [01_data_creation.ipynb](01_data_creation.ipynb)
- [02_training.ipynb](02_training.ipynb)
- [03_testing.ipynb](03_testing.ipynb)

## Repository Contents

- [notebook.ipynb](notebook.ipynb): main development notebook
- [01_data_creation.ipynb](01_data_creation.ipynb): dataset/synthetic generation notebook
- [02_training.ipynb](02_training.ipynb): training notebook
- [03_testing.ipynb](03_testing.ipynb): testing/evaluation notebook
- [train.py](train.py): training pipeline skeleton
- [src/dataset.py](src/dataset.py): dataset utilities
- [ASSIGNMENT.md](ASSIGNMENT.md): preserved assignment text
- [REPORT.md](REPORT.md): draft report template
- [links.txt](links.txt): placeholder file for public dataset/model links

## Expected Data Layout

The COCO dataset is not included in this repository. After setup, the project expects the following structure:

```text
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

## COCO Setup

This repository includes a helper script that downloads the required COCO 2017 files and creates the expected directory structure automatically.

If you use `uv`:

```bash
uv run python -m src.setup_coco
```

If you use `pip`:

```bash
python -m src.setup_coco
```

This script downloads:

- `train2017.zip`
- `val2017.zip`
- `annotations_trainval2017.zip`

and extracts them into `data/coco/`.

## Environment

Python `3.10+` is expected.

Main dependencies are defined in [pyproject.toml](pyproject.toml):

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
uv run python -m src.setup_coco
uv run jupyter notebook
```

If you use `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m src.setup_coco
jupyter notebook
```

Then open [notebook.ipynb](notebook.ipynb) and run the cells in order.

For the split submission workflow, run:

1. [01_data_creation.ipynb](01_data_creation.ipynb)
2. [02_training.ipynb](02_training.ipynb)
3. [03_testing.ipynb](03_testing.ipynb)

The split notebooks keep [notebook.ipynb](notebook.ipynb) intact as the all-in-one version.

## Kaggle Note

The split notebooks include a Kaggle setup cell that tries to detect an attached COCO dataset and create a local symlink:

```text
data/coco -> /kaggle/input/.../coco2017
```

The current direct path used during development was:

```text
/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017
```

If Kaggle finds that dataset, the notebooks can keep using the same project-relative `data/coco` path without manual path rewrites.

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
