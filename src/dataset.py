"""
src/dataset.py
──────────────
COCO base initialization, fixed split protocol, and dataset wrapper.
"""

import hashlib
import random
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


# ──────────────────────────────────────────────────────────────────
# Constants — given by the assignment
# ──────────────────────────────────────────────────────────────────
GLOBAL_SEED          = 2025
TRAIN_SIZE           = 5000   # first 5000 images from train2017
VAL_SIZE             = 1000   # first 1000 images from val2017
TEST_SIZE            = 1000   # next  1000 images from val2017 (indices 1000–1999)
POSITIVE_RATIO       = 0.70   # 70% positive, 30% negative
MAX_SHAPES_PER_IMAGE = 3      # 1 to 3 shapes per positive image


# ──────────────────────────────────────────────────────────────────
# Deterministic seed — given by the assignment
# Do NOT use Python's built-in hash(); output varies between sessions.
# ──────────────────────────────────────────────────────────────────
def make_seed(split_name: str, image_id: int, global_seed: int = GLOBAL_SEED) -> int:
    key = f"{split_name}_{image_id}_{global_seed}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


# ──────────────────────────────────────────────────────────────────
# COCO base initialization — given by the assignment
#
# Required directory structure:
#   data/coco/
#   ├── train2017/
#   ├── val2017/
#   └── annotations/
#       ├── instances_train2017.json
#       └── instances_val2017.json
# ──────────────────────────────────────────────────────────────────
def build_coco_bases(data_root: str | Path = "data/coco"):
    root = Path(data_root)

    train_base = CocoDetection(
        root=str(root / "train2017"),
        annFile=str(root / "annotations" / "instances_train2017.json"),
    )

    val_base = CocoDetection(
        root=str(root / "val2017"),
        annFile=str(root / "annotations" / "instances_val2017.json"),
    )

    return train_base, val_base


# ──────────────────────────────────────────────────────────────────
# Fixed split protocol — given by the assignment
#   - Sort all image IDs in increasing order
#   - train : first 5000 from train2017
#   - val   : first 1000 from val2017
#   - test  : next  1000 from val2017  (do NOT use for model selection)
# ──────────────────────────────────────────────────────────────────
def get_split_ids(
    coco_base: CocoDetection,
    split: Literal["train", "val", "test"],
) -> list[int]:
    all_ids = sorted(coco_base.coco.getImgIds())

    if split == "train":
        return all_ids[:TRAIN_SIZE]
    elif split == "val":
        return all_ids[:VAL_SIZE]
    elif split == "test":
        return all_ids[VAL_SIZE : VAL_SIZE + TEST_SIZE]
    else:
        raise ValueError(f"Unknown split: {split!r}")


# ──────────────────────────────────────────────────────────────────
# Synthetic shape generator — implement this
#
# Shapes to support (§4): circles, rectangles, triangles, ellipses,
#   polygons, line segments, stars or simple icons
# Properties to vary (§4): type, location, size, color, opacity,
#   rotation, number of shapes per image
# At least 4 difficulty mechanisms required (§4):
#   random opacity, anti-aliased edges, blur, noise,
#   low-contrast colors, colors from local image stats,
#   partial transparency, overlapping shapes,
#   random resize/crop, hard negatives, distractor shapes
# ──────────────────────────────────────────────────────────────────
class ShapeGenerator:
    def generate(self, image, n_shapes: int, rng: random.Random):
        """
        Draw n_shapes synthetic shapes onto image.

        Args:
            image    : PIL.Image (RGB)
            n_shapes : number of shapes to draw (0 for negative images)
            rng      : seeded random.Random instance

        Returns:
            augmented_image : PIL.Image
            boxes           : list[list[float]]   [[x1,y1,x2,y2], ...]
            mask            : np.ndarray [H, W]   1=shape pixel, 0=background
        """
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper — implement this
#
# Must (§2):
#   1. Load a COCO image
#   2. Add one or more synthetic shapes
#   3. Generate the corresponding target label automatically
#   4. Return the modified image and generated target
#
# Target format — Option A: Object Detection (§5)
#   target = {
#       "boxes":       FloatTensor[N, 4]   # x_min, y_min, x_max, y_max
#       "labels":      LongTensor[N]        # all synthetic shapes share label 1
#       "image_id":    int
#       "is_positive": bool
#   }
#   If negative: boxes = zeros((0,4)), labels = zeros((0,))
#
# Target format — Option B: Semantic Segmentation (§5)
#   target = {
#       "mask":        LongTensor or FloatTensor[H, W]   # 1=shape, 0=background
#       "image_id":    int
#       "is_positive": bool
#   }
#   If negative: mask contains only zeros
# ──────────────────────────────────────────────────────────────────
class SyntheticShapeDataset(Dataset):
    def __init__(
        self,
        coco_base: CocoDetection,
        image_ids: list[int],
        split: Literal["train", "val", "test"],
        task: Literal["detection", "segmentation"],
        transform=None,
    ):
        self.coco_base = coco_base
        self.image_ids = image_ids
        self.split     = split
        self.task      = task
        self.transform = transform
        self.generator = ShapeGenerator()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        # Seed: random for train, deterministic for val/test (§3)
        if self.split == "train":
            rng = random.Random()
        else:
            rng = random.Random(make_seed(self.split, image_id))

        # Positive / negative decision — 70% positive, 30% negative (§4)
        is_positive = rng.random() < POSITIVE_RATIO
        n_shapes    = rng.randint(1, MAX_SHAPES_PER_IMAGE) if is_positive else 0

        # Step 1: Load a COCO image
        # Step 2: Add synthetic shapes
        # Step 3: Generate target label automatically
        # Step 4: Return (image, target)
        raise NotImplementedError
