"""
src/dataset.py
──────────────
CocoDetection wrapper + synthetic-shape augmentation.

Hocanın verdiği sabit parametreler:
    GLOBAL_SEED      = 2025
    TRAIN_SIZE       = 5000   (train2017'den ilk 5000 görsel)
    VAL_SIZE         = 1000   (val2017'den ilk 1000 görsel)
    TEST_SIZE        = 1000   (val2017'den sonraki 1000 görsel)
    Positive ratio   = %70, Negative ratio = %30
    Shapes per image = 1–3 (pozitif görseller için)
"""

import hashlib
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

# ─────────────────────────────────────────────
# Hocanın verdiği sabit değerler
# ─────────────────────────────────────────────
GLOBAL_SEED = 2025
TRAIN_SIZE  = 5000
VAL_SIZE    = 1000
TEST_SIZE   = 1000
POSITIVE_RATIO = 0.70   # %70 pozitif, %30 negatif
MAX_SHAPES_PER_IMAGE = 3


# ─────────────────────────────────────────────
# Hocanın verdiği deterministik seed fonksiyonu
# (Python hash() kullanılamaz – oturum başına değişir)
# ─────────────────────────────────────────────
def make_seed(split_name: str, image_id: int, global_seed: int = GLOBAL_SEED) -> int:
    """Reproduce the exact same synthetic shapes for val/test splits."""
    key = f"{split_name}_{image_id}_{global_seed}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


# ─────────────────────────────────────────────
# Hocanın verdiği COCO başlatma şablonu
# ─────────────────────────────────────────────
def build_coco_bases(data_root: str | Path = "data/coco"):
    """
    Returns (train_base, val_base) as torchvision CocoDetection instances.

    Expected directory layout (hocanın verdiği yapı):
        data/coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
    """
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


# ─────────────────────────────────────────────
# Hocanın verdiği sabit split protokolü
# ─────────────────────────────────────────────
def get_split_ids(coco_base: CocoDetection, split: Literal["train", "val", "test"]):
    """
    Image IDs are sorted in increasing order as required.

    - train : ilk TRAIN_SIZE (5000) görsel – train2017'den
    - val   : ilk VAL_SIZE  (1000) görsel – val2017'den
    - test  : sonraki TEST_SIZE (1000) görsel – val2017'den
    """
    all_ids = sorted(coco_base.coco.getImgIds())

    if split == "train":
        return all_ids[:TRAIN_SIZE]
    elif split == "val":
        return all_ids[:VAL_SIZE]
    elif split == "test":
        return all_ids[VAL_SIZE : VAL_SIZE + TEST_SIZE]
    else:
        raise ValueError(f"Unknown split: {split!r}")


# ─────────────────────────────────────────────
# Shape generator – senin implement etmen gereken kısım
# ─────────────────────────────────────────────
class ShapeGenerator:
    """
    Synthetic shape generator.

    Desteklenmesi gereken şekil türleri (README §4):
        circles, rectangles, triangles, ellipses, polygons,
        line segments, stars or simple icons

    Değiştirilmesi gereken özellikler (README §4):
        shape type, location, size, color, opacity,
        rotation (where applicable), number of shapes per image

    Zorluk mekanizmaları – en az 4 tanesi (README §4):
        random opacity, anti-aliased boundaries, random blur,
        additive noise, low-contrast colors, colors from local stats,
        partial transparency, overlapping shapes,
        random resizing/cropping, hard negatives, distractor shapes
    """

    # @ TODO: __init__ – zorluk mekanizmalarını parametre olarak al
    #         (örn. use_blur=True, use_noise=True, low_contrast=False, …)
    def __init__(self):
        pass

    # @ TODO: draw_circle(canvas, rng) → (canvas, bbox)
    #         PIL veya OpenCV ile çember çiz, bounding box döndür
    def draw_circle(self, canvas, rng: random.Random):
        raise NotImplementedError

    # @ TODO: draw_rectangle(canvas, rng) → (canvas, bbox)
    def draw_rectangle(self, canvas, rng: random.Random):
        raise NotImplementedError

    # @ TODO: draw_triangle(canvas, rng) → (canvas, bbox)
    def draw_triangle(self, canvas, rng: random.Random):
        raise NotImplementedError

    # @ TODO: draw_ellipse(canvas, rng) → (canvas, bbox)
    def draw_ellipse(self, canvas, rng: random.Random):
        raise NotImplementedError

    # @ TODO: draw_polygon(canvas, rng) → (canvas, bbox)
    def draw_polygon(self, canvas, rng: random.Random):
        raise NotImplementedError

    # @ TODO: draw_line(canvas, rng) → (canvas, bbox)
    def draw_line(self, canvas, rng: random.Random):
        raise NotImplementedError

    # @ TODO: draw_star(canvas, rng) → (canvas, bbox)
    def draw_star(self, canvas, rng: random.Random):
        raise NotImplementedError

    def generate(self, image, n_shapes: int, rng: random.Random):
        """
        Görsel üzerine n_shapes adet sentetik şekil çizer.

        Returns:
            augmented_image : PIL.Image veya np.ndarray
            boxes           : list[list[float]]  # [[x1,y1,x2,y2], …]
            mask            : np.ndarray[H,W]    # 1 = shape pixel, 0 = background

        # @ TODO: Her draw_* metodunu çağır, çıktıları birleştir.
        #         Zorluk mekanizmalarını burada uygula
        #         (blur, noise, opacity blending, …).
        """
        raise NotImplementedError


# ─────────────────────────────────────────────
# PyTorch Dataset wrapper – senin implement etmen gereken kısım
# ─────────────────────────────────────────────
class SyntheticShapeDataset(Dataset):
    """
    Hocanın istediği PyTorch Dataset wrapper.

    Her __getitem__ çağrısında:
        1. Bir COCO görseli yüklenir.             (implement edildi)
        2. Pozitif/negatif kararı verilir.        (implement edildi)
        3. Sentetik şekiller eklenir.             (# @ TODO)
        4. Label (bbox veya mask) üretilir.       (# @ TODO)
        5. (image, target) döndürülür.            (# @ TODO)
    """

    def __init__(
        self,
        coco_base: CocoDetection,
        image_ids: list[int],
        split: Literal["train", "val", "test"],
        task: Literal["detection", "segmentation"] = "detection",
        transform=None,
    ):
        self.coco_base   = coco_base
        self.image_ids   = image_ids
        self.split       = split
        self.task        = task
        self.transform   = transform

        # Hocanın verdiği pozitif/negatif oranı
        self.positive_ratio = POSITIVE_RATIO

        # @ TODO: ShapeGenerator örneği oluştur
        self.generator = ShapeGenerator()

        # Hocanın verdiği: train'de rastgele, val/test'te deterministik
        self._is_train = (split == "train")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        # ── 1. COCO görselini yükle ──────────────────────────────────────
        # Hocanın verdiği CocoDetection indeksi image_id ile eşleşmeyebilir;
        # coco.loadImgs kullanarak doğrudan yükle.
        img_info   = self.coco_base.coco.loadImgs(image_id)[0]
        img_path   = Path(self.coco_base.root) / img_info["file_name"]

        # @ TODO: img_path'i PIL ile aç, RGB'ye çevir
        #         image = Image.open(img_path).convert("RGB")
        image = None  # replace with actual load

        # ── 2. Pozitif / negatif kararı ─────────────────────────────────
        # Hocanın verdiği seed mantığı: train'de rastgele, val/test'te deterministik
        if self._is_train:
            rng = random.Random()   # her seferinde gerçekten rastgele
        else:
            seed = make_seed(self.split, image_id)
            rng  = random.Random(seed)

        is_positive = rng.random() < self.positive_ratio
        n_shapes    = rng.randint(1, MAX_SHAPES_PER_IMAGE) if is_positive else 0

        # ── 3. Sentetik şekilleri ekle ───────────────────────────────────
        # @ TODO: self.generator.generate(image, n_shapes, rng) çağır
        #         augmented_image, boxes, mask = self.generator.generate(...)
        augmented_image = image   # placeholder
        boxes           = []
        mask            = None

        # ── 4. Target dict oluştur ───────────────────────────────────────
        if self.task == "detection":
            # Hocanın verdiği hedef formatı (Option A):
            # boxes  : FloatTensor[N, 4]  (x_min, y_min, x_max, y_max)
            # labels : LongTensor[N]      (tüm sentetik şekiller label=1)

            # @ TODO: boxes listesini torch.FloatTensor'a çevir.
            #         Negatif görseller için empty tensor kullan:
            #         boxes  = torch.zeros((0, 4), dtype=torch.float32)
            #         labels = torch.zeros((0,),   dtype=torch.long)
            target = {
                "boxes":       torch.zeros((0, 4), dtype=torch.float32),  # @ TODO: gerçek boxes
                "labels":      torch.zeros((0,),   dtype=torch.long),      # @ TODO: gerçek labels
                "image_id":    image_id,
                "is_positive": is_positive,
            }

        elif self.task == "segmentation":
            # Hocanın verdiği hedef formatı (Option B):
            # mask : LongTensor veya FloatTensor[H, W]  (1=şekil, 0=arka plan)

            # @ TODO: mask'i torch.LongTensor'a çevir.
            #         Negatif görseller için sıfır mask kullan.
            target = {
                "mask":        torch.zeros((1, 1), dtype=torch.long),  # @ TODO: gerçek mask
                "image_id":    image_id,
                "is_positive": is_positive,
            }
        else:
            raise ValueError(f"Unknown task: {self.task!r}")

        # ── 5. Transform uygula ve döndür ────────────────────────────────
        # @ TODO: self.transform varsa augmented_image'e uygula
        if self.transform is not None:
            augmented_image = self.transform(augmented_image)

        return augmented_image, target
