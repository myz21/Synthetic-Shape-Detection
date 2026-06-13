"""
train.py
────────
Training entry point.

Hocanın istediği zorunlu bileşenler (README §7):
    ✓ PyTorch Dataset
    ✓ PyTorch DataLoader
    ✓ Loss function
    ✓ Optimizer
    ✓ Learning rate
    ✓ Fixed epochs / stopping criterion
    ✓ Validation-set monitoring
    ✓ Saved quantitative results
    ✓ Prediction visualizations

Raporlamak zorunda olduğun training detayları (README §7):
    input_size, batch_size, epochs, optimizer, lr,
    loss_fn, pretrained, hardware, train_time
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticShapeDataset, build_coco_bases, get_split_ids

# ─────────────────────────────────────────────
# Hocanın istediği training parametreleri
# (bunları raporda açıkça belirtmen gerekiyor)
# ─────────────────────────────────────────────
CONFIG = {
    "input_size":   (224, 224),   # @ TODO: dene – 320x320, 416x416, vs.
    "batch_size":   8,            # @ TODO: GPU belleğine göre ayarla
    "num_epochs":   10,           # @ TODO: erken durdurma kriterine göre değiştir
    "lr":           1e-4,         # @ TODO: scheduler ekle (StepLR, CosineAnnealing, …)
    "optimizer":    "Adam",       # @ TODO: SGD dene – deney §10.3
    "pretrained":   True,         # @ TODO: False dene – Deney 1 (scratch vs fine-tune)
    "task":         "detection",  # @ TODO: "segmentation" dene – Deney 8
    "data_root":    "data/coco",
    "results_dir":  "results",
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}


def get_transforms(split: str):
    """
    Veri artırma (data augmentation) transform'ları.

    # @ TODO: torchvision.transforms veya albumentations kullan.
    #         Deney 5: augmentation var vs. yok.
    #         En azından:
    #           - Resize(input_size)
    #           - ToTensor()
    #           - Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    """
    # @ TODO: gerçek transform pipeline döndür
    import torchvision.transforms as T
    if split == "train":
        return T.Compose([
            T.Resize(CONFIG["input_size"]),
            T.ToTensor(),
            # @ TODO: augmentations ekle (RandomHorizontalFlip, ColorJitter, …)
        ])
    else:
        return T.Compose([
            T.Resize(CONFIG["input_size"]),
            T.ToTensor(),
        ])


def build_model(task: str, pretrained: bool):
    """
    Hocanın önerdiği model seçenekleri (README §6):

    Detection:
        - Faster R-CNN  ← torchvision.models.detection.fasterrcnn_resnet50_fpn
        - SSD           ← torchvision.models.detection.ssd300_vgg16
        - YOLO          ← ultralytics kütüphanesi (pyproject.toml'a ekle)
        - Custom CNN    ← senin mimarin

    Segmentation:
        - U-Net         ← segmentation_models_pytorch (pyproject.toml'a ekle)
        - FCN           ← torchvision.models.segmentation.fcn_resnet50
        - DeepLab       ← torchvision.models.segmentation.deeplabv3_resnet50
        - Custom CNN    ← senin mimarin

    # @ TODO: bir model seç ve aşağıya implement et.
    #         Pretrained kısımları ve kendi eğittiğin kısımları raporda belirt.
    """
    # @ TODO: modeli yükle / oluştur ve döndür
    raise NotImplementedError("build_model: modeli implement et")


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Tek epoch training döngüsü.

    # @ TODO:
    #   for images, targets in loader:
    #       images  = images.to(device)
    #       targets = [{k: v.to(device) for k,v in t.items() if hasattr(v,'to')} ...]
    #       loss = loss_fn(model(images), targets)
    #       optimizer.zero_grad(); loss.backward(); optimizer.step()
    #   return epoch_loss
    """
    raise NotImplementedError


def evaluate(model, loader, device):
    """
    Hocanın istediği metrikler (README §9):

    Detection → Precision@0.5, Recall@0.5, F1@0.5, mean IoU
    Segmentation → foreground IoU, Dice, Precision, Recall

    # @ TODO: torchmetrics veya kendi hesaplama fonksiyonunu yaz.
    #         Test setinde asla model seçimi yapma (README §7).
    """
    raise NotImplementedError


def save_results(metrics: dict, epoch: int):
    """Hocanın istediği: sonuçları results/metrics.json'a kaydet."""
    import json, time
    out = Path(CONFIG["results_dir"]) / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    history = json.loads(out.read_text()) if out.exists() else []
    history.append({"epoch": epoch, "timestamp": time.time(), **metrics})
    out.write_text(json.dumps(history, indent=2))


def main():
    device = CONFIG["device"]
    print(f"Device: {device}")

    # ── COCO dataset'lerini yükle (hocanın verdiği yapı) ─────────────────
    train_base, val_base = build_coco_bases(CONFIG["data_root"])

    # ── Hocanın verdiği sabit split protokolü ────────────────────────────
    train_ids = get_split_ids(train_base, "train")   # ilk 5000
    val_ids   = get_split_ids(val_base,   "val")     # ilk 1000
    test_ids  = get_split_ids(val_base,   "test")    # sonraki 1000

    # ── Dataset & DataLoader ─────────────────────────────────────────────
    train_ds = SyntheticShapeDataset(train_base, train_ids, "train", CONFIG["task"], get_transforms("train"))
    val_ds   = SyntheticShapeDataset(val_base,   val_ids,   "val",   CONFIG["task"], get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(CONFIG["task"], CONFIG["pretrained"]).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # @ TODO: farklı optimizer'ları dene (Deney §10.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # @ TODO: learning rate scheduler ekle (isteğe bağlı ama önerilir)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ── Loss fonksiyonu ───────────────────────────────────────────────────
    # @ TODO: göreve uygun loss seç:
    #   Detection     → Faster R-CNN torchvision'dan zaten loss döndürür
    #   Segmentation  → nn.BCEWithLogitsLoss() veya nn.CrossEntropyLoss()
    loss_fn = None  # @ TODO: implement et

    # ── Training döngüsü ─────────────────────────────────────────────────
    best_val_metric = 0.0

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}/{CONFIG['num_epochs']}  loss={train_loss:.4f}  val={val_metrics}")
        save_results({"train_loss": train_loss, **val_metrics}, epoch)

        # @ TODO: en iyi modeli kaydet (test setini kullanma!)
        # if val_metrics["f1"] > best_val_metric:
        #     best_val_metric = val_metrics["f1"]
        #     torch.save(model.state_dict(), "results/best_model.pth")


if __name__ == "__main__":
    main()
