"""
src/experiments.py
──────────────────
Hocanın istediği en az 3 anlamlı deney (README §10).

Her deney için:
    - Sayısal sonuçlar rapor edilmeli
    - Kısa yorum / yorum yazılmalı (sadece sayı yeterli değil!)

Deney seçenekleri (README §10):
    1.  Scratch vs fine-tune
    2.  Küçük vs büyük training seti
    3.  Kolay vs zor sentetik şekiller
    4.  Yüksek vs düşük opacity şekiller
    5.  Augmentation var vs yok
    6.  Farklı CNN mimarileri
    7.  Farklı input çözünürlükleri
    8.  Detection vs segmentation
    9.  Shallow baseline vs derin CNN
    10. Görsel başına şekil sayısının etkisi
"""

# @ TODO: Hangi 3 (veya daha fazla) deneyi yapacağını seç.
#         Her deney için aşağıdaki şablonu kullan.


def run_experiment_1():
    """
    Deney 1: Training from Scratch vs Fine-tuning (README §10.1)

    Hipotez: Pretrained bir backbone (ImageNet) ile başlamak,
             sıfırdan eğitime göre daha iyi performans sağlamalıdır.

    Yapılacaklar:
        A) pretrained=False  → train_one_epoch → evaluate → sonuç_A
        B) pretrained=True   → train_one_epoch → evaluate → sonuç_B
        Karşılaştır ve yorumla.

    # @ TODO: train.py → build_model(pretrained=False) ve build_model(pretrained=True)
    #         ile iki ayrı çalıştırma yap, sonuçları tablo olarak göster.
    """
    raise NotImplementedError


def run_experiment_2():
    """
    Deney 2: Küçük vs Büyük Training Seti (README §10.2)

    Hipotez: Daha fazla eğitim verisi modelin genelleme yeteneğini artırır.

    Yapılacaklar:
        A) train_size = 1000  → eğit → değerlendir → sonuç_A
        B) train_size = 5000  → eğit → değerlendir → sonuç_B
        (C) train_size = tüm train2017  → isteğe bağlı, güçlü GPU gerektirir

    # @ TODO: get_split_ids yerine kendi slice'ınla farklı boyutlarda
    #         train seti oluştur.
    """
    raise NotImplementedError


def run_experiment_3():
    """
    Deney 3: Kolay vs Zor Sentetik Şekiller (README §10.3)

    Hipotez: Düşük opacity, gürültü ve anti-aliasing ile oluşturulan
             zor şekiller modeli daha fazla zorlar.

    Yapılacaklar:
        A) ShapeGenerator(difficulty="easy")  → high opacity, no blur, no noise
        B) ShapeGenerator(difficulty="hard")  → low opacity, blur, noise, low contrast
        Her ikisi için aynı model ile eğit ve değerlendir.

    # @ TODO: ShapeGenerator'a difficulty parametresi ekle,
    #         iki farklı config ile eğitim ve değerlendirme yap.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────
# Sonuç tablosu yazdırıcı – implement edildi
# ─────────────────────────────────────────────

def print_experiment_table(results: list[dict]):
    """
    results = [
        {"name": "Scratch", "precision": 0.72, "recall": 0.68, "f1": 0.70, "mIoU": 0.61},
        {"name": "Fine-tune", "precision": 0.85, "recall": 0.81, "f1": 0.83, "mIoU": 0.77},
    ]
    """
    header = ["Experiment", "Precision", "Recall", "F1", "mIoU"]
    row_fmt = "{:<20} {:<12} {:<10} {:<8} {:<8}"
    print(row_fmt.format(*header))
    print("-" * 60)
    for r in results:
        print(row_fmt.format(
            r.get("name", "?"),
            f"{r.get('precision', 0):.4f}",
            f"{r.get('recall', 0):.4f}",
            f"{r.get('f1', 0):.4f}",
            f"{r.get('mIoU', 0):.4f}",
        ))
