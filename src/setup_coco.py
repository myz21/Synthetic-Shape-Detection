"""
Download and extract the required MS COCO 2017 files for this project.

Usage:
    python -m src.setup_coco

By default this script creates the following structure:

data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


COCO_FILES = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = 100.0 * downloaded / total_size
        sys.stdout.write(
            f"\rDownloading {destination.name}: {downloaded / (1024 ** 2):7.1f} MB"
            f" / {total_size / (1024 ** 2):7.1f} MB ({percent:5.1f}%)"
        )
        sys.stdout.flush()

    print(f"Starting download: {url}")
    urllib.request.urlretrieve(url, destination, reporthook=_reporthook)
    sys.stdout.write("\n")


def _extract(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def setup_coco(data_root: str | Path = "data/coco", keep_archives: bool = False) -> Path:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    downloads_dir = root / "_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in COCO_FILES.items():
        archive_path = downloads_dir / filename
        if archive_path.exists():
            print(f"Archive already exists, skipping download: {archive_path}")
        else:
            _download(url, archive_path)

        _extract(archive_path, root)

    expected = [
        root / "train2017",
        root / "val2017",
        root / "annotations" / "instances_train2017.json",
        root / "annotations" / "instances_val2017.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "COCO setup finished, but some required files are still missing:\n"
            + "\n".join(missing)
        )

    if not keep_archives:
        shutil.rmtree(downloads_dir, ignore_errors=True)

    print("\nCOCO setup completed successfully.")
    print(f"Data root: {root.resolve()}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the required COCO 2017 files.")
    parser.add_argument(
        "--data-root",
        default="data/coco",
        help="Target directory for the COCO dataset. Default: data/coco",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded zip files after extraction.",
    )
    args = parser.parse_args()

    setup_coco(data_root=args.data_root, keep_archives=args.keep_archives)


if __name__ == "__main__":
    main()
