"""
Prepare the scene-classification dataset.

The professor provides the raw images in dataset/dataset/ with two sub-folders:
    dataset/dataset/training/<class>/   (2 985 images)
    dataset/dataset/validation/<class>/ (1 500 images)

This script:
    1. Merges both folders into data/raw/<class>/  (one flat ImageFolder).
    2. Splits the merged data 70 / 15 / 15 into data/processed/{train,val,test}.

Usage:
    python scripts/download_data.py            # merge + split (default)
    python scripts/download_data.py --split-only  # only re-split data/raw → data/processed
"""
import argparse
import shutil
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.dataset import split_dataset

# Location of the professor's original dataset
DATASET_DIR = PROJECT_ROOT / "dataset" / "dataset"


def merge_dataset_folders():
    """
    Merge dataset/dataset/training/ and dataset/dataset/validation/
    into data/raw/<class>/, preserving the original class folder names.
    """
    source_dirs = [DATASET_DIR / "training", DATASET_DIR / "validation"]

    for src in source_dirs:
        if not src.exists():
            raise FileNotFoundError(
                f"Expected source folder not found: {src}\n"
                "Make sure the professor's dataset is placed in dataset/dataset/ "
                "with training/ and validation/ sub-folders."
            )

    # Clean previous raw data to avoid stale files
    if RAW_DATA_DIR.exists():
        shutil.rmtree(RAW_DATA_DIR)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for src in source_dirs:
        class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            dest_class = RAW_DATA_DIR / class_dir.name
            dest_class.mkdir(parents=True, exist_ok=True)
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            ]
            for img in images:
                shutil.copy2(img, dest_class / img.name)
                total_copied += 1

    print(f"Merged {total_copied} images into {RAW_DATA_DIR}")
    for d in sorted(RAW_DATA_DIR.iterdir()):
        if d.is_dir():
            count = len([f for f in d.iterdir() if f.is_file()])
            print(f"  {d.name}: {count} images")

    return total_copied


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset (merge + split)")
    parser.add_argument(
        "--split-only", action="store_true",
        help="Skip merge, only re-split data/raw → data/processed",
    )
    args = parser.parse_args()

    if not args.split_only:
        print("Step 1: Merging training + validation → data/raw/ ...")
        merge_dataset_folders()

    # Clean previous processed data
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    print("\nStep 2: Splitting data/raw → data/processed (70/15/15) ...")
    split_dataset(source_dir=RAW_DATA_DIR, dest_dir=PROCESSED_DATA_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
