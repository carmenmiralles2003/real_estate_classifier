"""
Download and prepare the real-estate image dataset.

Option 1 — Kaggle (recommended):
    pip install kaggle
    export KAGGLE_USERNAME=<your_user>
    export KAGGLE_KEY=<your_key>
    python scripts/download_data.py --kaggle

Option 2 — Manual:
    Place images in data/raw/<class_name>/ and run:
    python scripts/download_data.py --split-only

Classes expected: bathroom, bedroom, dining_room, exterior, kitchen, living_room
"""
import argparse
import os
import zipfile
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.dataset import split_dataset


def download_kaggle_dataset():
    """
    Downloads the 'House Rooms Image Dataset' from Kaggle.
    Requires KAGGLE_USERNAME and KAGGLE_KEY env vars or ~/.kaggle/kaggle.json.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Install kaggle: pip install kaggle")
        return False

    api = KaggleApi()
    api.authenticate()

    dataset_slug = "robinreni/house-rooms-image-dataset"
    download_dir = RAW_DATA_DIR.parent

    print(f"Downloading {dataset_slug} → {download_dir}")
    api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=False)

    # Unzip
    zip_path = download_dir / "house-rooms-image-dataset.zip"
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(download_dir)
        zip_path.unlink()

    # Reorganize into expected structure if needed
    _reorganize_kaggle_data(download_dir)
    return True


def _reorganize_kaggle_data(download_dir: Path):
    """
    Reorganize the Kaggle download into data/raw/<class>/
    The Kaggle dataset often has a nested structure — adapt as needed.
    """
    import shutil

    # Common Kaggle structure: House_Rooms/Bathroom/, House_Rooms/Bedroom/, etc.
    possible_roots = list(download_dir.glob("**/House*Rooms*")) + \
                     list(download_dir.glob("**/house*room*"))

    if not possible_roots:
        # Data may already be in the right place
        if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
            print("Data already in expected location.")
            return
        print("WARNING: Could not find expected folder structure. "
              "Please manually organize images into data/raw/<class_name>/")
        return

    source_root = possible_roots[0]
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Map common folder names to our class names
    name_map = {
        "bathroom": "bathroom",
        "bedroom": "bedroom",
        "dining": "dining_room",
        "dining_room": "dining_room",
        "dining room": "dining_room",
        "exterior": "exterior",
        "front": "exterior",
        "kitchen": "kitchen",
        "living": "living_room",
        "living_room": "living_room",
        "living room": "living_room",
        "livingroom": "living_room",
    }

    for subdir in source_root.iterdir():
        if subdir.is_dir():
            key = subdir.name.lower().strip()
            target_name = name_map.get(key, key.replace(" ", "_"))
            target_dir = RAW_DATA_DIR / target_name
            if target_dir.exists():
                # Merge
                for f in subdir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, target_dir / f.name)
            else:
                shutil.copytree(subdir, target_dir)

    print(f"Data reorganized into {RAW_DATA_DIR}")
    for d in sorted(RAW_DATA_DIR.iterdir()):
        if d.is_dir():
            count = len(list(d.glob("*")))
            print(f"  {d.name}: {count} images")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare dataset")
    parser.add_argument("--kaggle", action="store_true", help="Download from Kaggle")
    parser.add_argument("--split-only", action="store_true", help="Skip download, only split data/raw → data/processed")
    args = parser.parse_args()

    if args.kaggle:
        success = download_kaggle_dataset()
        if not success:
            return

    if args.split_only or args.kaggle:
        print("\nSplitting dataset into train/val/test...")
        split_dataset(source_dir=RAW_DATA_DIR, dest_dir=PROCESSED_DATA_DIR)
        print("Done!")
    else:
        print("Usage:")
        print("  python scripts/download_data.py --kaggle       # Download from Kaggle + split")
        print("  python scripts/download_data.py --split-only   # Just split existing data/raw/")


if __name__ == "__main__":
    main()
