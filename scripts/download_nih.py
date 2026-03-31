"""Download and set up the NIH ChestX-ray14 dataset from Kaggle.

Usage:
    python scripts/download_nih.py           # 224x224 resized version (~2.5GB, recommended)
    python scripts/download_nih.py --full     # Original full-resolution (~42GB)

The 224x224 resized version contains ALL 112,120 images with all 14 disease
labels, pre-resized to the input size our models use. Zero quality loss
since our pipeline resizes to 224x224 anyway.

Requires:
    - Kaggle CLI installed: pip install kaggle
    - Kaggle API credentials at ~/.kaggle/kaggle.json
"""

import argparse
import subprocess
import sys
from pathlib import Path


DATA_DIR = Path("data/nih-chestxray14")

# 224x224 resized version: ~2.5GB, all 112K images
DATASET_RESIZED = "khanfashee/nih-chest-x-ray-14-224x224-resized"
# Original full-resolution: ~42GB
DATASET_FULL = "nih-chest-xrays/data"


def check_kaggle_cli():
    """Verify Kaggle CLI is installed and configured."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: Kaggle CLI not found. Install with: pip install kaggle")
        print("Then configure: place kaggle.json in ~/.kaggle/")
        sys.exit(1)


def download_resized():
    """Download 224x224 resized NIH ChestX-ray14 (~2.5GB).

    Contains all 112,120 images pre-resized to 224x224.
    Since our models use 224x224 input, this is lossless for our purpose.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if (DATA_DIR / "images").exists() and len(list((DATA_DIR / "images").glob("*.png"))) > 1000:
        print(f"Dataset already exists at {DATA_DIR}")
        return

    print(f"Downloading NIH ChestX-ray14 (224x224 resized, ~2.5GB)...")
    print(f"Target: {DATA_DIR}\n")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_RESIZED,
         "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )

    # Organize files into expected structure
    organize_resized_dataset()
    print("\nDownload complete!")


def download_full():
    """Download original full-resolution NIH ChestX-ray14 (~42GB)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "Data_Entry_2017.csv"
    if csv_path.exists():
        print(f"Dataset already exists at {DATA_DIR}")
        return

    print(f"Downloading FULL NIH ChestX-ray14 (~42GB)...")
    print(f"Target: {DATA_DIR}\n")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_FULL,
         "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )

    consolidate_images()
    print("\nFull download complete!")


def consolidate_images():
    """Move images from subdirectories (images_001/ etc.) into a single images/ dir."""
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    for subdir in sorted(DATA_DIR.glob("images_*")):
        if subdir.is_dir():
            for img in subdir.glob("*.png"):
                dest = images_dir / img.name
                if not dest.exists():
                    img.rename(dest)
            try:
                subdir.rmdir()
            except OSError:
                pass


def organize_resized_dataset():
    """Organize the resized dataset into the expected structure.

    The resized dataset may have a different directory layout.
    We need: DATA_DIR/images/*.png, DATA_DIR/Data_Entry_2017.csv, etc.
    """
    # Check for common layout variations
    images_dir = DATA_DIR / "images"

    # Look for images in subdirectories
    for subdir in DATA_DIR.iterdir():
        if subdir.is_dir() and subdir.name != "images":
            pngs = list(subdir.glob("*.png"))
            if pngs:
                images_dir.mkdir(exist_ok=True)
                print(f"  Moving {len(pngs)} images from {subdir.name}/ to images/...")
                for img in pngs:
                    dest = images_dir / img.name
                    if not dest.exists():
                        img.rename(dest)
                try:
                    subdir.rmdir()
                except OSError:
                    pass

    # If images are directly in DATA_DIR (not in a subdirectory)
    root_pngs = list(DATA_DIR.glob("*.png"))
    if root_pngs and not images_dir.exists():
        images_dir.mkdir(exist_ok=True)
        print(f"  Moving {len(root_pngs)} images to images/...")
        for img in root_pngs:
            img.rename(images_dir / img.name)

    # Check if we need the metadata files from the original dataset
    csv_path = DATA_DIR / "Data_Entry_2017.csv"
    train_list = DATA_DIR / "train_val_list.txt"
    test_list = DATA_DIR / "test_list.txt"

    missing = []
    if not csv_path.exists():
        missing.append("Data_Entry_2017.csv")
    if not train_list.exists():
        missing.append("train_val_list.txt")
    if not test_list.exists():
        missing.append("test_list.txt")

    if missing:
        print(f"\n  Downloading missing metadata files: {', '.join(missing)}")
        for fname in missing:
            try:
                subprocess.run(
                    ["kaggle", "datasets", "download", "-d", "nih-chest-xrays/data",
                     "-f", fname, "-p", str(DATA_DIR), "--unzip"],
                    check=True, capture_output=True,
                )
                print(f"    [OK] {fname}")
            except subprocess.CalledProcessError:
                # If individual file download fails, try alternative approach
                print(f"    [WARN] Could not download {fname} individually.")
                if fname == "Data_Entry_2017.csv":
                    print("    Trying alternative source...")
                    try:
                        subprocess.run(
                            ["kaggle", "datasets", "download", "-d", DATASET_RESIZED,
                             "-f", fname, "-p", str(DATA_DIR), "--unzip"],
                            check=True, capture_output=True,
                        )
                        print(f"    [OK] {fname} (from resized dataset)")
                    except subprocess.CalledProcessError:
                        print(f"    [FAIL] Please download {fname} manually")


def verify_dataset():
    """Verify the dataset structure is correct."""
    print("\nVerifying dataset...")

    required_files = [
        DATA_DIR / "Data_Entry_2017.csv",
        DATA_DIR / "train_val_list.txt",
        DATA_DIR / "test_list.txt",
    ]

    all_ok = True
    for f in required_files:
        status = "OK" if f.exists() else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"  [{status}] {f.name}")

    images_dir = DATA_DIR / "images"
    if images_dir.exists():
        count = len(list(images_dir.glob("*.png")))
        print(f"  [{'OK' if count > 1000 else 'LOW'}] images/: {count} images")
    else:
        # Check if images are elsewhere
        all_pngs = len(list(DATA_DIR.rglob("*.png")))
        if all_pngs > 0:
            print(f"  [WARN] Found {all_pngs} images but not in images/ directory")
        else:
            print("  [MISSING] No images found")
        all_ok = False

    if all_ok:
        print("\nDataset is ready for training!")
    else:
        print("\nSome files are missing. Check the output above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14 dataset")
    parser.add_argument("--full", action="store_true",
                        help="Download original full-resolution dataset (~42GB)")
    args = parser.parse_args()

    check_kaggle_cli()

    if args.full:
        download_full()
    else:
        download_resized()

    verify_dataset()
