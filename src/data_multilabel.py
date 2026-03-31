"""Data loading for NIH ChestX-ray14 multi-label classification.

Handles CSV parsing, multi-hot encoding, patient-level train/val splitting,
class weight computation, and fastai DataLoader creation.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastai.vision.all import (
    DataLoaders,
    DataBlock,
    ImageBlock,
    MultiCategoryBlock,
    ColReader,
    ColSplitter,
    Normalize,
    Resize,
    aug_transforms,
    imagenet_stats,
)

from src.data import load_config

# The 14 pathologies in NIH ChestX-ray14
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def load_nih_metadata(
    csv_path: str,
    image_dir: str,
    train_list: str,
    test_list: str,
    valid_pct: float = 0.15,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess NIH ChestX-ray14 metadata.

    Parses the CSV, builds multi-hot label strings, and splits data
    by Patient ID to prevent data leakage. Optionally samples a
    fraction of patients for faster training.

    Args:
        csv_path: Path to Data_Entry_2017.csv.
        image_dir: Path to directory containing .png images.
        train_list: Path to train_val_list.txt (official split).
        test_list: Path to test_list.txt (official split).
        valid_pct: Fraction of training patients for validation.
        sample_fraction: Fraction of patients to use (0.0-1.0).
            Set to 1.0 for full dataset, 0.25 for quick training.
            Sampling is done at the patient level to prevent leakage.
        seed: Random seed for reproducible patient-level split.

    Returns:
        Tuple of (train_val_df, test_df) DataFrames with columns:
        path, labels, patient_id, is_valid
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Image Index": "filename", "Finding Labels": "raw_labels",
                             "Patient ID": "patient_id"})

    # Build full image paths
    image_dir = Path(image_dir)
    df["path"] = df["filename"].apply(lambda x: str(image_dir / x))

    # Parse labels: "Infiltration|Effusion" -> "Infiltration;Effusion"
    # "No Finding" -> "" (empty string = no diseases)
    def parse_labels(raw: str) -> str:
        if raw == "No Finding":
            return ""
        # Replace pipe with semicolon (fastai MultiCategoryBlock uses ; by default)
        labels = raw.split("|")
        # Replace space in "Pleural Thickening" with underscore to match config
        labels = [l.replace(" ", "_") for l in labels]
        return ";".join(labels)

    df["labels"] = df["raw_labels"].apply(parse_labels)

    # Load official train/test split files
    train_filenames = set(Path(train_list).read_text().strip().split("\n"))
    test_filenames = set(Path(test_list).read_text().strip().split("\n"))

    # Split into train+val and test
    train_val_df = df[df["filename"].isin(train_filenames)].copy()
    test_df = df[df["filename"].isin(test_filenames)].copy()

    # Patient-level train/val split (prevent data leakage)
    np.random.seed(seed)
    patient_ids = train_val_df["patient_id"].unique()
    np.random.shuffle(patient_ids)

    # Subsample patients if sample_fraction < 1.0
    if sample_fraction < 1.0:
        n_sample = max(int(len(patient_ids) * sample_fraction), 100)
        patient_ids = patient_ids[:n_sample]
        train_val_df = train_val_df[train_val_df["patient_id"].isin(patient_ids)].copy()

        # Also subsample test set proportionally
        test_patients = test_df["patient_id"].unique()
        np.random.shuffle(test_patients)
        n_test_sample = max(int(len(test_patients) * sample_fraction), 50)
        test_df = test_df[test_df["patient_id"].isin(test_patients[:n_test_sample])].copy()

        print(f"  Subset mode: using {sample_fraction:.0%} of patients ({n_sample} train, {n_test_sample} test)")

    n_val = int(len(patient_ids) * valid_pct)
    val_patients = set(patient_ids[:n_val])

    train_val_df["is_valid"] = train_val_df["patient_id"].isin(val_patients)

    # Test set: all marked as valid (not used for training)
    test_df["is_valid"] = True

    print(f"Dataset loaded:")
    print(f"  Train: {(~train_val_df['is_valid']).sum()} images")
    print(f"  Val:   {train_val_df['is_valid'].sum()} images")
    print(f"  Test:  {len(test_df)} images")
    print(f"  Patients (train): {len(patient_ids) - n_val}")
    print(f"  Patients (val):   {n_val}")

    return train_val_df, test_df


def compute_class_weights(
    df: pd.DataFrame,
    classes: list[str] = CLASSES,
    cap: float = 10.0,
) -> torch.Tensor:
    """Compute positive class weights for BCEWithLogitsLoss.

    Args:
        df: DataFrame with 'labels' column (semicolon-separated).
        classes: List of class names.
        cap: Maximum weight to prevent instability for very rare classes.

    Returns:
        Tensor of shape (num_classes,) with pos_weight per class.
    """
    n_total = len(df)
    weights = []

    for cls in classes:
        n_pos = df["labels"].str.contains(cls, regex=False).sum()
        n_neg = n_total - n_pos
        if n_pos > 0:
            w = min(n_neg / n_pos, cap)
        else:
            w = cap
        weights.append(w)

    pos_weight = torch.tensor(weights, dtype=torch.float32)

    print(f"\nClass weights (pos_weight, capped at {cap}):")
    for cls, w in zip(classes, weights):
        n_pos = df["labels"].str.contains(cls, regex=False).sum()
        pct = n_pos / n_total * 100
        print(f"  {cls:<22s}: {w:>6.2f}  ({n_pos:>5d} pos, {pct:.1f}%)")

    return pos_weight


def create_multilabel_dataloaders(
    train_val_df: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoaders:
    """Create train and validation DataLoaders for multi-label classification.

    Args:
        train_val_df: DataFrame with path, labels, is_valid columns.
        image_size: Target image size.
        batch_size: Batch size.
        num_workers: Data loading workers.

    Returns:
        FastAI DataLoaders with multi-hot encoded labels.
    """
    item_tfms = [Resize(256)]
    batch_tfms = [
        *aug_transforms(
            size=image_size,
            min_scale=0.75,
            flip_vert=False,
            max_rotate=15.0,
            max_lighting=0.2,
            max_warp=0.1,
        ),
        Normalize.from_stats(*imagenet_stats),
    ]

    data_block = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock),
        get_x=ColReader("path"),
        get_y=ColReader("labels", label_delim=";"),
        splitter=ColSplitter("is_valid"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    dls = data_block.dataloaders(
        train_val_df,
        bs=batch_size,
        num_workers=num_workers,
    )

    return dls


def create_multilabel_test_dl(learn, test_df: pd.DataFrame):
    """Create a test DataLoader from the held-out test DataFrame.

    Args:
        learn: Trained fastai Learner.
        test_df: Test DataFrame with 'path' and 'labels' columns.

    Returns:
        Tuple of (test_dl, test_df) for evaluation.
    """
    test_dl = learn.dls.test_dl(test_df["path"].tolist())
    return test_dl
