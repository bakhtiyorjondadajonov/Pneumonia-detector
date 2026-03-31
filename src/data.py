"""Data loading, augmentation, and DataLoader creation for chest X-ray classification."""

from pathlib import Path
from typing import Optional

import yaml
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    DataLoaders,
    ImageBlock,
    Normalize,
    Resize,
    aug_transforms,
    get_image_files,
    imagenet_stats,
    parent_label,
    GrandparentSplitter,
    RandomSplitter,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataloaders(
    data_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    valid_pct: float = 0.2,
    seed: int = 42,
    use_augmentation: bool = True,
) -> DataLoaders:
    """Create training and validation DataLoaders with proper augmentation.

    Uses the train/ folder from the Kaggle chest X-ray dataset, splitting it
    into train/validation sets. The test/ folder is reserved for final evaluation.

    Args:
        data_path: Path to dataset root (containing train/, test/ folders).
        image_size: Target image size after resizing.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        valid_pct: Fraction of training data to use for validation.
        seed: Random seed for reproducible splits.
        use_augmentation: Whether to apply data augmentation.

    Returns:
        FastAI DataLoaders with train and validation sets.
    """
    train_path = Path(data_path) / "train"

    item_tfms = [Resize(256)]

    if use_augmentation:
        batch_tfms = [
            *aug_transforms(
                size=image_size,
                min_scale=0.75,
                flip_vert=False,  # vertical flips not anatomically meaningful for CXR
                max_rotate=15.0,
                max_lighting=0.2,
                max_warp=0.1,
            ),
            Normalize.from_stats(*imagenet_stats),
        ]
    else:
        batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    dls = data_block.dataloaders(train_path, bs=batch_size, num_workers=num_workers)
    return dls


def create_test_dataloader(
    learn,
    test_path: str,
    image_size: int = 224,
) -> "DataLoader":
    """Create a test DataLoader from the held-out test set.

    Args:
        learn: Trained fastai Learner (provides transforms and vocab).
        test_path: Path to test/ folder.
        image_size: Target image size.

    Returns:
        FastAI DataLoader for the test set.
    """
    test_files = get_image_files(Path(test_path))
    test_dl = learn.dls.test_dl(test_files)
    return test_dl, test_files
