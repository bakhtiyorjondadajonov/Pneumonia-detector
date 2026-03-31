"""Multi-label architecture factory for 14-disease chest X-ray classification."""

import torch
import torch.nn as nn
import numpy as np
from fastai.vision.all import (
    Learner,
    DataLoaders,
    accuracy_multi,
    vision_learner,
    BCEWithLogitsLossFlat,
)

from src.model import ARCHITECTURES, get_target_layer, count_parameters


def create_multilabel_learner(
    dls: DataLoaders,
    arch_name: str = "resnet34",
    num_classes: int = 14,
    pos_weight: torch.Tensor = None,
    pretrained: bool = True,
    use_fp16: bool = True,
) -> Learner:
    """Create a fastai learner for multi-label classification.

    Args:
        dls: FastAI DataLoaders with multi-hot labels.
        arch_name: Architecture name. One of: resnet34, resnet50, densenet121.
        num_classes: Number of disease classes (default 14).
        pos_weight: Class weights for BCEWithLogitsLoss.
        pretrained: Whether to use ImageNet pretrained weights.
        use_fp16: Whether to enable mixed precision training.

    Returns:
        FastAI Learner configured for multi-label classification.
    """
    if arch_name not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Supported: {list(ARCHITECTURES.keys())}"
        )

    arch = ARCHITECTURES[arch_name]

    # Build loss function with class weights
    if pos_weight is not None:
        loss_func = BCEWithLogitsLossFlat(pos_weight=pos_weight)
    else:
        loss_func = BCEWithLogitsLossFlat()

    learn = vision_learner(
        dls,
        arch,
        n_out=num_classes,
        loss_func=loss_func,
        metrics=[accuracy_multi],
        pretrained=pretrained,
    )

    # Move pos_weight to same device as model
    if pos_weight is not None:
        device = next(learn.model.parameters()).device
        learn.loss_func.pos_weight = pos_weight.to(device)

    # Enable mixed precision for RTX 5070 (12GB VRAM)
    if use_fp16:
        learn = learn.to_fp16()

    return learn
