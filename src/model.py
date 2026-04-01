"""Architecture factory for creating fastai vision learners."""

from typing import Optional

from fastai.vision.all import (
    Learner,
    accuracy,
    vision_learner,
    F1Score,
    Precision,
    Recall,
    DataLoaders,
)
from torchvision.models import resnet34, resnet50, densenet121


ARCHITECTURES = {
    "resnet34": resnet34,
    "resnet50": resnet50,
    "densenet121": densenet121,
}


def create_learner(
    dls: DataLoaders,
    arch_name: str = "resnet34",
    pretrained: bool = True,
) -> Learner:
    """Create a fastai vision learner for the specified architecture.

    Args:
        dls: FastAI DataLoaders with train and validation sets.
        arch_name: Architecture name. One of: resnet34, resnet50, densenet121.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        FastAI Learner ready for training.

    Raises:
        ValueError: If arch_name is not a supported architecture.
    """
    if arch_name not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Supported: {list(ARCHITECTURES.keys())}"
        )

    arch = ARCHITECTURES[arch_name]

    learn = vision_learner(
        dls,
        arch,
        metrics=[accuracy, F1Score(), Precision(), Recall()],
        pretrained=pretrained,
    )

    return learn


def get_target_layer(learn: Learner, arch_name: str):
    """Get the target convolutional layer for Grad-CAM.

    Args:
        learn: Trained fastai Learner.
        arch_name: Architecture name to determine layer location.

    Returns:
        The target nn.Module (last conv block).
    """
    model = learn.model
    if arch_name.startswith("resnet"):
        return model[0][-1][-1]  # last block of the body
    elif arch_name == "densenet121":
        body = model[0]
        if hasattr(body, 'features'):
            return body.features.denseblock4
        return body[0].denseblock4
    else:
        raise ValueError(f"No Grad-CAM target layer defined for '{arch_name}'")


def count_parameters(learn: Learner) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in learn.model.parameters() if p.requires_grad)
