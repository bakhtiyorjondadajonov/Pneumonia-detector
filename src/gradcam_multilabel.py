"""Multi-label Grad-CAM wrapper for generating per-disease heatmaps."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.gradcam import GradCAM, overlay_gradcam
from src.model import get_target_layer


def generate_multilabel_gradcams(
    learn,
    img_tensor: torch.Tensor,
    arch_name: str,
    class_names: list[str],
    class_indices: Optional[list[int]] = None,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Generate Grad-CAM heatmaps for multiple diseases.

    Args:
        learn: Trained multi-label Learner.
        img_tensor: Preprocessed image tensor (1, C, H, W).
        arch_name: Architecture name.
        class_names: List of all class names.
        class_indices: Specific class indices to generate CAMs for.
            If None, generates for all classes above threshold.
        y_prob: Predicted probabilities (1D array). Used with threshold
            to auto-select classes. Required if class_indices is None.
        threshold: Probability threshold for auto-selecting classes.

    Returns:
        Dict mapping class_name to heatmap numpy array (H, W) in [0, 1].
    """
    if class_indices is None:
        if y_prob is None:
            raise ValueError("Either class_indices or y_prob must be provided")
        class_indices = [i for i, p in enumerate(y_prob) if p >= threshold]
        # Always include at least the top prediction
        if not class_indices:
            class_indices = [int(np.argmax(y_prob))]

    target_layer = get_target_layer(learn, arch_name)
    gradcam = GradCAM(learn.model, target_layer)

    cams_by_idx = gradcam.generate_multiple(img_tensor, class_indices)

    return {class_names[idx]: cam for idx, cam in cams_by_idx.items()}


def create_gradcam_grid(
    image: Image.Image,
    cams: dict[str, np.ndarray],
    alpha: float = 0.4,
    cols: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a grid showing original image + Grad-CAM overlays per disease.

    Args:
        image: Original PIL Image.
        cams: Dict mapping disease name to heatmap array.
        alpha: Overlay transparency.
        cols: Number of columns in the grid.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure with the grid.
    """
    n_cams = len(cams)
    n_total = n_cams + 1  # +1 for original image
    rows = (n_total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    # Original image
    axes[0, 0].imshow(image.convert("RGB"))
    axes[0, 0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Grad-CAM overlays
    for i, (disease_name, cam) in enumerate(cams.items(), start=1):
        r, c = divmod(i, cols)
        overlay = overlay_gradcam(image, cam, alpha=alpha)
        axes[r, c].imshow(overlay)
        axes[r, c].set_title(disease_name, fontsize=10, fontweight="bold")
        axes[r, c].axis("off")

    # Hide unused axes
    for i in range(n_total, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle("Grad-CAM: Per-Disease Attention Maps", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig
