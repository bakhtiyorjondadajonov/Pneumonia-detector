"""Grad-CAM implementation from scratch using PyTorch hooks.

Generates visual explanations showing which regions of an X-ray image
most influenced the model's prediction. Implemented without external
libraries to demonstrate understanding of the technique.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Hook-based Grad-CAM for generating class activation maps.

    Registers forward and backward hooks on a target convolutional layer
    to capture activations and gradients, then computes a weighted
    combination to produce the heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: store activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: store gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed input image tensor (1, C, H, W).
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Compute weights: global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Apply ReLU (only keep positive contributions)
        cam = F.relu(cam)

        # Resize to input spatial dimensions
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam.squeeze()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    def generate_multiple(
        self, input_tensor: torch.Tensor, class_indices: list[int]
    ) -> dict[int, np.ndarray]:
        """Generate Grad-CAM heatmaps for multiple classes from a single forward pass.

        Efficient for multi-label models: one forward pass, multiple backward passes.
        Uses retain_graph=True to preserve the computation graph between calls.

        Args:
            input_tensor: Preprocessed input tensor (1, C, H, W).
            class_indices: List of target class indices.

        Returns:
            Dict mapping class_idx to heatmap numpy array (H, W) in [0, 1].
        """
        self.model.eval()
        cams = {}

        # Single forward pass
        output = self.model(input_tensor)

        for i, class_idx in enumerate(class_indices):
            is_last = (i == len(class_indices) - 1)

            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=not is_last)

            # Compute CAM
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
            )

            cam = cam.squeeze()
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            cams[class_idx] = cam.cpu().numpy()

        return cams


def create_heatmap(cam: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Convert a Grad-CAM heatmap to a colored RGB image.

    Args:
        cam: Heatmap array (H, W) with values in [0, 1].
        colormap: Matplotlib colormap name.

    Returns:
        RGB heatmap as uint8 numpy array (H, W, 3).
    """
    cmap = plt.colormaps.get_cmap(colormap)
    colored = cmap(cam)[:, :, :3]  # drop alpha channel
    return (colored * 255).astype(np.uint8)


def overlay_gradcam(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: Original PIL Image.
        cam: Heatmap array (H, W) with values in [0, 1].
        alpha: Transparency of the heatmap overlay (0=invisible, 1=opaque).
        colormap: Matplotlib colormap name.

    Returns:
        PIL Image with heatmap overlay.
    """
    # Resize heatmap to match image
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    # Apply colormap
    heatmap_rgb = create_heatmap(cam_resized)
    heatmap_img = Image.fromarray(heatmap_rgb).resize((w, h))

    # Blend
    overlay = Image.blend(img_rgb, heatmap_img, alpha)
    return overlay


def generate_gradcam_for_learner(
    learn, img_tensor: torch.Tensor, arch_name: str, class_idx: Optional[int] = None
) -> np.ndarray:
    """Convenience function to generate Grad-CAM from a fastai Learner.

    Args:
        learn: Trained fastai Learner.
        img_tensor: Preprocessed image tensor (1, C, H, W).
        arch_name: Architecture name to determine target layer.
        class_idx: Target class. If None, uses predicted class.

    Returns:
        Grad-CAM heatmap as numpy array (H, W) in [0, 1].
    """
    from src.model import get_target_layer

    target_layer = get_target_layer(learn, arch_name)
    gradcam = GradCAM(learn.model, target_layer)
    cam = gradcam.generate(img_tensor, class_idx=class_idx)
    return cam
