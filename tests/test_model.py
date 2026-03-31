"""Basic tests for model loading and prediction pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ARCHITECTURES, count_parameters, get_target_layer
from src.gradcam import GradCAM, create_heatmap, overlay_gradcam


class TestArchitectures:
    """Tests for model architecture factory."""

    def test_all_architectures_defined(self):
        expected = {"resnet34", "resnet50", "densenet121"}
        assert set(ARCHITECTURES.keys()) == expected

    def test_invalid_architecture_raises(self):
        from src.model import create_learner
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_learner(MagicMock(), arch_name="invalid_model")


class TestGradCAM:
    """Tests for Grad-CAM implementation."""

    def test_create_heatmap_shape(self):
        cam = torch.rand(224, 224).numpy()
        heatmap = create_heatmap(cam)
        assert heatmap.shape == (224, 224, 3)
        assert heatmap.dtype.name == "uint8"
        assert heatmap.max() <= 255
        assert heatmap.min() >= 0

    def test_create_heatmap_normalized_input(self):
        # All zeros
        cam_zeros = torch.zeros(100, 100).numpy()
        heatmap = create_heatmap(cam_zeros)
        assert heatmap.shape == (100, 100, 3)

        # All ones
        cam_ones = torch.ones(100, 100).numpy()
        heatmap = create_heatmap(cam_ones)
        assert heatmap.shape == (100, 100, 3)

    def test_overlay_gradcam_output(self):
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype="uint8"))
        cam = np.random.rand(224, 224).astype("float32")

        overlay = overlay_gradcam(img, cam, alpha=0.4)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (224, 224)
        assert overlay.mode == "RGB"

    def test_overlay_gradcam_different_sizes(self):
        from PIL import Image
        import numpy as np

        # Image and CAM have different sizes - should still work
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype="uint8"))
        cam = np.random.rand(7, 7).astype("float32")

        overlay = overlay_gradcam(img, cam, alpha=0.4)
        assert overlay.size == (512, 512)


class TestGradCAMHooks:
    """Tests for Grad-CAM hook mechanism."""

    def test_gradcam_generate(self):
        # Create a minimal CNN
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 2),
        )
        target_layer = model[0]  # conv layer

        gradcam = GradCAM(model, target_layer)
        input_tensor = torch.randn(1, 3, 32, 32)

        cam = gradcam.generate(input_tensor, class_idx=0)
        assert cam.shape == (32, 32)
        assert cam.min() >= 0
        assert cam.max() <= 1.0
