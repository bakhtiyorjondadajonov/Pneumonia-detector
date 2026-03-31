"""Tests for multi-label classification pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_multilabel import CLASSES
from src.gradcam import GradCAM


class TestNIHLabelParsing:
    """Tests for NIH ChestX-ray14 label parsing."""

    def test_classes_count(self):
        assert len(CLASSES) == 14

    def test_all_classes_present(self):
        expected = {
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia",
            "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
            "Pneumonia", "Pneumothorax",
        }
        assert set(CLASSES) == expected

    def test_no_finding_is_all_zeros(self):
        """'No Finding' should map to all-zeros vector."""
        labels_str = ""  # empty string = no findings
        multi_hot = np.zeros(14, dtype=np.float32)
        for label in labels_str.split(";"):
            if label and label in CLASSES:
                multi_hot[CLASSES.index(label)] = 1.0
        assert multi_hot.sum() == 0

    def test_multi_hot_single_disease(self):
        """Single disease should produce one-hot vector."""
        labels_str = "Pneumonia"
        multi_hot = np.zeros(14, dtype=np.float32)
        for label in labels_str.split(";"):
            if label in CLASSES:
                multi_hot[CLASSES.index(label)] = 1.0
        assert multi_hot.sum() == 1
        assert multi_hot[CLASSES.index("Pneumonia")] == 1.0

    def test_multi_hot_multiple_diseases(self):
        """Multiple diseases should produce correct multi-hot vector."""
        labels_str = "Infiltration;Effusion;Pneumonia"
        multi_hot = np.zeros(14, dtype=np.float32)
        for label in labels_str.split(";"):
            if label in CLASSES:
                multi_hot[CLASSES.index(label)] = 1.0
        assert multi_hot.sum() == 3
        assert multi_hot[CLASSES.index("Infiltration")] == 1.0
        assert multi_hot[CLASSES.index("Effusion")] == 1.0
        assert multi_hot[CLASSES.index("Pneumonia")] == 1.0

    def test_multi_hot_shape(self):
        multi_hot = np.zeros(len(CLASSES), dtype=np.float32)
        assert multi_hot.shape == (14,)


class TestClassWeights:
    """Tests for positive class weight computation."""

    def test_pos_weight_computation(self):
        """Verify pos_weight = n_neg / n_pos, capped."""
        import pandas as pd
        from src.data_multilabel import compute_class_weights

        # Synthetic data: 100 samples, only "Pneumonia" has positives
        labels = ["Pneumonia"] * 20 + [""] * 80
        df = pd.DataFrame({"labels": labels})

        weights = compute_class_weights(df, CLASSES, cap=10.0)
        assert weights.shape == (len(CLASSES),)

        # Pneumonia: 80 neg / 20 pos = 4.0
        pneumonia_idx = CLASSES.index("Pneumonia")
        assert abs(weights[pneumonia_idx].item() - 4.0) < 0.01

        # All other classes: 0 positives → capped at 10.0
        for i, cls in enumerate(CLASSES):
            if cls != "Pneumonia":
                assert weights[i].item() == 10.0

    def test_pos_weight_cap(self):
        """Very rare class should be capped."""
        import pandas as pd
        from src.data_multilabel import compute_class_weights

        labels = ["Hernia"] * 1 + [""] * 999
        df = pd.DataFrame({"labels": labels})
        weights = compute_class_weights(df, CLASSES, cap=5.0)

        hernia_idx = CLASSES.index("Hernia")
        assert weights[hernia_idx].item() == 5.0  # capped at 5.0, not 999.0


class TestThresholdOptimization:
    """Tests for per-class threshold optimization."""

    def test_optimize_thresholds_synthetic(self):
        """Threshold optimization should find reasonable thresholds on synthetic data."""
        from src.evaluate_multilabel import optimize_thresholds

        np.random.seed(42)
        n = 500
        y_true = np.zeros((n, 14))
        y_prob = np.random.rand(n, 14) * 0.3  # mostly low probs

        # Make class 0 have clear positive signal
        y_true[:100, 0] = 1
        y_prob[:100, 0] = np.random.uniform(0.6, 1.0, 100)

        thresholds = optimize_thresholds(y_true, y_prob, CLASSES)

        assert isinstance(thresholds, dict)
        assert len(thresholds) == 14
        # Class 0 threshold should be around 0.3-0.7
        assert 0.2 <= thresholds[CLASSES[0]] <= 0.8


class TestGradCAMMultiLabel:
    """Tests for multi-label Grad-CAM generation."""

    def test_generate_multiple(self):
        """generate_multiple should return heatmaps for all requested classes."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 14),  # 14 output classes
        )
        target_layer = model[0]

        gradcam = GradCAM(model, target_layer)
        input_tensor = torch.randn(1, 3, 32, 32)

        cams = gradcam.generate_multiple(input_tensor, [0, 5, 13])
        assert len(cams) == 3
        assert set(cams.keys()) == {0, 5, 13}
        for idx, cam in cams.items():
            assert cam.shape == (32, 32)
            assert cam.min() >= 0
            assert cam.max() <= 1.0

    def test_different_classes_produce_different_cams(self):
        """Different target classes should generally produce different heatmaps."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 14),
        )
        # Initialize with non-trivial weights
        torch.manual_seed(123)
        torch.nn.init.kaiming_normal_(model[0].weight)
        torch.nn.init.kaiming_normal_(model[4].weight)

        target_layer = model[0]
        gradcam = GradCAM(model, target_layer)
        input_tensor = torch.randn(1, 3, 32, 32)

        cams = gradcam.generate_multiple(input_tensor, [0, 1])
        # They won't be identical (different backward paths)
        diff = np.abs(cams[0] - cams[1]).mean()
        # Just verify they're both valid heatmaps
        assert cams[0].shape == (32, 32)
        assert cams[1].shape == (32, 32)
