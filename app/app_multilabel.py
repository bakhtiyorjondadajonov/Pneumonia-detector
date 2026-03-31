"""Streamlit app for multi-label chest X-ray classification (14 diseases)."""

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from fastai.vision.all import PILImage, load_learner

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gradcam import GradCAM, overlay_gradcam
from src.model import get_target_layer
from src.data_multilabel import CLASSES

# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Disease Chest X-Ray Analyzer",
    page_icon="🫁",
    layout="wide",
)

# ─── Styling ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-title { text-align: center; margin-bottom: 0; }
    .subtitle { text-align: center; color: #666; font-size: 1.1em; }
    .finding-likely { color: #d32f2f; font-weight: bold; }
    .finding-possible { color: #f57c00; font-weight: bold; }
    .finding-unlikely { color: #388e3c; font-weight: bold; }
    .disclaimer {
        background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
        padding: 12px; margin-top: 20px; font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Constants ─────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "outputs" / "models"
METRICS_DIR = Path(__file__).parent.parent / "outputs" / "metrics"
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "nih-chestxray14" / "images"

AVAILABLE_MODELS = {
    "ResNet34": ("resnet34", "resnet34_multilabel.pkl"),
    "ResNet50": ("resnet50", "resnet50_multilabel.pkl"),
    "DenseNet121": ("densenet121", "densenet121_multilabel.pkl"),
}


# ─── Model Loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_filename: str):
    """Load a trained model with caching."""
    model_path = MODELS_DIR / model_filename
    if model_path.exists():
        return load_learner(model_path)
    return None


@st.cache_data
def load_thresholds(arch_name: str) -> dict:
    """Load optimized per-class thresholds."""
    path = METRICS_DIR / f"{arch_name}_multilabel_thresholds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Default thresholds
    return {cls: 0.5 for cls in CLASSES}


@st.cache_data
def load_metrics(arch_name: str) -> dict:
    """Load evaluation metrics."""
    path = METRICS_DIR / f"{arch_name}_multilabel_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_available_models() -> dict:
    """Return available trained multi-label models."""
    available = {}
    for display_name, (arch_name, filename) in AVAILABLE_MODELS.items():
        if (MODELS_DIR / filename).exists():
            available[display_name] = (arch_name, filename)
    return available


def get_severity(prob: float, threshold: float) -> tuple[str, str, str]:
    """Determine severity level based on probability and threshold.

    Returns: (level, color, css_class)
    """
    if prob >= threshold:
        return "Likely", "#d32f2f", "finding-likely"
    elif prob >= threshold * 0.6:
        return "Possible", "#f57c00", "finding-possible"
    else:
        return "Unlikely", "#388e3c", "finding-unlikely"


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    available = get_available_models()
    if not available:
        st.error("No trained multi-label models found. Run training first:\n`python -m src.train_multilabel`")
        st.stop()

    model_choice = st.selectbox("Model Architecture", list(available.keys()))
    arch_name, model_filename = available[model_choice]

    st.divider()

    show_gradcam = st.toggle("Show Grad-CAM", value=True)

    if show_gradcam:
        gradcam_disease = st.selectbox(
            "Grad-CAM Disease",
            ["Auto (Top Finding)"] + CLASSES,
            help="Select which disease's attention map to display.",
        )

    st.divider()
    st.subheader("About")
    st.markdown(
        """
        **Multi-label classifier** detecting 14 chest pathologies
        from X-ray images using transfer learning.

        **Dataset:** [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)
        (112,120 images)
        """
    )

# ─── Main Content ──────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">Multi-Disease Chest X-Ray Analyzer 🫁</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-powered detection of 14 chest pathologies with visual explanations</p>',
    unsafe_allow_html=True,
)

# Load model and thresholds
model = load_model(model_filename)
if model is None:
    st.error(f"Failed to load model: {model_filename}")
    st.stop()

thresholds = load_thresholds(arch_name)

# ─── Image Upload ──────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if not uploaded_file:
    st.info("Upload a chest X-ray image to begin analysis.")
    st.stop()

# ─── Prediction ───────────────────────────────────────────────────────────────

img = PILImage.create(uploaded_file)
pil_img = Image.open(uploaded_file).convert("RGB")

with st.spinner("Analyzing X-ray for 14 pathologies..."):
    # Get predictions
    pred, pred_idx, raw_probs = model.predict(img)
    # For multi-label, raw_probs are already sigmoid outputs
    probs = raw_probs.numpy()

# ─── Results Layout ────────────────────────────────────────────────────────────

col_img, col_cam = st.columns(2)

with col_img:
    st.subheader("Original X-Ray")
    st.image(pil_img, use_container_width=True)

with col_cam:
    st.subheader("Grad-CAM Visualization")
    if show_gradcam:
        try:
            dl = model.dls.test_dl([img])
            batch = next(iter(dl))
            x = batch[0].to(next(model.model.parameters()).device)

            # Determine target class for Grad-CAM
            if gradcam_disease == "Auto (Top Finding)":
                target_idx = int(np.argmax(probs))
                target_name = CLASSES[target_idx]
            else:
                target_idx = CLASSES.index(gradcam_disease)
                target_name = gradcam_disease

            target_layer = get_target_layer(model, arch_name)
            gradcam_obj = GradCAM(model.model, target_layer)
            cam = gradcam_obj.generate(x, class_idx=target_idx)

            overlay = overlay_gradcam(pil_img, cam, alpha=0.4)
            st.image(overlay, use_container_width=True)
            prob_val = probs[target_idx]
            st.caption(f"Showing attention for: **{target_name}** (prob: {prob_val:.1%})")
        except Exception as e:
            st.warning(f"Grad-CAM unavailable: {e}")
            st.image(pil_img, use_container_width=True)
    else:
        st.info("Enable Grad-CAM in the sidebar.")
        st.image(pil_img, use_container_width=True)

st.divider()

# ─── Disease Predictions Panel ─────────────────────────────────────────────────

st.subheader("Disease Predictions")

# Sort by probability (highest first)
sorted_indices = np.argsort(probs)[::-1]

# Count findings
n_likely = sum(1 for i in range(len(CLASSES)) if probs[i] >= thresholds.get(CLASSES[i], 0.5))

if n_likely == 0:
    st.success("No significant findings detected.")
else:
    st.warning(f"**{n_likely} potential finding(s) detected** — see details below.")

# Create horizontal bar chart
fig = go.Figure()

bar_colors = []
labels_sorted = []
probs_sorted = []
threshold_lines = []

for idx in sorted_indices:
    cls = CLASSES[idx]
    prob = probs[idx]
    thresh = thresholds.get(cls, 0.5)
    severity, color, _ = get_severity(prob, thresh)

    labels_sorted.append(cls)
    probs_sorted.append(prob * 100)
    bar_colors.append(color)
    threshold_lines.append(thresh * 100)

fig.add_trace(go.Bar(
    y=labels_sorted[::-1],  # reverse for top-to-bottom
    x=probs_sorted[::-1],
    orientation="h",
    marker_color=bar_colors[::-1],
    text=[f"{p:.1f}%" for p in probs_sorted[::-1]],
    textposition="outside",
))

# Add threshold markers
for i, (label, thresh) in enumerate(zip(labels_sorted[::-1], threshold_lines[::-1])):
    fig.add_shape(
        type="line",
        x0=thresh, x1=thresh,
        y0=i - 0.4, y1=i + 0.4,
        line=dict(color="black", width=2, dash="dot"),
    )

fig.update_layout(
    title="Prediction Probabilities (dotted line = optimized threshold)",
    xaxis_title="Probability (%)",
    xaxis_range=[0, 105],
    height=500,
    margin=dict(l=150),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

# ─── Detail Tabs ───────────────────────────────────────────────────────────────

tab_findings, tab_gradcam_grid, tab_model = st.tabs(
    ["All Findings", "Grad-CAM Comparison", "Model Info"]
)

with tab_findings:
    # Detailed findings table
    for idx in sorted_indices:
        cls = CLASSES[idx]
        prob = probs[idx]
        thresh = thresholds.get(cls, 0.5)
        severity, color, css_class = get_severity(prob, thresh)

        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"**{cls}**")
        with col2:
            st.progress(min(prob, 1.0), text=f"{prob:.1%}")
        with col3:
            st.markdown(f'<span class="{css_class}">{severity}</span>', unsafe_allow_html=True)

with tab_gradcam_grid:
    if show_gradcam:
        # Show Grad-CAM for top 6 findings
        top_indices = sorted_indices[:6]
        try:
            dl = model.dls.test_dl([img])
            batch = next(iter(dl))
            x = batch[0].to(next(model.model.parameters()).device)

            target_layer = get_target_layer(model, arch_name)
            gradcam_obj = GradCAM(model.model, target_layer)
            cams = gradcam_obj.generate_multiple(x, list(top_indices))

            cols = st.columns(3)
            for i, idx in enumerate(top_indices):
                with cols[i % 3]:
                    overlay = overlay_gradcam(pil_img, cams[idx], alpha=0.4)
                    st.image(overlay, caption=f"{CLASSES[idx]} ({probs[idx]:.1%})",
                             use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM grid unavailable: {e}")
    else:
        st.info("Enable Grad-CAM in the sidebar to see per-disease attention maps.")

with tab_model:
    st.write(f"**Architecture:** {model_choice}")
    st.write(f"**Task:** Multi-label classification (14 diseases)")
    st.write(f"**Input Size:** 224 x 224 pixels")
    st.write(f"**Dataset:** NIH ChestX-ray14 (112,120 images)")

    metrics = load_metrics(arch_name)
    if metrics:
        st.subheader("Test Set Performance")
        macro = metrics.get("macro", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Macro AUC-ROC", f"{macro.get('auc_roc', 0):.4f}")
        m2.metric("Macro Precision", f"{macro.get('precision', 0):.1%}")
        m3.metric("Macro Recall", f"{macro.get('recall', 0):.1%}")
        m4.metric("Macro F1", f"{macro.get('f1_score', 0):.1%}")

        st.subheader("Per-Class AUC-ROC")
        per_class = metrics.get("per_class", {})
        if per_class:
            import pandas as pd
            df = pd.DataFrame([
                {"Disease": cls, "AUC-ROC": f"{m['auc_roc']:.4f}",
                 "Precision": f"{m['precision']:.1%}", "Recall": f"{m['recall']:.1%}",
                 "F1": f"{m['f1_score']:.1%}", "Prevalence": f"{m['prevalence']:.1%}"}
                for cls, m in per_class.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Run evaluation (`python -m src.evaluate_multilabel`) to see metrics.")

# ─── Disclaimer ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
        It is NOT a substitute for professional medical diagnosis. AI predictions are based on
        pattern recognition in training data and may not generalize to all clinical scenarios.
        Always consult a qualified healthcare provider for medical decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
