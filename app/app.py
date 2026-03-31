"""Streamlit web application for chest X-ray pneumonia detection with Grad-CAM."""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import torch
import numpy as np
from PIL import Image
from fastai.vision.all import PILImage, load_learner

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gradcam import GradCAM, overlay_gradcam
from src.model import get_target_layer

# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="X-Ray Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
)

# ─── Styling ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-title { text-align: center; margin-bottom: 0; }
    .subtitle { text-align: center; color: #666; font-size: 1.1em; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 15px;
        text-align: center; border: 1px solid #e9ecef;
    }
    .disclaimer {
        background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
        padding: 12px; margin-top: 20px; font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Model Loading ─────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "outputs" / "models"
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "chest_xray" / "test"

AVAILABLE_MODELS = {
    "ResNet34": ("resnet34", "resnet34_best.pkl"),
    "ResNet50": ("resnet50", "resnet50_best.pkl"),
    "DenseNet121": ("densenet121", "densenet121_best.pkl"),
}


@st.cache_resource
def load_model(model_filename: str):
    """Load a trained model with caching to avoid reloading on every interaction."""
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        # Fallback to root-level model for backward compatibility
        fallback = Path(__file__).parent.parent / "Pneumonia_analizer.pkl"
        if fallback.exists():
            return load_learner(fallback)
        return None
    return load_learner(model_path)


def get_available_models() -> dict:
    """Return dict of model names that have .pkl files available."""
    available = {}
    for display_name, (arch_name, filename) in AVAILABLE_MODELS.items():
        if (MODELS_DIR / filename).exists():
            available[display_name] = (arch_name, filename)

    # Fallback: check for original model
    fallback = Path(__file__).parent.parent / "Pneumonia_analizer.pkl"
    if not available and fallback.exists():
        available["ResNet34 (Original)"] = ("resnet34", "Pneumonia_analizer.pkl")
    return available


def get_sample_images(n: int = 3) -> dict:
    """Get sample X-ray images from test set for demo purposes."""
    samples = {"NORMAL": [], "PNEUMONIA": []}
    for label in samples:
        folder = SAMPLE_DIR / label
        if folder.exists():
            files = sorted(folder.glob("*.jpeg"))[:n]
            samples[label] = files
    return samples


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    # Model selection
    available = get_available_models()
    if not available:
        st.error("No trained models found. Run training first.")
        st.stop()

    model_choice = st.selectbox("Model Architecture", list(available.keys()))
    arch_name, model_filename = available[model_choice]

    st.divider()

    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.75,
        step=0.05,
        help="Predictions below this threshold will be flagged as uncertain.",
    )

    # Grad-CAM toggle
    show_gradcam = st.toggle("Show Grad-CAM Overlay", value=True)

    st.divider()

    # Model info
    st.subheader("About")
    st.markdown(
        """
        This model detects **pneumonia** from chest X-ray
        images using transfer learning with pretrained CNNs.

        **Dataset:** [Kaggle Chest X-Ray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

        **Metrics:** See the _Model Info_ tab after making a prediction.
        """
    )

# ─── Main Content ──────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">X-Ray Pneumonia Detector (XPD) 🫁</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-powered chest X-ray analysis using deep learning</p>',
    unsafe_allow_html=True,
)

st.write("")

# Load selected model
model = load_model(model_filename)
if model is None:
    st.error(f"Failed to load model: {model_filename}")
    st.stop()

# ─── Image Upload ──────────────────────────────────────────────────────────────

tab_upload, tab_samples = st.tabs(["Upload Image", "Sample Images"])

uploaded_file = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

with tab_samples:
    samples = get_sample_images()
    if any(samples.values()):
        st.write("Click a sample image to use it for prediction:")
        cols = st.columns(6)
        idx = 0
        for label, files in samples.items():
            for f in files:
                with cols[idx % 6]:
                    st.caption(label)
                    if st.button(f.stem[:15], key=f"sample_{idx}"):
                        st.session_state["sample_file"] = str(f)
                    st.image(str(f), width=100)
                idx += 1
    else:
        st.info("No sample images found. Download the dataset to `data/chest_xray/` to enable samples.")

# Determine which image to use
image_source = None
if uploaded_file is not None:
    image_source = uploaded_file
elif "sample_file" in st.session_state:
    image_source = st.session_state["sample_file"]

# ─── Prediction ───────────────────────────────────────────────────────────────

if image_source:
    # Load image
    img = PILImage.create(image_source)
    pil_img = Image.open(image_source).convert("RGB") if isinstance(image_source, str) else Image.open(image_source).convert("RGB")

    # Run prediction
    with st.spinner("Analyzing X-ray..."):
        prediction, pred_idx, probabilities = model.predict(img)
        confidence = probabilities[pred_idx].item()

    # ─── Results Layout ────────────────────────────────────────────────────

    col_img, col_gradcam = st.columns(2)

    with col_img:
        st.subheader("Original X-Ray")
        st.image(pil_img, use_container_width=True)

    with col_gradcam:
        st.subheader("Grad-CAM Visualization")
        if show_gradcam:
            try:
                # Prepare input tensor
                dl = model.dls.test_dl([img])
                batch = next(iter(dl))
                x = batch[0].to(next(model.model.parameters()).device)

                # Generate Grad-CAM
                target_layer = get_target_layer(model, arch_name)
                gradcam = GradCAM(model.model, target_layer)
                cam = gradcam.generate(x, class_idx=pred_idx)

                # Create overlay
                overlay = overlay_gradcam(pil_img, cam, alpha=0.4)
                st.image(overlay, use_container_width=True)
                st.caption("Highlighted regions show areas most influential for the prediction.")
            except Exception as e:
                st.warning(f"Grad-CAM unavailable: {e}")
                st.image(pil_img, use_container_width=True)
        else:
            st.info("Enable Grad-CAM in the sidebar to see visual explanations.")
            st.image(pil_img, use_container_width=True)

    st.divider()

    # ─── Prediction Result ─────────────────────────────────────────────────

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == "PNEUMONIA":
            st.error(f"**Prediction: {prediction}**")
        else:
            st.success(f"**Prediction: {prediction}**")

    with col2:
        if confidence >= confidence_threshold:
            st.info(f"**Confidence: {confidence * 100:.1f}%**")
        else:
            st.warning(f"**Confidence: {confidence * 100:.1f}%** (Below threshold)")

    with col3:
        if confidence < confidence_threshold:
            st.warning("**Requires Expert Review**")
        else:
            st.info(f"**Threshold: {confidence_threshold * 100:.0f}%**")

    # ─── Detail Tabs ───────────────────────────────────────────────────────

    tab_details, tab_model_info = st.tabs(["Prediction Details", "Model Info"])

    with tab_details:
        # Probability bar chart
        class_names = model.dls.vocab
        probs_pct = probabilities.numpy() * 100
        fig = px.bar(
            x=probs_pct,
            y=class_names,
            orientation="h",
            labels={"x": "Probability (%)", "y": "Class"},
            title="Class Probabilities",
            color=class_names,
            color_discrete_map={"NORMAL": "#4CAF50", "PNEUMONIA": "#f44336"},
        )
        fig.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)

    with tab_model_info:
        st.write(f"**Architecture:** {model_choice}")
        st.write(f"**Input Size:** 224 x 224 pixels")
        st.write(f"**Classes:** {', '.join(model.dls.vocab)}")

        # Try to load saved metrics
        metrics_file = Path(__file__).parent.parent / "outputs" / "metrics" / f"{arch_name}_metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file) as f:
                metrics = json.load(f)
            st.subheader("Test Set Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            m2.metric("Precision", f"{metrics['precision']:.1%}")
            m3.metric("Recall", f"{metrics['recall']:.1%}")
            m4.metric("F1 Score", f"{metrics['f1_score']:.1%}")

            if "auc_roc" in metrics:
                st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
        else:
            st.info("Run evaluation (`python src/evaluate.py`) to see test set metrics here.")

    # ─── Disclaimer ────────────────────────────────────────────────────────

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
            It is NOT a substitute for professional medical diagnosis. Always consult a qualified
            healthcare provider for medical decisions. AI predictions should be used as a
            supplementary tool, not as the sole basis for clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )
