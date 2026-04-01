"""Production-grade Streamlit app for chest X-ray pneumonia detection."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from fastai.vision.all import PILImage, load_learner

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gradcam import GradCAM, overlay_gradcam
from src.model import get_target_layer

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CXR Pneumonia Detector",
    page_icon="\U0001FA7B",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

.app-header {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
    color: white; padding: 1.5rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.app-header::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px; border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.app-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
.app-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.15); padding: 4px 12px;
    border-radius: 20px; font-size: 0.8rem; margin-top: 0.5rem;
}
.model-badge .dot { width: 8px; height: 8px; background: #69f0ae; border-radius: 50%; }

.metric-card {
    background: white; border-radius: 10px; padding: 1.2rem;
    border: 1px solid #e8eaed; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    text-align: center;
}
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1565c0; }
.metric-card .label { font-size: 0.8rem; color: #5f6368; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card.danger .value { color: #d32f2f; }
.metric-card.safe .value { color: #2e7d32; }

.section-header {
    font-size: 1.1rem; font-weight: 600; color: #202124;
    padding-bottom: 8px; border-bottom: 2px solid #1565c0;
    margin-bottom: 1rem;
}

.result-banner {
    padding: 16px 24px; border-radius: 10px; font-size: 1.2rem;
    font-weight: 600; text-align: center; margin: 1rem 0;
}
.result-banner.pneumonia { background: #ffebee; border: 1px solid #ef9a9a; color: #c62828; }
.result-banner.normal { background: #e8f5e9; border: 1px solid #a5d6a7; color: #2e7d32; }

.disclaimer {
    background: #f8f9fa; border-left: 4px solid #1565c0; border-radius: 4px;
    padding: 12px 16px; margin-top: 2rem; font-size: 0.8rem; color: #5f6368; line-height: 1.5;
}
.disclaimer strong { color: #202124; }

.sidebar-label { font-size: 0.75rem; font-weight: 600; color: #5f6368; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "outputs" / "models"
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "chest_xray" / "test"

AVAILABLE_MODELS = {
    "DenseNet121": ("densenet121", "densenet121_best.pkl", "7.0M"),
    "ResNet50": ("resnet50", "resnet50_best.pkl", "23.5M"),
    "ResNet34": ("resnet34", "resnet34_best.pkl", "21.3M"),
}


@st.cache_resource
def load_model(model_filename: str):
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        fallback = Path(__file__).parent.parent / "Pneumonia_analizer.pkl"
        if fallback.exists():
            return load_learner(fallback)
        return None
    return load_learner(model_path)


def get_available_models() -> dict:
    available = {}
    for display_name, (arch_name, filename, params) in AVAILABLE_MODELS.items():
        if (MODELS_DIR / filename).exists():
            available[display_name] = (arch_name, filename, params)
    fallback = Path(__file__).parent.parent / "Pneumonia_analizer.pkl"
    if not available and fallback.exists():
        available["ResNet34 (Original)"] = ("resnet34", "Pneumonia_analizer.pkl", "21.3M")
    return available


def get_sample_images(n: int = 3) -> dict:
    samples = {"NORMAL": [], "PNEUMONIA": []}
    for label in samples:
        folder = SAMPLE_DIR / label
        if folder.exists():
            files = sorted(folder.glob("*.jpeg"))[:n]
            samples[label] = files
    return samples


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-label">Model Configuration</div>', unsafe_allow_html=True)

    available = get_available_models()
    if not available:
        st.error("No trained models found.\n\nRun: `python -m src.train`")
        st.stop()

    model_choice = st.selectbox("Architecture", list(available.keys()))
    arch_name, model_filename, param_count = available[model_choice]
    st.caption(f"Parameters: {param_count}")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)

    confidence_threshold = st.slider(
        "Confidence Threshold", min_value=0.5, max_value=0.99,
        value=0.75, step=0.05,
        help="Predictions below this threshold are flagged as uncertain.",
    )
    show_gradcam = st.toggle("Enable Grad-CAM", value=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">System Info</div>', unsafe_allow_html=True)
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.caption(f"Device: {device}")
    if torch.cuda.is_available():
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    st.caption("Task: Binary Classification")

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="app-header">
    <h1>CXR Pneumonia Detector</h1>
    <p>AI-Powered Chest X-Ray Analysis for Pneumonia Detection</p>
    <div class="model-badge">
        <span class="dot"></span>
        {model_choice} &middot; {param_count} params &middot; Binary Classification
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model(model_filename)
if model is None:
    st.error(f"Failed to load model: {model_filename}")
    st.stop()

# ─── Image Input ──────────────────────────────────────────────────────────────

tab_upload, tab_samples = st.tabs(["Upload Image", "Sample Images"])

uploaded_file = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with tab_samples:
    samples = get_sample_images()
    if any(samples.values()):
        st.write("Select a sample image:")
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
        st.info("No sample images found. Download dataset to `data/chest_xray/`.")

image_source = None
if uploaded_file is not None:
    image_source = uploaded_file
elif "sample_file" in st.session_state:
    image_source = st.session_state["sample_file"]

if not image_source:
    st.markdown("---")
    st.info("Upload a chest X-ray image or select a sample above to begin.")
    st.stop()

# ─── Prediction ──────────────────────────────────────────────────────────────

img = PILImage.create(image_source)
pil_img = Image.open(image_source).convert("RGB") if isinstance(image_source, str) else Image.open(image_source).convert("RGB")

t0 = time.time()
with st.spinner("Analyzing..."):
    prediction, pred_idx, probabilities = model.predict(img)
    confidence = probabilities[pred_idx].item()
inference_time = time.time() - t0

is_pneumonia = prediction == "PNEUMONIA"

# ─── Result Banner ────────────────────────────────────────────────────────────

if is_pneumonia:
    st.markdown(f'<div class="result-banner pneumonia">PNEUMONIA DETECTED &mdash; Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="result-banner normal">NORMAL &mdash; Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)

# ─── Metrics Row ──────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    card_class = "danger" if is_pneumonia else "safe"
    st.markdown(f"""<div class="metric-card {card_class}"><div class="value">{prediction}</div><div class="label">Prediction</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="value">{confidence:.1%}</div><div class="label">Confidence</div></div>""", unsafe_allow_html=True)
with c3:
    status = "PASS" if confidence >= confidence_threshold else "REVIEW"
    card_class = "" if confidence >= confidence_threshold else "danger"
    st.markdown(f"""<div class="metric-card {card_class}"><div class="value">{status}</div><div class="label">Threshold Check</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><div class="value">{inference_time:.2f}s</div><div class="label">Analysis Time</div></div>""", unsafe_allow_html=True)

st.markdown("")

# ─── Image + Grad-CAM ────────────────────────────────────────────────────────

col_img, col_cam = st.columns(2)

with col_img:
    st.markdown('<div class="section-header">Original X-Ray</div>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)

with col_cam:
    st.markdown('<div class="section-header">Grad-CAM Attention Map</div>', unsafe_allow_html=True)
    if show_gradcam:
        try:
            dl = model.dls.test_dl([img])
            batch = next(iter(dl))
            x = batch[0].to(next(model.model.parameters()).device)
            target_layer = get_target_layer(model, arch_name)
            gradcam = GradCAM(model.model, target_layer)
            cam = gradcam.generate(x, class_idx=pred_idx)
            overlay = overlay_gradcam(pil_img, cam, alpha=0.4)
            st.image(overlay, use_container_width=True)
            st.caption("Highlighted regions indicate areas most influential for the prediction.")
        except Exception as e:
            st.warning(f"Grad-CAM unavailable: {e}")
            st.image(pil_img, use_container_width=True)
    else:
        st.image(pil_img, use_container_width=True, caption="Enable Grad-CAM in sidebar")

# ─── Details ──────────────────────────────────────────────────────────────────

tab_details, tab_model_info = st.tabs(["Prediction Details", "Model Performance"])

with tab_details:
    class_names = list(model.dls.vocab)
    probs_pct = probabilities.numpy() * 100

    colors = ["#43a047" if c == "NORMAL" else "#d32f2f" for c in class_names]

    fig = go.Figure(go.Bar(
        y=class_names, x=probs_pct, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probs_pct],
        textposition="outside",
        textfont=dict(size=13, color="#5f6368"),
    ))
    fig.update_layout(
        xaxis_title="Probability (%)", xaxis_range=[0, 110],
        height=200, margin=dict(l=100, r=60, t=10, b=40),
        showlegend=False, plot_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_model_info:
    st.markdown('<div class="section-header">Model Details</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="metric-card"><div class="value">{model_choice}</div><div class="label">Architecture</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><div class="value">{param_count}</div><div class="label">Parameters</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card"><div class="value">224x224</div><div class="label">Input Size</div></div>""", unsafe_allow_html=True)

    metrics_file = Path(__file__).parent.parent / "outputs" / "metrics" / f"{arch_name}_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        st.markdown("")
        st.markdown('<div class="section-header">Test Set Performance</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}")
        m3.metric("Recall", f"{metrics['recall']:.1%}")
        m4.metric("F1 Score", f"{metrics['f1_score']:.1%}")
        if "auc_roc" in metrics:
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
    else:
        st.info("Run `python -m src.evaluate` to see test set metrics.")

# ─── Disclaimer ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="disclaimer">
    <strong>Disclaimer:</strong> This system is designed for educational and research purposes only.
    It is not intended as a substitute for professional medical diagnosis.
    Always consult a qualified healthcare provider for medical decisions.
</div>
""", unsafe_allow_html=True)
