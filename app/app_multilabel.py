"""Production-grade Streamlit app for multi-label chest X-ray classification."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from fastai.vision.all import PILImage, load_learner

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gradcam import GradCAM, overlay_gradcam
from src.model import get_target_layer
from src.data_multilabel import CLASSES

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CXR-14 Analyzer",
    page_icon="\U0001FA7B",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* ── Header ── */
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

/* ── Cards ── */
.metric-card {
    background: white; border-radius: 10px; padding: 1.2rem;
    border: 1px solid #e8eaed; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    text-align: center; transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1565c0; }
.metric-card .label { font-size: 0.8rem; color: #5f6368; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card.alert .value { color: #d32f2f; }
.metric-card.success .value { color: #2e7d32; }

/* ── Status Pills ── */
.pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.3px;
}
.pill-likely { background: #ffebee; color: #c62828; }
.pill-possible { background: #fff3e0; color: #e65100; }
.pill-unlikely { background: #e8f5e9; color: #2e7d32; }

/* ── Finding Row ── */
.finding-row {
    display: flex; align-items: center; padding: 10px 16px;
    border-radius: 8px; margin-bottom: 4px; transition: background 0.15s;
}
.finding-row:hover { background: #f5f7fa; }
.finding-row .disease-name { flex: 1; font-weight: 500; font-size: 0.9rem; color: #202124; }
.finding-row .prob-bar-container { flex: 1.5; padding: 0 16px; }
.finding-row .prob-value { width: 50px; text-align: right; font-size: 0.85rem; font-weight: 600; color: #5f6368; }
.finding-row .status { width: 80px; text-align: center; }

.prob-bar { height: 8px; border-radius: 4px; background: #e8eaed; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
.prob-bar-fill.likely { background: linear-gradient(90deg, #ef5350, #d32f2f); }
.prob-bar-fill.possible { background: linear-gradient(90deg, #ffa726, #f57c00); }
.prob-bar-fill.unlikely { background: linear-gradient(90deg, #66bb6a, #43a047); }

/* ── Section Headers ── */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #202124;
    padding-bottom: 8px; border-bottom: 2px solid #1565c0;
    margin-bottom: 1rem;
}

/* ── Upload Area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #c4c9d4 !important; border-radius: 12px !important;
    background: #fafbfc !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #f8f9fb; }
.sidebar-section { margin-bottom: 1.5rem; }
.sidebar-label { font-size: 0.75rem; font-weight: 600; color: #5f6368; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }

/* ── Disclaimer ── */
.disclaimer {
    background: #f8f9fa; border-left: 4px solid #1565c0; border-radius: 4px;
    padding: 12px 16px; margin-top: 2rem; font-size: 0.8rem; color: #5f6368; line-height: 1.5;
}
.disclaimer strong { color: #202124; }

/* ── Image container ── */
.img-container {
    background: #000; border-radius: 10px; overflow: hidden;
    border: 1px solid #e8eaed;
}

/* ── Summary Alert ── */
.summary-alert {
    padding: 12px 20px; border-radius: 10px; font-weight: 500;
    display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;
}
.summary-alert.warning { background: #fff3e0; border: 1px solid #ffcc02; color: #e65100; }
.summary-alert.safe { background: #e8f5e9; border: 1px solid #a5d6a7; color: #2e7d32; }

/* ── Hide Streamlit defaults ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "outputs" / "models"
METRICS_DIR = Path(__file__).parent.parent / "outputs" / "metrics"

AVAILABLE_MODELS = {
    "DenseNet121": ("densenet121", "densenet121_multilabel.pkl", "7.0M"),
    "ResNet50": ("resnet50", "resnet50_multilabel.pkl", "23.5M"),
    "ResNet34": ("resnet34", "resnet34_multilabel.pkl", "21.3M"),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_filename: str):
    model_path = MODELS_DIR / model_filename
    if model_path.exists():
        return load_learner(model_path)
    return None


@st.cache_data
def load_thresholds(arch_name: str) -> dict:
    path = METRICS_DIR / f"{arch_name}_multilabel_thresholds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {cls: 0.5 for cls in CLASSES}


@st.cache_data
def load_metrics(arch_name: str) -> dict:
    path = METRICS_DIR / f"{arch_name}_multilabel_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_available_models() -> dict:
    available = {}
    for display_name, (arch_name, filename, params) in AVAILABLE_MODELS.items():
        if (MODELS_DIR / filename).exists():
            available[display_name] = (arch_name, filename, params)
    return available


def get_severity(prob: float, threshold: float) -> tuple:
    if prob >= threshold:
        return "Likely", "likely", "pill-likely"
    elif prob >= threshold * 0.6:
        return "Possible", "possible", "pill-possible"
    else:
        return "Unlikely", "unlikely", "pill-unlikely"


def generate_gradcam(model, img, arch_name, class_idx):
    dl = model.dls.test_dl([img])
    batch = next(iter(dl))
    x = batch[0].to(next(model.model.parameters()).device)
    target_layer = get_target_layer(model, arch_name)
    gradcam_obj = GradCAM(model.model, target_layer)
    cam = gradcam_obj.generate(x, class_idx=class_idx)
    return cam, x, gradcam_obj


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-label">Model Configuration</div>', unsafe_allow_html=True)

    available = get_available_models()
    if not available:
        st.error("No trained models found.\n\nRun: `python -m src.train_multilabel`")
        st.stop()

    model_choice = st.selectbox(
        "Architecture",
        list(available.keys()),
        help="Select the CNN backbone for classification",
    )
    arch_name, model_filename, param_count = available[model_choice]
    st.caption(f"Parameters: {param_count}")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Explainability</div>', unsafe_allow_html=True)

    show_gradcam = st.toggle("Enable Grad-CAM", value=True)
    if show_gradcam:
        gradcam_disease = st.selectbox(
            "Target Disease",
            ["Auto (Top Finding)"] + CLASSES,
            help="Which disease's attention map to visualize",
        )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">System Info</div>', unsafe_allow_html=True)
    device = "GPU" if torch.cuda.is_available() else "CPU"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    st.caption(f"Device: {device}")
    if device == "GPU":
        st.caption(f"GPU: {gpu_name}")
    st.caption(f"Classes: {len(CLASSES)}")
    st.caption(f"Framework: PyTorch + fastai")

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="app-header">
    <h1>CXR-14 Analyzer</h1>
    <p>Multi-Disease Chest X-Ray Analysis System</p>
    <div class="model-badge">
        <span class="dot"></span>
        {model_choice} &middot; {param_count} params &middot; 14 pathologies
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────

model = load_model(model_filename)
if model is None:
    st.error(f"Failed to load model: {model_filename}")
    st.stop()

thresholds = load_thresholds(arch_name)

# ─── Upload Section ───────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not uploaded_file:
    # Hero state
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="value">14</div>
            <div class="label">Pathologies Detected</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="value">112K</div>
            <div class="label">Training Images (NIH)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        metrics = load_metrics(arch_name)
        auc_val = f"{metrics['macro']['auc_roc']:.3f}" if metrics else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{auc_val}</div>
            <div class="label">Macro AUC-ROC</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.info("Upload a chest X-ray image above to begin analysis.")
    st.stop()

# ─── Run Prediction ──────────────────────────────────────────────────────────

img = PILImage.create(uploaded_file)
pil_img = Image.open(uploaded_file).convert("RGB")

t0 = time.time()
with st.spinner("Analyzing..."):
    pred, pred_idx, raw_probs = model.predict(img)
    probs = raw_probs.numpy()
inference_time = time.time() - t0

sorted_indices = np.argsort(probs)[::-1]
n_likely = sum(1 for i in range(len(CLASSES)) if probs[i] >= thresholds.get(CLASSES[i], 0.5))
top_finding = CLASSES[sorted_indices[0]]
top_prob = probs[sorted_indices[0]]

# ─── Summary Metrics Row ─────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    card_class = "alert" if n_likely > 0 else "success"
    st.markdown(f"""
    <div class="metric-card {card_class}">
        <div class="value">{n_likely}</div>
        <div class="label">Findings Detected</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{top_finding.replace('_', ' ')}</div>
        <div class="label">Top Finding</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{top_prob:.1%}</div>
        <div class="label">Highest Probability</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{inference_time:.2f}s</div>
        <div class="label">Analysis Time</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ─── Summary Alert ────────────────────────────────────────────────────────────

if n_likely == 0:
    st.markdown("""
    <div class="summary-alert safe">
        No significant pathologies detected above threshold.
    </div>""", unsafe_allow_html=True)
else:
    findings_list = ", ".join(
        CLASSES[i].replace("_", " ") for i in sorted_indices[:n_likely]
        if probs[i] >= thresholds.get(CLASSES[i], 0.5)
    )
    st.markdown(f"""
    <div class="summary-alert warning">
        {n_likely} potential finding(s): {findings_list}
    </div>""", unsafe_allow_html=True)

# ─── Image + Grad-CAM Row ────────────────────────────────────────────────────

col_img, col_cam = st.columns(2)

with col_img:
    st.markdown('<div class="section-header">Original X-Ray</div>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)

with col_cam:
    st.markdown('<div class="section-header">Grad-CAM Attention Map</div>', unsafe_allow_html=True)
    if show_gradcam:
        try:
            if gradcam_disease == "Auto (Top Finding)":
                target_idx = int(sorted_indices[0])
                target_name = CLASSES[target_idx]
            else:
                target_idx = CLASSES.index(gradcam_disease)
                target_name = gradcam_disease

            cam, x, gradcam_obj = generate_gradcam(model, img, arch_name, target_idx)
            overlay = overlay_gradcam(pil_img, cam, alpha=0.4)
            st.image(overlay, use_container_width=True)
            prob_val = probs[target_idx]
            severity_label, _, pill_class = get_severity(prob_val, thresholds.get(target_name, 0.5))
            st.markdown(
                f'Showing: **{target_name.replace("_", " ")}** ({prob_val:.1%}) '
                f'<span class="pill {pill_class}">{severity_label}</span>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Grad-CAM unavailable: {e}")
            st.image(pil_img, use_container_width=True)
    else:
        st.image(pil_img, use_container_width=True, caption="Enable Grad-CAM in sidebar")

# ─── Probability Chart ────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Prediction Probabilities</div>', unsafe_allow_html=True)

bar_colors = []
labels_sorted = []
probs_sorted = []
threshold_lines = []

for idx in sorted_indices:
    cls = CLASSES[idx]
    prob = probs[idx]
    thresh = thresholds.get(cls, 0.5)
    severity, level, _ = get_severity(prob, thresh)

    color_map = {"likely": "#d32f2f", "possible": "#f57c00", "unlikely": "#43a047"}
    labels_sorted.append(cls.replace("_", " "))
    probs_sorted.append(prob * 100)
    bar_colors.append(color_map[level])
    threshold_lines.append(thresh * 100)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=labels_sorted[::-1],
    x=probs_sorted[::-1],
    orientation="h",
    marker_color=bar_colors[::-1],
    text=[f"{p:.1f}%" for p in probs_sorted[::-1]],
    textposition="outside",
    textfont=dict(size=11, color="#5f6368"),
))

for i, (label, thresh) in enumerate(zip(labels_sorted[::-1], threshold_lines[::-1])):
    fig.add_shape(
        type="line", x0=thresh, x1=thresh, y0=i - 0.4, y1=i + 0.4,
        line=dict(color="#333", width=1.5, dash="dot"),
    )

fig.update_layout(
    xaxis_title="Probability (%)", xaxis_range=[0, 105],
    height=480, margin=dict(l=160, r=60, t=10, b=40),
    showlegend=False, plot_bgcolor="white",
    xaxis=dict(gridcolor="#f0f0f0", zeroline=False),
    yaxis=dict(gridcolor="#f0f0f0"),
    font=dict(family="Inter, sans-serif"),
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Dotted lines indicate per-class optimized thresholds (F1-maximized)")

# ─── Detail Tabs ──────────────────────────────────────────────────────────────

tab_findings, tab_gradcam_grid, tab_model = st.tabs([
    "Detailed Findings", "Grad-CAM Comparison", "Model Performance"
])

with tab_findings:
    st.markdown('<div class="section-header">All 14 Pathologies</div>', unsafe_allow_html=True)

    # Build findings as HTML
    rows_html = ""
    for idx in sorted_indices:
        cls = CLASSES[idx]
        prob = probs[idx]
        thresh = thresholds.get(cls, 0.5)
        severity, level, pill_class = get_severity(prob, thresh)
        bar_pct = min(prob * 100, 100)

        rows_html += f"""
        <div class="finding-row">
            <div class="disease-name">{cls.replace('_', ' ')}</div>
            <div class="prob-bar-container">
                <div class="prob-bar">
                    <div class="prob-bar-fill {level}" style="width: {bar_pct}%"></div>
                </div>
            </div>
            <div class="prob-value">{prob:.1%}</div>
            <div class="status"><span class="pill {pill_class}">{severity}</span></div>
        </div>"""

    st.markdown(rows_html, unsafe_allow_html=True)

with tab_gradcam_grid:
    if show_gradcam:
        st.markdown('<div class="section-header">Top 6 Findings - Attention Maps</div>', unsafe_allow_html=True)
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
                    cls_name = CLASSES[idx].replace("_", " ")
                    prob_val = probs[idx]
                    severity, _, pill_class = get_severity(prob_val, thresholds.get(CLASSES[idx], 0.5))
                    st.image(overlay, use_container_width=True)
                    st.markdown(
                        f'**{cls_name}** ({prob_val:.1%}) '
                        f'<span class="pill {pill_class}">{severity}</span>',
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.warning(f"Grad-CAM grid unavailable: {e}")
    else:
        st.info("Enable Grad-CAM in the sidebar to see per-disease attention maps.")

with tab_model:
    st.markdown('<div class="section-header">Model Details</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""<div class="metric-card"><div class="value">{model_choice}</div><div class="label">Architecture</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><div class="value">{param_count}</div><div class="label">Parameters</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card"><div class="value">224x224</div><div class="label">Input Size</div></div>""", unsafe_allow_html=True)
    m4.markdown(f"""<div class="metric-card"><div class="value">112,120</div><div class="label">Training Images</div></div>""", unsafe_allow_html=True)

    metrics = load_metrics(arch_name)
    if metrics:
        st.markdown("")
        st.markdown('<div class="section-header">Test Set Performance</div>', unsafe_allow_html=True)

        macro = metrics.get("macro", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Macro AUC-ROC", f"{macro.get('auc_roc', 0):.4f}")
        m2.metric("Macro Precision", f"{macro.get('precision', 0):.1%}")
        m3.metric("Macro Recall", f"{macro.get('recall', 0):.1%}")
        m4.metric("Macro F1", f"{macro.get('f1_score', 0):.1%}")

        st.markdown("")
        st.markdown('<div class="section-header">Per-Class Metrics</div>', unsafe_allow_html=True)

        per_class = metrics.get("per_class", {})
        if per_class:
            df = pd.DataFrame([
                {
                    "Disease": cls.replace("_", " "),
                    "AUC-ROC": round(m["auc_roc"], 4),
                    "Precision": f"{m['precision']:.1%}",
                    "Recall": f"{m['recall']:.1%}",
                    "F1-Score": f"{m['f1_score']:.1%}",
                    "Threshold": f"{m['threshold']:.2f}",
                    "Prevalence": f"{m['prevalence']:.1%}",
                }
                for cls, m in per_class.items()
            ])
            st.dataframe(
                df.style.background_gradient(subset=["AUC-ROC"], cmap="RdYlGn", vmin=0.6, vmax=1.0),
                use_container_width=True, hide_index=True, height=540,
            )
    else:
        st.info("Run `python -m src.evaluate_multilabel` to see test metrics.")

# ─── Disclaimer ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="disclaimer">
    <strong>Disclaimer:</strong> This system is designed for educational and research purposes only.
    It is not intended as a substitute for professional medical diagnosis or clinical decision-making.
    AI-generated predictions are based on statistical patterns learned from training data and may not
    generalize across all clinical scenarios, imaging protocols, or patient populations.
    Always consult a qualified healthcare provider for medical decisions.
</div>
""", unsafe_allow_html=True)
