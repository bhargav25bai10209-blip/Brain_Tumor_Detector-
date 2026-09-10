"""
app.py — Brain Tumor MRI Classifier · Streamlit Web UI
========================================================
Launch with:
    streamlit run app.py
"""

import os
import io
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ── Page config (must be first Streamlit call) ─────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ─────────────────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  .stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2e 50%, #091624 100%);
    color: #e2e8f0;
  }

  /* ── Sidebar ─────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: rgba(15, 25, 50, 0.95);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
  }

  /* ── Cards ───────────────────────────────────────────── */
  .glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
  }
  .result-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 28px;
    border: 1px solid rgba(255, 255, 255, 0.10);
  }

  /* ── Title ───────────────────────────────────────────── */
  .main-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
  }
  .subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
  }

  /* ── Prediction badge ────────────────────────────────── */
  .pred-badge {
    display: inline-block;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 10px 22px;
    border-radius: 40px;
    margin: 8px 0 18px;
    letter-spacing: 0.03em;
  }
  .badge-glioma      { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }
  .badge-meningioma  { background: rgba(249,115,22,0.18); color: #fdba74; border: 1px solid rgba(249,115,22,0.4); }
  .badge-pituitary   { background: rgba(139,92,246,0.18); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.4); }
  .badge-notumor     { background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.4); }
  .badge-uncertain   { background: rgba(234,179,8,0.18); color: #fde047; border: 1px solid rgba(234,179,8,0.5); }

  /* ── Confidence meter ────────────────────────────────── */
  .conf-bar-wrapper {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 10px;
    width: 100%;
    margin: 4px 0 12px;
  }

  /* ── Disclaimer ─────────────────────────────────────── */
  .disclaimer {
    font-size: 0.75rem;
    color: #64748b;
    border-top: 1px solid rgba(255,255,255,0.07);
    padding-top: 14px;
    margin-top: 24px;
    line-height: 1.6;
  }

  /* ── Metric box ──────────────────────────────────────── */
  .metric-box {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.07);
  }
  .metric-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; margin-top: 4px; }

  /* ── Upload zone ─────────────────────────────────────── */
  [data-testid="stFileUploadDropzone"] {
    background: rgba(99,179,237,0.04) !important;
    border: 2px dashed rgba(99,179,237,0.3) !important;
    border-radius: 14px !important;
  }

  /* ── Tab styling ─────────────────────────────────────── */
  .stTabs [data-baseweb="tab"] {
    color: #94a3b8;
  }
  .stTabs [aria-selected="true"] {
    color: #63b3ed !important;
    border-bottom-color: #63b3ed !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model.keras")
H5_PATH    = os.path.join(BASE_DIR, "brain_tumor_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "class_names.json")
CONFIDENCE_THRESHOLD = 60.0

CLASS_CONFIG = {
    "glioma":     {"badge": "badge-glioma",     "icon": "🔴", "desc": "Glioma — arises from glial cells; most common primary brain tumor."},
    "meningioma": {"badge": "badge-meningioma",  "icon": "🟠", "desc": "Meningioma — originates in the meninges; usually benign."},
    "notumor":    {"badge": "badge-notumor",     "icon": "🟢", "desc": "No Tumor Detected — MRI appears normal."},
    "pituitary":  {"badge": "badge-pituitary",   "icon": "🟣", "desc": "Pituitary Tumor — forms near the pituitary gland."},
}

MODEL_METRICS = {
    "overall_accuracy": 88.56,
    "macro_avg": {"precision": 89.00, "recall": 88.56, "f1_score": 88.28},
    "weighted_avg": {"precision": 89.00, "recall": 88.56, "f1_score": 88.28},
    "classes": {
        "glioma":     {"name": "Glioma",     "precision": 95.56, "recall": 75.25, "f1": 84.20, "support": 400},
        "meningioma": {"name": "Meningioma", "precision": 85.00, "recall": 80.75, "f1": 82.82, "support": 400},
        "notumor":    {"name": "No Tumor",   "precision": 90.66, "recall": 99.50, "f1": 94.87, "support": 400},
        "pituitary":  {"name": "Pituitary",  "precision": 84.76, "recall": 98.75, "f1": 91.22, "support": 400},
    },
    "confusion_matrix": {
        "labels": ["glioma", "meningioma", "notumor", "pituitary"],
        "matrix": [
            [301, 53, 29, 17],
            [13, 323, 10, 54],
            [1, 1, 398, 0],
            [0, 3, 2, 395]
        ]
    }
}


# ── Model & utils loading ──────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def load_model_and_labels():
    """Load model once and cache across sessions."""
    import tensorflow as tf

    model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else H5_PATH
    if not os.path.exists(model_file):
        return None, None

    model = tf.keras.models.load_model(
        model_file,
        compile=False,
    )

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            class_names = json.load(f)
    else:
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    return model, class_names


def preprocess_image(pil_img):
    """Resize and apply MobileNetV2 preprocessing. Returns (orig_arr, prep_batch)."""
    import tensorflow as tf
    img_resized = pil_img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype(np.float32)
    batch = np.expand_dims(arr, 0)
    prep  = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    return np.array(img_resized, dtype=np.uint8), prep


def run_gradcam(model, img_prep, class_idx):
    """Return Grad-CAM heatmap array, or None."""
    import tensorflow as tf

    try:
        last_conv = model.get_layer("global_avg_pool").input
        grad_model = tf.keras.Model(inputs=model.inputs,
                                    outputs=[last_conv, model.output])
    except Exception:
        return None

    img_t = tf.cast(img_prep, tf.float32)
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(img_t, training=False)
        loss = preds[:, class_idx]

    grads        = tape.gradient(loss, conv_outs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map     = conv_outs[0]
    heatmap      = conv_map @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.nn.relu(heatmap)
    heatmap      = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(orig_rgb, heatmap, alpha=0.45):
    """Blend JET heatmap over original RGB image."""
    import matplotlib.pyplot as plt

    H, W = orig_rgb.shape[:2]
    hm   = Image.fromarray(np.uint8(heatmap * 255)).resize((W, H), Image.LANCZOS)
    hm_arr = np.array(hm) / 255.0
    cmap   = plt.get_cmap("jet")
    colored = (cmap(hm_arr)[:, :, :3] * 255).astype(np.uint8)
    blended = np.clip(colored * alpha + orig_rgb * (1 - alpha), 0, 255).astype(np.uint8)
    return blended


def pil_to_bytes(img_array):
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return buf.getvalue()


# ── Plotly confidence chart ────────────────────────────────

def make_confidence_chart(probs_dict, class_names):
    COLOR_MAP = {
        "glioma":     "#f87171",
        "meningioma": "#fb923c",
        "notumor":    "#4ade80",
        "pituitary":  "#a78bfa",
    }
    labels = class_names
    values = [probs_dict.get(c, 0.0) for c in labels]
    colors = [COLOR_MAP.get(c, "#94a3b8") for c in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=12),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94a3b8"),
            ticksuffix="%",
        ),
        yaxis=dict(
            tickfont=dict(color="#e2e8f0", size=13),
            categoryorder="array",
            categoryarray=list(reversed(labels)),
        ),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Brain Tumor Classifier")
    st.markdown("---")
    st.markdown("""
**Model Architecture**
- MobileNetV2 (ImageNet pretrained)
- 2-phase transfer learning
- Fine-tuned from layer 100

**Classes Detected**
- 🔴 Glioma
- 🟠 Meningioma
- 🟢 No Tumor
- 🟣 Pituitary Tumor

**Dataset**
- 5,600 Training images
- 1,600 Testing images
""")
    st.markdown("---")
    show_gradcam = st.toggle("🔥 Show Grad-CAM", value=True,
                              help="Highlight the regions the model focuses on.")
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#475569;'>"
        "Powered by TensorFlow · MobileNetV2"
        "</p>",
        unsafe_allow_html=True
    )

# ── Header ─────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🧠 Brain Tumor MRI Classifier</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload an MRI scan to instantly detect and classify brain tumors '
    'using a deep learning model trained on 5,600+ MRI scans.</p>',
    unsafe_allow_html=True
)

# ── Load model ─────────────────────────────────────────────
model, class_names = load_model_and_labels()

if model is None:
    st.error(
        "⚠️ **No trained model found.**\n\n"
        "Please train the model first by running:\n"
        "```bash\npython train_model.py\n```",
    )
    st.stop()

st.success(f"✅ Model loaded — {len(class_names)} classes: {', '.join(class_names)}")

# ── Two-column layout ──────────────────────────────────────
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("#### 📁 Upload MRI Scan")
    uploaded = st.file_uploader(
        label="Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")

        tab_orig, tab_info = st.tabs(["🖼️ MRI Image", "📋 Image Info"])
        with tab_orig:
            st.image(pil_img, use_container_width=True)
        with tab_info:
            st.markdown(f"""
| Property | Value |
|---|---|
| Filename | `{uploaded.name}` |
| Format | {pil_img.format or 'N/A'} |
| Size (px) | {pil_img.size[0]} × {pil_img.size[1]} |
| Mode | {pil_img.mode} |
""")

with col_right:
    if not uploaded:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:60px 20px;">'
            '<p style="font-size:3rem;margin:0">📤</p>'
            '<p style="color:#64748b;margin-top:12px;">Upload an MRI image on the left<br>to see the prediction here.</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # ── Inference ─────────────────────────────────────
        with st.spinner("Analysing MRI scan…"):
            orig_arr, img_prep = preprocess_image(pil_img)
            preds     = model.predict(img_prep, verbose=0)[0]
            pred_idx  = int(np.argmax(preds))
            pred_cls  = class_names[pred_idx]
            confidence = float(preds[pred_idx]) * 100
            uncertain  = confidence < CONFIDENCE_THRESHOLD

        probs_pct = {c: float(preds[i]) * 100 for i, c in enumerate(class_names)}
        cfg = CLASS_CONFIG.get(pred_cls, {})
        badge_cls = "badge-uncertain" if uncertain else cfg.get("badge", "")
        icon = cfg.get("icon", "❓")

        # ── Result card ───────────────────────────────────
        st.markdown("#### 🔬 Diagnosis Result")

        if uncertain:
            st.warning(
                f"⚠️ **Low Confidence ({confidence:.1f}%)** — "
                "result may be unreliable. Consult a radiologist.",
                icon="⚠️"
            )

        st.markdown(
            f'<div class="pred-badge {badge_cls}">'
            f'{icon} &nbsp; {pred_cls.upper()}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<p style="color:#94a3b8;font-size:0.9rem;margin-bottom:18px;">'
            f'{cfg.get("desc", "")}'
            f'</p>',
            unsafe_allow_html=True
        )

        # ── Metrics row ───────────────────────────────────
        cls_m = MODEL_METRICS["classes"].get(pred_cls, {})
        rec_val  = cls_m.get("recall", 0.0)
        f1_val   = cls_m.get("f1", 0.0)
        prec_val = cls_m.get("precision", 0.0)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Confidence</div>'
                f'<div class="metric-value">{confidence:.1f}%</div>'
                f'</div>', unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Class Recall</div>'
                f'<div class="metric-value">{rec_val:.1f}%</div>'
                f'</div>', unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Class F1-Score</div>'
                f'<div class="metric-value">{f1_val:.1f}%</div>'
                f'</div>', unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Test Accuracy</div>'
                f'<div class="metric-value">{MODEL_METRICS["overall_accuracy"]:.1f}%</div>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Model Performance & Confusion Matrix ───────────
        with st.expander("📊 Model Validation Metrics & Confusion Matrix (1,600 Test Scans)", expanded=True):
            st.markdown("##### 📈 Per-Class Performance Breakdown")
            metrics_data = []
            for c in ["glioma", "meningioma", "notumor", "pituitary"]:
                info = MODEL_METRICS["classes"][c]
                is_detected = "👉 " if c == pred_cls else ""
                metrics_data.append({
                    "Class": f"{is_detected}{info['name']}",
                    "Recall": f"{info['recall']:.2f}%",
                    "F1-Score": f"{info['f1']:.2f}%",
                    "Precision": f"{info['precision']:.2f}%",
                    "Test Samples": info["support"]
                })
            st.table(metrics_data)

            cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.markdown("##### 🔲 Confusion Matrix Heatmap")
                st.image(cm_path, caption="Brain Tumor Classification Confusion Matrix (1,600 Test Scans)", use_container_width=True)

        st.markdown("---")

        # ── Probability chart ─────────────────────────────
        st.markdown("**Class Probabilities**")
        fig = make_confidence_chart(probs_pct, class_names)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Grad-CAM section ──────────────────────────────
        if show_gradcam:
            st.markdown("---")
            st.markdown("**🔥 Grad-CAM — Model Attention Map**")
            st.caption(
                "Highlights regions that most influenced the prediction. "
                "Red/yellow = high attention, blue = low attention."
            )
            with st.spinner("Generating Grad-CAM…"):
                heatmap = run_gradcam(model, img_prep, pred_idx)

            if heatmap is not None:
                overlay = overlay_gradcam(orig_arr, heatmap)
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.image(orig_arr, caption="Original", use_container_width=True)
                with gc2:
                    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

                # Download button
                overlay_bytes = pil_to_bytes(overlay)
                st.download_button(
                    label="⬇️ Download Grad-CAM",
                    data=overlay_bytes,
                    file_name=f"{os.path.splitext(uploaded.name)[0]}_gradcam.png",
                    mime="image/png",
                )
            else:
                st.info("Grad-CAM is not available for this model configuration.")

        # ── Download JSON report ──────────────────────────
        st.markdown("---")
        report = {
            "image": uploaded.name,
            "predicted_class": pred_cls,
            "confidence_pct": round(confidence, 2),
            "uncertain": uncertain,
            "class_probabilities": {c: round(v, 2) for c, v in probs_pct.items()},
            "validation_metrics": {
                "detected_class_metrics": {
                    "class": pred_cls,
                    "recall_pct": rec_val,
                    "f1_score_pct": f1_val,
                    "precision_pct": prec_val,
                    "support": cls_m.get("support", 400)
                },
                "overall_test_accuracy_pct": MODEL_METRICS["overall_accuracy"],
                "macro_average_pct": MODEL_METRICS["macro_avg"],
                "per_class_summary": MODEL_METRICS["classes"],
                "confusion_matrix": MODEL_METRICS["confusion_matrix"]
            }
        }
        st.download_button(
            label="📄 Download Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"{os.path.splitext(uploaded.name)[0]}_report.json",
            mime="application/json",
        )

# ── Footer disclaimer ──────────────────────────────────────
st.markdown(
    '<div class="disclaimer">'
    '⚕️ <b>Medical Disclaimer:</b> This tool is intended for research and educational '
    'purposes only. It is <b>not a medical device</b> and must not be used as a substitute '
    'for professional medical advice, diagnosis, or treatment. Always consult a qualified '
    'radiologist or physician for clinical decisions.'
    '</div>',
    unsafe_allow_html=True
)
