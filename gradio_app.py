"""
gradio_app.py — Brain Tumor MRI Classifier · Gradio Web UI
===========================================================
Deployed on Hugging Face Spaces.
"""

import os
import io
import json
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ───────────────────────────────────────────────
IMG_SIZE             = (224, 224)
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH           = os.path.join(BASE_DIR, "brain_tumor_model.keras")
H5_PATH              = os.path.join(BASE_DIR, "brain_tumor_model.h5")
LABELS_PATH          = os.path.join(BASE_DIR, "class_names.json")
CONFIDENCE_THRESHOLD = 60.0

CLASS_CONFIG = {
    "glioma":     {"color": "#f87171", "icon": "🔴", "desc": "Arises from glial cells; most common primary brain tumor."},
    "meningioma": {"color": "#fb923c", "icon": "🟠", "desc": "Originates in the meninges; usually benign."},
    "notumor":    {"color": "#4ade80", "icon": "🟢", "desc": "No Tumor Detected — MRI appears normal."},
    "pituitary":  {"color": "#a78bfa", "icon": "🟣", "desc": "Forms near the pituitary gland; often treatable."},
}

# ── Model loading ────────────────────────────────────────────
_model       = None
_class_names = None

def get_model():
    global _model, _class_names
    if _model is not None:
        return _model, _class_names

    import tensorflow as tf

    model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else H5_PATH
    if not os.path.exists(model_file):
        return None, None

    _model = tf.keras.models.load_model(model_file, compile=False)

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            _class_names = json.load(f)
    else:
        _class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    return _model, _class_names


# ── Image preprocessing ─────────────────────────────────────
def preprocess_image(pil_img):
    import tensorflow as tf
    img_resized = pil_img.resize(IMG_SIZE)
    arr         = np.array(img_resized).astype(np.float32)
    batch       = np.expand_dims(arr, 0)
    prep        = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    return np.array(img_resized, dtype=np.uint8), prep


# ── Grad-CAM ─────────────────────────────────────────────────
def run_gradcam(model, img_prep, class_idx):
    import tensorflow as tf
    try:
        last_conv  = model.get_layer("global_avg_pool").input
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
    H, W   = orig_rgb.shape[:2]
    hm     = Image.fromarray(np.uint8(heatmap * 255)).resize((W, H), Image.LANCZOS)
    hm_arr = np.array(hm) / 255.0
    cmap   = plt.get_cmap("jet")
    colored = (cmap(hm_arr)[:, :, :3] * 255).astype(np.uint8)
    blended = np.clip(colored * alpha + orig_rgb * (1 - alpha), 0, 255).astype(np.uint8)
    return blended


# ── Plotly confidence chart ──────────────────────────────────
def make_confidence_chart(probs_dict, class_names):
    labels = class_names
    values = [probs_dict.get(c, 0.0) for c in labels]
    colors = [CLASS_CONFIG.get(c, {}).get("color", "#94a3b8") for c in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(15,23,42,0)",
        plot_bgcolor="rgba(15,23,42,0)",
        height=220,
        margin=dict(l=0, r=70, t=10, b=10),
        xaxis=dict(
            range=[0, 115],
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


# ── Main inference function ──────────────────────────────────
def predict(pil_img, show_gradcam):
    """
    Called by Gradio on every submission.
    Returns: result_html, chart_fig, gradcam_img (or None), report_json_path
    """
    model, class_names = get_model()

    if model is None:
        return (
            "<p style='color:#f87171;font-size:1.1rem'>⚠️ Model file not found.</p>",
            None, None, None
        )

    if pil_img is None:
        return (
            "<p style='color:#94a3b8;font-size:1rem'>👆 Upload an MRI scan to get started.</p>",
            None, None, None
        )

    # Convert to RGB PIL
    if not isinstance(pil_img, Image.Image):
        pil_img = Image.fromarray(pil_img)
    pil_img = pil_img.convert("RGB")

    orig_arr, img_prep = preprocess_image(pil_img)
    preds      = model.predict(img_prep, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    pred_cls   = class_names[pred_idx]
    confidence = float(preds[pred_idx]) * 100
    uncertain  = confidence < CONFIDENCE_THRESHOLD

    probs_pct  = {c: float(preds[i]) * 100 for i, c in enumerate(class_names)}
    cfg        = CLASS_CONFIG.get(pred_cls, {})
    icon       = cfg.get("icon", "❓")
    color      = cfg.get("color", "#e2e8f0")
    desc       = cfg.get("desc", "")

    # ── HTML result card ────────────────────────────────────
    badge_bg   = "rgba(234,179,8,0.18)" if uncertain else f"{color}22"
    badge_border = "rgba(234,179,8,0.5)"  if uncertain else f"{color}66"
    badge_color  = "#fde047" if uncertain else color
    display_cls  = "UNCERTAIN" if uncertain else pred_cls.upper()

    warning_html = ""
    if uncertain:
        warning_html = f"""
        <div style="background:rgba(234,179,8,0.12);border:1px solid rgba(234,179,8,0.4);
                    border-radius:10px;padding:10px 16px;margin-bottom:14px;
                    color:#fde047;font-size:0.88rem;">
          ⚠️ <b>Low Confidence ({confidence:.1f}%)</b> — result may be unreliable. Consult a radiologist.
        </div>"""

    result_html = f"""
    <div style="font-family:'Inter',sans-serif;color:#e2e8f0;">
      {warning_html}
      <div style="display:inline-block;background:{badge_bg};border:1px solid {badge_border};
                  border-radius:40px;padding:10px 24px;font-size:1.4rem;font-weight:700;
                  color:{badge_color};letter-spacing:0.03em;margin-bottom:14px;">
        {icon}&nbsp; {display_cls}
      </div>
      <p style="color:#94a3b8;font-size:0.92rem;margin:0 0 18px;">{desc}</p>

      <div style="display:flex;gap:14px;margin-bottom:10px;">
        <div style="flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:14px 18px;text-align:center;">
          <div style="font-size:0.74rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;">Confidence</div>
          <div style="font-size:1.9rem;font-weight:700;color:#e2e8f0;margin-top:4px;">{confidence:.1f}%</div>
        </div>
        <div style="flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:14px 18px;text-align:center;">
          <div style="font-size:0.74rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;">Classes</div>
          <div style="font-size:1.9rem;font-weight:700;color:#e2e8f0;margin-top:4px;">{len(class_names)}</div>
        </div>
        <div style="flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:14px 18px;text-align:center;">
          <div style="font-size:0.74rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;">Status</div>
          <div style="font-size:1.9rem;font-weight:700;margin-top:4px;">{"⚠️" if uncertain else "✅"}</div>
        </div>
      </div>

      <p style="font-size:0.72rem;color:#475569;border-top:1px solid rgba(255,255,255,0.07);
               padding-top:12px;margin-top:8px;line-height:1.6;">
        ⚕️ <b>Medical Disclaimer:</b> This tool is for research and educational purposes only.
        It is <b>not a medical device</b> and must not replace professional medical advice.
        Always consult a qualified radiologist or physician for clinical decisions.
      </p>
    </div>
    """

    # ── Confidence chart ────────────────────────────────────
    chart = make_confidence_chart(probs_pct, class_names)

    # ── Grad-CAM ────────────────────────────────────────────
    gradcam_img = None
    if show_gradcam:
        heatmap = run_gradcam(model, img_prep, pred_idx)
        if heatmap is not None:
            overlay     = overlay_gradcam(orig_arr, heatmap)
            # Side-by-side figure
            fig_gc, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4),
                                               facecolor="#0a0f1e")
            for ax, img_data, title in [
                (ax1, orig_arr,  "Original MRI"),
                (ax2, overlay,   "Grad-CAM Overlay"),
            ]:
                ax.imshow(img_data)
                ax.set_title(title, color="#e2e8f0", fontsize=11, pad=8)
                ax.axis("off")
                ax.set_facecolor("#0a0f1e")
            plt.tight_layout(pad=1.0)
            buf = io.BytesIO()
            plt.savefig(buf, format="PNG", dpi=150,
                        facecolor="#0a0f1e", bbox_inches="tight")
            plt.close(fig_gc)
            buf.seek(0)
            gradcam_img = Image.open(buf).copy()

    # ── JSON report ─────────────────────────────────────────
    report = {
        "predicted_class":     pred_cls,
        "confidence_pct":      round(confidence, 2),
        "uncertain":           uncertain,
        "class_probabilities": {c: round(v, 2) for c, v in probs_pct.items()},
    }
    report_path = os.path.join(BASE_DIR, "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return result_html, chart, gradcam_img, report_path


# ── Custom CSS ───────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2e 50%, #091624 100%) !important;
    color: #e2e8f0 !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 32px 20px 20px;
}
.app-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.app-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin: 8px 0 0;
}

/* Panels */
.gr-panel, .gr-box, .gr-form, .gr-block {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}

/* Upload zone */
.gr-file-drop, [data-testid="image"], .upload-container {
    background: rgba(99,179,237,0.04) !important;
    border: 2px dashed rgba(99,179,237,0.3) !important;
    border-radius: 14px !important;
}

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.gr-button-primary:hover { opacity: 0.88 !important; }

/* Checkboxes, labels */
label, .gr-checkbox-label { color: #cbd5e1 !important; }

/* Hide Gradio footer */
footer { display: none !important; }

/* Info box */
.info-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 22px;
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.7;
}
"""

# ── Build Gradio UI ──────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="Brain Tumor MRI Classifier") as demo:

    # Header
    gr.HTML("""
    <div class="app-header">
      <h1 class="app-title">🧠 Brain Tumor MRI Classifier</h1>
      <p class="app-subtitle">
        Upload an MRI scan to instantly detect and classify brain tumors using a
        deep learning model trained on 5,600+ MRI scans.
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── Left column — inputs ───────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📁 Upload MRI Scan",
                type="pil",
                height=320,
            )
            show_gradcam = gr.Checkbox(
                label="🔥 Show Grad-CAM Attention Map",
                value=True,
                info="Highlights regions the model focuses on (red = high attention)."
            )
            run_btn = gr.Button("🔬 Analyse MRI", variant="primary", size="lg")

            # Info card
            gr.HTML("""
            <div class="info-box">
              <b>Model Architecture</b><br>
              MobileNetV2 (ImageNet pretrained)<br>
              2-phase transfer learning · Fine-tuned from layer 100<br><br>
              <b>Classes Detected</b><br>
              🔴 Glioma &nbsp; 🟠 Meningioma &nbsp; 🟢 No Tumor &nbsp; 🟣 Pituitary<br><br>
              <b>Dataset</b><br>
              5,600 training · 1,600 testing images
            </div>
            """)

        # ── Right column — outputs ─────────────────────────
        with gr.Column(scale=1):
            result_html  = gr.HTML(
                label="🔬 Diagnosis Result",
                value="<p style='color:#475569;font-family:Inter,sans-serif;"
                      "padding:40px 0;text-align:center;font-size:1rem;'>"
                      "📤 Upload an MRI image and click <b>Analyse MRI</b> to see results.</p>"
            )
            chart_output = gr.Plot(label="📊 Class Probabilities")
            gradcam_out  = gr.Image(label="🔥 Grad-CAM — Model Attention", visible=True)
            report_dl    = gr.File(label="📄 Download Report (JSON)", visible=True)

    # ── Wire up ────────────────────────────────────────────
    run_btn.click(
        fn=predict,
        inputs=[image_input, show_gradcam],
        outputs=[result_html, chart_output, gradcam_out, report_dl],
    )
    # Also trigger on image upload (optional auto-predict)
    image_input.change(
        fn=predict,
        inputs=[image_input, show_gradcam],
        outputs=[result_html, chart_output, gradcam_out, report_dl],
    )

if __name__ == "__main__":
    demo.launch()
