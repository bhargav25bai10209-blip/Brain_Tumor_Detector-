"""
predict.py — Brain Tumor MRI Classifier Inference
==================================================
Usage:
  Single image:
    python predict.py path/to/mri.jpg
    python predict.py path/to/mri.jpg --gradcam --save
    python predict.py path/to/mri.jpg --plot --gradcam

  Batch (whole folder):
    python predict.py --batch path/to/folder --gradcam --save

  Sample test-set inference (no args):
    python predict.py
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# ──────────────────────────────────────────────
#  Confidence Threshold
# ──────────────────────────────────────────────
CONFIDENCE_WARNING_THRESHOLD = 60.0   # % — flag as "Uncertain" below this

# ──────────────────────────────────────────────
#  Grad-CAM Utilities
# ──────────────────────────────────────────────

def generate_gradcam_heatmap(model, img_preprocessed, class_idx):
    """
    Produce a Grad-CAM heatmap for the given class index.

    Args:
        model:            Loaded Keras Functional model.
        img_preprocessed: np.ndarray (1, H, W, 3), MobileNetV2-preprocessed.
        class_idx:        Integer class index to explain.

    Returns:
        heatmap: np.ndarray (7, 7) in [0, 1], or None on failure.
    """
    try:
        last_conv_output = model.get_layer("global_avg_pool").input
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_output, model.output]
        )
    except Exception:
        return None

    img_tensor = tf.cast(img_preprocessed, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, class_idx]

    grads       = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_map = conv_outputs[0]
    heatmap  = conv_map @ pooled_grads[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.nn.relu(heatmap)
    heatmap  = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_rgb, heatmap, alpha=0.45):
    """
    Blend a Grad-CAM heatmap (JET colormap) over the original image.

    Args:
        original_rgb: np.ndarray (H, W, 3) uint8 or float32.
        heatmap:      np.ndarray (h, w) in [0, 1].
        alpha:        Heatmap opacity.

    Returns:
        superimposed: np.ndarray (H, W, 3) uint8.
    """
    orig = np.array(original_rgb)
    if orig.dtype != np.uint8:
        orig = np.clip(orig, 0, 255).astype(np.uint8)

    H, W = orig.shape[:2]
    hm_img = Image.fromarray(np.uint8(heatmap * 255))
    hm_img = hm_img.resize((W, H), Image.LANCZOS)
    hm_arr = np.array(hm_img) / 255.0

    cmap    = plt.get_cmap("jet")
    colored = (cmap(hm_arr)[:, :, :3] * 255).astype(np.uint8)

    blended = colored * alpha + orig * (1 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_gradcam_overlay(original_rgb, heatmap, save_path, title=""):
    """Save a side-by-side figure: original | heatmap | overlay."""
    overlay = overlay_gradcam(original_rgb, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_rgb); axes[0].set_title("Original MRI");    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Overlay");              axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM overlay saved -> {save_path}")

# ──────────────────────────────────────────────
#  Model Loading
# ──────────────────────────────────────────────

def load_brain_tumor_model(model_path=None, labels_path=None):
    """Load the trained model and class-name list."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if model_path is None:
        keras_path = os.path.join(base_dir, "brain_tumor_model.keras")
        h5_path    = os.path.join(base_dir, "brain_tumor_model.h5")
        model_path = keras_path if os.path.exists(keras_path) else h5_path

    if labels_path is None:
        labels_path = os.path.join(base_dir, "class_names.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run train_model.py first."
        )

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    return model, class_names

# ──────────────────────────────────────────────
#  Single-Image Prediction
# ──────────────────────────────────────────────

def predict_image(image_path, model, class_names,
                  img_size=(224, 224),
                  show_plot=False,
                  generate_gradcam=False,
                  save_results=False):
    """
    Classify a single MRI image and (optionally) generate Grad-CAM.

    Args:
        image_path:       Absolute/relative path to the image.
        model:            Loaded Keras model.
        class_names:      List of class name strings.
        img_size:         Target resize tuple (H, W).
        show_plot:        If True, open a matplotlib window.
        generate_gradcam: If True, save a Grad-CAM overlay PNG.
        save_results:     If True, save a JSON results file.

    Returns:
        dict with prediction details.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input

    # Load & preprocess
    img_pil      = Image.open(image_path).convert("RGB")
    img_resized  = img_pil.resize(img_size)
    img_arr      = np.array(img_resized)
    img_expanded = np.expand_dims(img_arr, axis=0).astype(np.float32)
    img_prep     = preprocess_fn(img_expanded)

    # Predict
    preds         = model.predict(img_prep, verbose=0)[0]
    pred_idx      = int(np.argmax(preds))
    pred_class    = class_names[pred_idx]
    confidence    = float(preds[pred_idx]) * 100
    uncertain     = confidence < CONFIDENCE_WARNING_THRESHOLD

    # Build result dict
    results = {
        "image_path"        : image_path,
        "predicted_class"   : pred_class,
        "confidence_pct"    : round(confidence, 2),
        "uncertain"         : uncertain,
        "class_probabilities": {
            cls: round(float(preds[i]) * 100, 2)
            for i, cls in enumerate(class_names)
        },
    }

    # ── Pretty print ─────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"  Prediction for: {os.path.basename(image_path)}")
    print("=" * 52)
    if uncertain:
        print(f"  [!] LOW CONFIDENCE - result may be unreliable")
    print(f"  -> Predicted: {pred_class.upper()}  ({confidence:.2f}%)")
    print()
    print("  Class Probabilities:")
    for cls, prob in results["class_probabilities"].items():
        bar = "#" * int(prob / 4)
        print(f"    {cls:<14}: {prob:6.2f}%  {bar}")
    print("=" * 52)

    # ── Grad-CAM ──────────────────────────────────────────
    gradcam_path = None
    if generate_gradcam:
        print("  Generating Grad-CAM...")
        heatmap = generate_gradcam_heatmap(model, img_prep, pred_idx)
        if heatmap is not None:
            stem    = os.path.splitext(image_path)[0]
            gradcam_path = f"{stem}_gradcam.png"
            title = f"Pred: {pred_class} ({confidence:.1f}%)"
            if uncertain:
                title += "  [LOW CONFIDENCE]"
            save_gradcam_overlay(img_arr, heatmap, gradcam_path, title=title)
            results["gradcam_path"] = gradcam_path
        else:
            print("  Warning: Grad-CAM could not be generated for this model.")

    # ── Save JSON ─────────────────────────────────────────
    if save_results:
        stem     = os.path.splitext(image_path)[0]
        json_path = f"{stem}_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Result JSON saved -> {json_path}")
        results["result_json_path"] = json_path

    # ── Optional plot ─────────────────────────────────────
    if show_plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(img_pil)
        title_str = f"Prediction: {pred_class}  ({confidence:.2f}%)"
        if uncertain:
            title_str += "\n[!] Low Confidence"
        plt.title(title_str, fontsize=11, fontweight="bold",
                  color="orange" if uncertain else "black")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return results

# ──────────────────────────────────────────────
#  Batch Prediction
# ──────────────────────────────────────────────

def predict_batch(folder_path, model, class_names,
                  img_size=(224, 224),
                  generate_gradcam=False,
                  save_results=False):
    """
    Classify every image in a folder and print a summary table.

    Args:
        folder_path:      Directory containing MRI images.
        model:            Loaded Keras model.
        class_names:      List of class name strings.
        img_size:         Target resize tuple.
        generate_gradcam: If True, save Grad-CAM overlays.
        save_results:     If True, save per-image JSON results.

    Returns:
        List of result dicts.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not image_files:
        print(f"No valid images found in: {folder_path}")
        return []

    print(f"\nBatch prediction on {len(image_files)} image(s) in: {folder_path}")
    print("-" * 52)

    all_results = []
    for img_path in image_files:
        try:
            r = predict_image(
                img_path, model, class_names, img_size=img_size,
                show_plot=False,
                generate_gradcam=generate_gradcam,
                save_results=save_results
            )
            all_results.append(r)
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: {e}")

    # Summary
    from collections import Counter
    counts = Counter(r["predicted_class"] for r in all_results)
    uncertain_count = sum(1 for r in all_results if r.get("uncertain", False))

    print("\n" + "=" * 52)
    print("  BATCH SUMMARY")
    print("=" * 52)
    for cls, cnt in counts.most_common():
        print(f"  {cls:<14}: {cnt} image(s)")
    if uncertain_count:
        print(f"\n  [!] {uncertain_count} image(s) had low confidence (< {CONFIDENCE_WARNING_THRESHOLD}%)")
    print("=" * 52)

    # Save batch summary JSON
    if save_results:
        summary_path = os.path.join(folder_path, "batch_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Batch results JSON saved -> {summary_path}")

    return all_results

# ──────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI Classifier — Inference",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "image_path", nargs="?",
        help="Path to a single MRI image to classify."
    )
    parser.add_argument(
        "--batch", type=str, default=None, metavar="FOLDER",
        help="Classify all images in FOLDER."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model (.keras or .h5)."
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Path to class_names.json."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Display the image with prediction (single image mode)."
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Generate and save Grad-CAM overlay image(s)."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save prediction result(s) as JSON."
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Path to a single MRI image to classify (alias for positional arg)."
    )
    args = parser.parse_args()

    model, class_names = load_brain_tumor_model(args.model, args.labels)

    target_img = args.image or args.image_path

    # ── Batch Mode ────────────────────────────────────────
    if args.batch:
        predict_batch(
            args.batch, model, class_names,
            generate_gradcam=args.gradcam,
            save_results=args.save
        )
        return

    # ── Single Image Mode ─────────────────────────────────
    if target_img:
        predict_image(
            target_img, model, class_names,
            show_plot=args.plot,
            generate_gradcam=args.gradcam,
            save_results=args.save
        )
        return

    # ── Demo Mode (no args) ───────────────────────────────
    test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "MRI_dataset", "Testing"
    )
    if os.path.exists(test_dir):
        print("No image provided. Running sample inference on one image per class:\n")
        for cls in class_names:
            cls_folder = os.path.join(test_dir, cls)
            if os.path.exists(cls_folder):
                files = [f for f in os.listdir(cls_folder)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if files:
                    sample = os.path.join(cls_folder, files[0])
                    predict_image(
                        sample, model, class_names,
                        generate_gradcam=args.gradcam,
                        save_results=args.save
                    )
    else:
        print("Usage: python predict.py <path_to_mri_image>")
        print("       python predict.py --batch <folder>")


if __name__ == "__main__":
    main()
