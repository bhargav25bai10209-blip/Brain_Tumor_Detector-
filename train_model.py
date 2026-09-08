import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast,
    RandomBrightness, RandomTranslation
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
#  Grad-CAM Utilities
# ──────────────────────────────────────────────

def generate_gradcam_heatmap(model, img_preprocessed, class_idx):
    """
    Generate a Grad-CAM heatmap for a single preprocessed image.

    Uses the feature maps entering GlobalAveragePooling2D (= MobileNetV2's
    'out_relu' output, shape (1, 7, 7, 1280)) as the Grad-CAM target layer.

    Args:
        model: Loaded Keras Functional model.
        img_preprocessed: np.ndarray of shape (1, H, W, 3), already preprocessed.
        class_idx: Integer index of the class to explain.

    Returns:
        heatmap: np.ndarray of shape (7, 7), values in [0, 1].
                 Returns None if Grad-CAM cannot be built.
    """
    try:
        # The GAP layer's input tensor == MobileNetV2's feature-map output
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

    grads = tape.gradient(loss, conv_outputs)          # (1, 7, 7, 1280)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (1280,)

    conv_map = conv_outputs[0]                         # (7, 7, 1280)
    heatmap = conv_map @ pooled_grads[..., tf.newaxis] # (7, 7, 1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_rgb, heatmap, alpha=0.45):
    """
    Overlay a Grad-CAM heatmap on the original RGB image.

    Args:
        original_rgb: np.ndarray (H, W, 3), dtype uint8 or float32 [0-255].
        heatmap:      np.ndarray (h, w) with values in [0, 1].
        alpha:        Heatmap blend weight.

    Returns:
        superimposed: np.ndarray (H, W, 3), dtype uint8.
    """
    from PIL import Image as PILImage

    orig = np.array(original_rgb)
    if orig.dtype != np.uint8:
        orig = np.clip(orig, 0, 255).astype(np.uint8)

    H, W = orig.shape[:2]

    # Resize heatmap → image size
    heatmap_img = PILImage.fromarray(np.uint8(heatmap * 255))
    heatmap_img = heatmap_img.resize((W, H), PILImage.LANCZOS)
    heatmap_arr = np.array(heatmap_img) / 255.0

    # Apply JET colormap
    cmap = plt.get_cmap("jet")
    heatmap_colored = (cmap(heatmap_arr)[:, :, :3] * 255).astype(np.uint8)

    superimposed = (heatmap_colored * alpha + orig * (1 - alpha))
    return np.clip(superimposed, 0, 255).astype(np.uint8)


def save_gradcam_samples(model, test_dir, class_names, preprocess_fn,
                          img_size, output_path, num_per_class=2):
    """
    Generate and save a Grad-CAM visualization grid for sample test images.

    Args:
        model:         Trained Keras model.
        test_dir:      Path to the Testing directory.
        class_names:   List of class name strings.
        preprocess_fn: Preprocessing function (e.g. mobilenet_v2.preprocess_input).
        img_size:      Tuple (H, W).
        output_path:   Save path for the output PNG.
        num_per_class: Number of sample images per class.
    """
    from PIL import Image as PILImage

    samples = []
    for cls in class_names:
        cls_folder = os.path.join(test_dir, cls)
        if not os.path.exists(cls_folder):
            continue
        files = [f for f in os.listdir(cls_folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_per_class]
        for fname in files:
            samples.append((os.path.join(cls_folder, fname), cls))

    if not samples:
        print("Warning: No samples found for Grad-CAM visualization.")
        return

    n = len(samples)
    cols = 3   # original | heatmap | overlay
    rows = n

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for row, (img_path, true_cls) in enumerate(samples):
        # Load original image
        orig_img = PILImage.open(img_path).convert("RGB").resize(img_size)
        orig_arr = np.array(orig_img)

        # Preprocess
        img_exp = np.expand_dims(orig_arr, 0).astype(np.float32)
        img_prep = preprocess_fn(img_exp)

        # Predict
        preds = model.predict(img_prep, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_cls = class_names[pred_idx]
        conf = preds[pred_idx] * 100

        # Grad-CAM
        heatmap = generate_gradcam_heatmap(model, img_prep, pred_idx)
        overlay = overlay_gradcam(orig_arr, heatmap) if heatmap is not None else orig_arr

        title_color = "green" if true_cls == pred_cls else "red"

        # Column 0: Original
        axes[row, 0].imshow(orig_arr)
        axes[row, 0].set_title(f"True: {true_cls}", fontsize=9)
        axes[row, 0].axis("off")

        # Column 1: Heatmap only
        if heatmap is not None:
            axes[row, 1].imshow(heatmap, cmap="jet")
            axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=9)
        else:
            axes[row, 1].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=9)
        axes[row, 1].axis("off")

        # Column 2: Overlay
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"Pred: {pred_cls} ({conf:.1f}%)",
            fontsize=9, color=title_color, fontweight="bold"
        )
        axes[row, 2].axis("off")

    plt.suptitle("Grad-CAM Explanations — Brain Tumor MRI", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM visualization to: {output_path}")


# ──────────────────────────────────────────────
#  Main Training Pipeline
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Brain Tumor MRI Classification Model Training")
    print("=" * 60)

    base_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MRI_dataset")
    train_dir  = os.path.join(base_dir, "Training")
    test_dir   = os.path.join(base_dir, "Testing")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Hyperparameters ─────────────────────────────────
    BATCH_SIZE        = 32
    IMG_SIZE          = (224, 224)
    INITIAL_EPOCHS    = 10
    FINE_TUNE_EPOCHS  = 8
    TOTAL_EPOCHS      = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

    print(f"Loading datasets from: {base_dir}")

    # ── Load Datasets ────────────────────────────────────
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    class_names = train_dataset.class_names
    print(f"Detected {len(class_names)} classes: {class_names}")

    # Save class names mapping
    classes_path = os.path.join(script_dir, "class_names.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    print(f"Saved class names to: {classes_path}")

    # ── Data Augmentation Pipeline ────────────────────────
    # Enhanced augmentation for MRI robustness
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.12),
        RandomZoom(0.12),
        RandomContrast(0.1),
        RandomBrightness(0.15),
        RandomTranslation(height_factor=0.08, width_factor=0.08),
    ], name="data_augmentation")

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )
    train_dataset = train_dataset.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE
    ).prefetch(buffer_size=AUTOTUNE)

    validation_dataset = validation_dataset.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE
    ).prefetch(buffer_size=AUTOTUNE)

    # ── Build Model ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building Model Architecture...")
    print("=" * 60)

    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x      = base_model(inputs, training=False)
    x      = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x      = BatchNormalization(name="batch_norm")(x)
    x      = Dense(256, activation="relu", name="dense_256")(x)
    x      = Dropout(0.3, name="dropout")(x)
    outputs = Dense(len(class_names), activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="Brain_Tumor_Classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.summary()

    # ── Callbacks ────────────────────────────────────────
    checkpoint_path = os.path.join(script_dir, "brain_tumor_model.keras")
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_accuracy",
            mode="max", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
    ]

    # ── Phase 1: Feature Extraction ───────────────────────
    print("\n" + "=" * 60)
    print("Phase 1: Training Classification Head (Feature Extraction)...")
    print("=" * 60)

    history_phase1 = model.fit(
        train_dataset, epochs=INITIAL_EPOCHS,
        validation_data=validation_dataset, callbacks=callbacks
    )

    # ── Phase 2: Fine-Tuning ──────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Fine-Tuning Top Layers of Base Model...")
    print("=" * 60)

    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Total layers in base model: {len(base_model.layers)}")
    print(f"Fine-tuning from layer {fine_tune_at} to {len(base_model.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        train_dataset, epochs=TOTAL_EPOCHS,
        initial_epoch=len(history_phase1.epoch),
        validation_data=validation_dataset, callbacks=callbacks
    )

    # ── Combine Histories ─────────────────────────────────
    acc      = history_phase1.history["accuracy"]     + history_fine.history["accuracy"]
    val_acc  = history_phase1.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss_h   = history_phase1.history["loss"]         + history_fine.history["loss"]
    val_loss = history_phase1.history["val_loss"]     + history_fine.history["val_loss"]

    # Save training history as JSON
    history_data = {
        "accuracy": acc, "val_accuracy": val_acc,
        "loss": loss_h, "val_loss": val_loss,
        "fine_tune_start_epoch": len(history_phase1.epoch)
    }
    history_json_path = os.path.join(script_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved training history JSON to: {history_json_path}")

    # ── Evaluate on Test Set ──────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating Model on Test Dataset...")
    print("=" * 60)

    if os.path.exists(checkpoint_path):
        print("Loading best model weights from checkpoint...")
        model = tf.keras.models.load_model(checkpoint_path)

    test_loss, test_accuracy = model.evaluate(validation_dataset)
    print(f"\nFinal Test Loss:     {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%\n")

    # Save .h5 backup
    h5_path = os.path.join(script_dir, "brain_tumor_model.h5")
    try:
        model.save(h5_path)
        print(f"Saved model (.h5) to: {h5_path}")
    except Exception as e:
        print(f"Note on .h5 save: {e}")

    # ── Detailed Metrics ──────────────────────────────────
    print("Generating Predictions & Detailed Metrics on Test Data...")

    test_ds_for_eval = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    y_true, y_pred_probs = [], []
    for images, labels in test_ds_for_eval:
        images_prep = preprocess_input(images)
        preds = model.predict(images_prep, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(script_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Per-class accuracy breakdown
    print("\nPer-Class Accuracy Breakdown:")
    print("-" * 35)
    for i, cls in enumerate(class_names):
        mask      = y_true == i
        cls_acc   = np.mean(y_pred[mask] == y_true[mask]) * 100
        n_samples = int(mask.sum())
        print(f"  {cls:<14}: {cls_acc:6.2f}%  ({n_samples} samples)")
    print("-" * 35)

    # ── Confusion Matrix ──────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Brain Tumor Classification — Confusion Matrix", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(script_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved Confusion Matrix to: {cm_path}")

    # ── Training Curves ───────────────────────────────────
    ft_epoch = len(history_phase1.epoch) - 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(acc,     label="Training Accuracy",   color="#1f77b4", lw=2)
    ax1.plot(val_acc, label="Validation Accuracy", color="#ff7f0e", lw=2)
    ax1.axvline(x=ft_epoch, color="green", linestyle="--", label="Fine-Tune Start")
    ax1.legend(loc="lower right")
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Training and Validation Accuracy", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss_h,   label="Training Loss",   color="#1f77b4", lw=2)
    ax2.plot(val_loss, label="Validation Loss", color="#ff7f0e", lw=2)
    ax2.axvline(x=ft_epoch, color="green", linestyle="--", label="Fine-Tune Start")
    ax2.legend(loc="upper right")
    ax2.set_ylabel("Loss (Cross Entropy)", fontsize=11)
    ax2.set_title("Training and Validation Loss", fontsize=13)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(script_dir, "training_history.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved Training History plot to: {hist_path}")

    # ── Sample Prediction Grid ────────────────────────────
    print("Generating Sample Predictions Visualization...")
    sample_images, sample_labels, sample_preds, sample_confs = [], [], [], []

    for images, labels in test_ds_for_eval.take(1):
        images_prep = preprocess_input(images)
        preds = model.predict(images_prep, verbose=0)
        for i in range(min(12, len(images))):
            sample_images.append(images[i].numpy().astype("uint8"))
            sample_labels.append(class_names[labels[i].numpy()])
            pred_idx = np.argmax(preds[i])
            sample_preds.append(class_names[pred_idx])
            sample_confs.append(preds[i][pred_idx] * 100)

    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    axes = axes.flatten()
    for i in range(len(sample_images)):
        axes[i].imshow(sample_images[i])
        color = "green" if sample_labels[i] == sample_preds[i] else "red"
        axes[i].set_title(
            f"True: {sample_labels[i]}\nPred: {sample_preds[i]} ({sample_confs[i]:.1f}%)",
            color=color, fontsize=9, fontweight="bold"
        )
        axes[i].axis("off")

    plt.suptitle("Brain Tumor MRI — Sample Predictions on Test Set", fontsize=15, y=0.98)
    plt.tight_layout()
    sample_path = os.path.join(script_dir, "sample_predictions.png")
    plt.savefig(sample_path, dpi=300)
    plt.close()
    print(f"Saved Sample Predictions to: {sample_path}")

    # ── Grad-CAM Visualization ────────────────────────────
    print("\nGenerating Grad-CAM Explanations...")
    gradcam_path = os.path.join(script_dir, "gradcam_samples.png")
    save_gradcam_samples(
        model, test_dir, class_names, preprocess_input,
        img_size=IMG_SIZE, output_path=gradcam_path, num_per_class=2
    )

    # ── Done ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Model Training & Evaluation Complete!")
    print(f"  Best model (.keras) : {checkpoint_path}")
    print(f"  Confusion Matrix    : {cm_path}")
    print(f"  Training History    : {hist_path}")
    print(f"  Grad-CAM samples    : {gradcam_path}")
    print(f"  Sample Predictions  : {sample_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
