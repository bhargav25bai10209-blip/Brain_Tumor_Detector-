# 🧠 Brain Tumor MRI Classifier

A fully client-side AI web application that classifies brain MRI scans into 4 categories using TFLite WebAssembly — **no server, no backend, no data leaves your browser.**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/bhargav25bai10209-blip/Brain_Tumor_Detector-)

## 🔬 Classes

| Class | Description |
|---|---|
| 🔴 **Glioma** | Arises from glial cells; most common primary brain tumor |
| 🟠 **Meningioma** | Originates in the meninges; usually benign & slow-growing |
| 🟢 **No Tumor** | MRI appears within normal intracranial anatomy |
| 🟣 **Pituitary Tumor** | Forms near the pituitary gland; highly treatable |

## 🏗️ Architecture

- **Model**: MobileNetV2 (ImageNet pretrained) → TFLite flat-buffer (`model.tflite`)
- **Runtime**: `@tensorflow/tfjs-tflite` WebAssembly — runs entirely in-browser
- **Input**: `[1, 224, 224, 3]` float32 · **Output**: `[1, 4]` softmax logits
- **Training**: 2-phase transfer learning + fine-tuning from layer 100
- **Dataset**: 5,600 training / 1,600 testing MRI images (4 balanced classes)
- **Calibration**: Temperature scaling (T = 0.7273) for calibrated confidence
- **OOD Detection**: Shannon entropy threshold to reject non-MRI images

## 📊 Model Performance

| Metric | Value |
|---|---|
| Overall Test Accuracy | 81.81% |
| Macro Avg F1-Score | 81.34% |
| Macro Avg Precision | 81.61% |
| Macro Avg Recall | 81.81% |

## 🚀 Deployment

This app is a **pure static site** — `index.html` + `model.tflite` + sample images.

```
Vercel (primary)       → https://brain-tumor-classifier.vercel.app
Hugging Face Spaces    → https://huggingface.co/spaces/BhargavRajS/Brain_Tumor_Detector_Revised
GitHub Repository      → https://github.com/bhargav25bai10209-blip/Brain_Tumor_Detector-
```

### Files served by Vercel

| File | Purpose |
|---|---|
| `index.html` | Full app UI + inference logic |
| `model.tflite` | TFLite model (10 MB, cached 1 year) |
| `samples/*.jpg` | 4 built-in sample MRI scans |
| `confusion_matrix.png` | Validation confusion matrix image |
| `class_names.json` | Class label mapping |
| `temperature.json` | Temperature scaling factor |
| `training_history.json` | Training loss/accuracy curves |

## ⚕️ Medical Disclaimer

This tool is intended for **research and educational purposes only**. It is **not a medical device** and must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified radiologist or physician for clinical decisions.