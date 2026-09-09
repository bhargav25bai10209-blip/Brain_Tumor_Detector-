---
title: Brain Tumor MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: gradio_app.py
pinned: false
license: mit
---

# 🧠 Brain Tumor MRI Classifier

A deep learning model that classifies brain MRI scans into 4 categories:

- 🔴 **Glioma** — arises from glial cells; most common primary brain tumor
- 🟠 **Meningioma** — originates in the meninges; usually benign
- 🟢 **No Tumor** — MRI appears normal
- 🟣 **Pituitary Tumor** — forms near the pituitary gland

## Model Architecture

- **Base**: MobileNetV2 (ImageNet pretrained)
- **Training**: 2-phase transfer learning + fine-tuning from layer 100
- **Dataset**: 5,600 training / 1,600 testing MRI images
- **Features**: Grad-CAM attention maps, confidence scores, JSON report export

## ⚕️ Medical Disclaimer

This tool is intended for **research and educational purposes only**. It is **not a medical device** and must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified radiologist or physician for clinical decisions.