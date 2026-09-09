# Brain Tumor MRI Classifier

A Streamlit web app that classifies uploaded brain MRI images into glioma, meningioma, pituitary tumor, or no tumor. It also displays confidence scores and an optional Grad-CAM visualization.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app expects `brain_tumor_model.keras` (or the fallback `brain_tumor_model.h5`) and `class_names.json` in the project root.

## Deploy with Streamlit Community Cloud

[![Deploy to Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=bhargav25bai10209-blip/Brain_Tumor_Detector-&branch=main&mainModule=app.py)

Use the button above, or open this direct deployment link:

<https://share.streamlit.io/deploy?repository=bhargav25bai10209-blip/Brain_Tumor_Detector-&branch=main&mainModule=app.py>

1. Push this repository to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/) and choose **New app**.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy the app.

The `MRI_dataset/` directory is intentionally excluded from GitHub because it is training data, not required at runtime. The trained model files are included so the hosted app can make predictions.

## Important

This project is for educational and research use only. Predictions are not a medical diagnosis and must not replace review by a qualified healthcare professional.