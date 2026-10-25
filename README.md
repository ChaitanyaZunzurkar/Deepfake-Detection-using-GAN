# Deepfake Lens — Image & Video Detection App

A local Streamlit application for image and video deepfake analysis using your trained GAN-enhancement and classifier checkpoints.

## Start the application (Windows PowerShell)

```powershell
cd deepfake_detection_app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address, usually `http://localhost:8501`.

## Features
- Image upload and analysis (JPG/JPEG/PNG/WEBP/BMP)
- Video upload and analysis (MP4/MOV/AVI/MKV/WEBM)
- Choose among M1 original, M2 GAN-enhanced, and M3 feature-fusion classifiers
- Video frame sampling control and per-frame results
- Download prediction results as JSON
- GPU detection and clear model readiness status
- UI opens even if model checkpoints have not yet been added

## Add checkpoints when ready

Put compatible trained checkpoints in `models/` with these filenames:
- `M1_original_best.pth`
- `M2_gan_enhanced_best.pth`
- `M3_feature_fusion_best.pth`
- `enhancement_best.pth`

The model architecture and preprocessing in `model_definitions.py` and `inference.py` must match the training notebook. If a checkpoint fails to load, the app will show a diagnostic rather than silently inventing predictions.

## Optional API

In another terminal:

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Important limitation
Video output is an experimental aggregation of sampled-frame predictions, not a calibrated video-level probability. This research prototype is not forensic proof.
