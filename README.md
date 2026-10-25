# Deepfake Detection Application (GAN + 3 classifiers)

This starter application is aligned with the `deepfake_detection_custom_gan_256px.ipynb` notebook:
- M1: EfficientNet-B0 classifier on the original face crop
- M2: EfficientNet-B0 classifier on the GAN-enhanced face crop
- M3: shared EfficientNet-B0 feature-fusion classifier
- Optional GAN enhancement using `EnhancementGenerator` (256×256)
- Streamlit web UI and FastAPI inference API
- Image and video upload; video inference samples frames and aggregates frame probabilities

## Important status
The application code is provided, but it does not include trained weights. It becomes inference-ready after you train the notebook and copy the exported checkpoint files into `models/`. The current notebook is a frame-level research baseline, not a forensic guarantee.

## 1. Create environment (Windows PowerShell)
```powershell
cd deepfake_detection_app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PyTorch installation fails, use the official PyTorch install selector for your Windows/CUDA setup. CPU inference works but is slower.

## 2. Train in Kaggle
1. Open `deepfake_detection_custom_gan_256px.ipynb` and attach the FaceForensics++ C23 dataset plus the trained GAN checkpoint.
2. Enable GPU in Kaggle settings.
3. Run all cells through the end. The notebook trains M1/M2/M3 and saves checkpoints in `/kaggle/working/deepfake_gan_detector/models/`.
4. Download these files from Kaggle Output:
   - `M1_original_best.pth`
   - `M2_gan_enhanced_best.pth`
   - `M3_feature_fusion_best.pth`
   - `enhancement_best.pth` (the GAN checkpoint, from the GAN-training notebook/output)
   - `test_metrics.csv` and charts if you want them in the app.
5. Put the three classifier checkpoints and GAN checkpoint into this project's `models/` folder, retaining the exact filenames. If you don't want GAN enhancement at inference, M1 still works without the GAN checkpoint.

Expected layout:
```
deepfake_detection_app/
  app.py
  api.py
  model_definitions.py
  inference.py
  requirements.txt
  models/
    M1_original_best.pth
    M2_gan_enhanced_best.pth
    M3_feature_fusion_best.pth
    enhancement_best.pth
  outputs/
```

## 3. Launch web UI
```powershell
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

## 4. Launch API (separate terminal)
```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
API docs: `http://127.0.0.1:8000/docs`

## 5. API example
```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@sample.jpg" -F "model=M1_original"
```

## Notes
- Inference uses the notebook's RGB image conversion, resize to 224×224, and ImageNet normalization.
- Video prediction samples a limited number of frames and reports the mean frame probability. This is an experimental aggregation, not a calibrated video-level probability.
- A face detector is used when `facenet-pytorch` is installed; otherwise the entire image is resized. For more faithful comparison with training, install it and ensure training used the same crop method.
- Keep `models/` private if weights are not intended for distribution.
- Do not interpret predictions as legal or definitive evidence of manipulation.
