import os, tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image, UnidentifiedImageError
from inference import Detector

app = FastAPI(title="Deepfake Detection API", version="1.0.0")
detector = Detector()

@app.get("/health")
def health():
    return {"status": "ok", "device": str(__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")),
            "loaded_models": detector.available_models(), "gan_loaded": detector.gan is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("M1_original")):
    name = (file.filename or "").lower()
    data = await file.read()
    try:
        if name.endswith((".jpg",".jpeg",".png",".webp",".bmp")):
            from io import BytesIO
            img = Image.open(BytesIO(data)).convert("RGB")
            return detector.predict_image(img, model)
        if name.endswith((".mp4",".mov",".avi",".mkv",".webm")):
            suffix = Path(name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data); path = tmp.name
            try:
                return detector.predict_video(path, model)
            finally:
                os.unlink(path)
        raise HTTPException(status_code=415, detail="Upload JPG/PNG/WEBP image or MP4/MOV/AVI/MKV video.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
