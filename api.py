import os
import tempfile
from pathlib import Path
from io import BytesIO

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image, UnidentifiedImageError

from inference import Detector


# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(
    title="Deepfake Detection API",
    version="1.0.0",
)

detector = Detector()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    """Return API status and loaded model information."""

    return {
        "status": "ok",
        "device": str(
            torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        ),
        "loaded_models": detector.available_models(),
        "gan_loaded": detector.gan is not None,
        "model_errors": detector.get_model_errors(),
        "class_mapping": {
            "0": "FAKE",
            "1": "REAL",
        },
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("M1_original"),
):
    """
    Predict whether an uploaded image or video is REAL or FAKE.

    Assumed class mapping:
        0 = FAKE
        1 = REAL

    Confirm this mapping against the training labels.
    """

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    try:
        # ----------------------------------------------------
        # Validate file extension
        # ----------------------------------------------------

        if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Unsupported file type. Upload JPG, JPEG, PNG, "
                    "WEBP, BMP, MP4, MOV, AVI, MKV, or WEBM."
                ),
            )

        # ----------------------------------------------------
        # Validate model selection
        # ----------------------------------------------------

        available_models = detector.available_models()

        if model not in available_models:
            errors = detector.get_model_errors()

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Requested model is unavailable.",
                    "requested_model": model,
                    "available_models": available_models,
                    "model_error": errors.get(model),
                },
            )

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        data = await file.read()

        if not data:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 100 MB upload limit.",
            )

        # ----------------------------------------------------
        # Image prediction
        # ----------------------------------------------------

        if suffix in IMAGE_EXTENSIONS:
            try:
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    image = image.convert("RGB")

            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file: {exc}",
                ) from exc

            try:
                result = detector.predict_image(
                    image,
                    model_name=model,
                )

            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Image prediction failed: {exc}",
                ) from exc

        # ----------------------------------------------------
        # Video prediction
        # ----------------------------------------------------

        else:
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=suffix,
                    delete=False,
                ) as temp_file:
                    temp_file.write(data)
                    temp_path = temp_file.name

                result = detector.predict_video(
                    video_path=temp_path,
                    model_name=model,
                )

            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Video prediction failed: {exc}",
                ) from exc

            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

        # ----------------------------------------------------
        # Validate prediction response
        # ----------------------------------------------------

        prediction = result.get("prediction")

        if prediction not in {"FAKE", "REAL"}:
            raise HTTPException(
                status_code=500,
                detail="Model returned an invalid prediction label.",
            )

        # Keep probability fields consistent across images/videos.
        fake_probability = result.get("fake_probability")
        real_probability = result.get("real_probability")

        if fake_probability is None or real_probability is None:
            raise HTTPException(
                status_code=500,
                detail="Model did not return both class probabilities.",
            )

        # Return a consistent response.
        return {
            **result,
            "model": model,
            "prediction": prediction,
            "fake_probability": float(fake_probability),
            "real_probability": float(real_probability),
        }

    finally:
        await file.close()