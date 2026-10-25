import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import Detector


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="centered",
)


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


# ============================================================
# LOAD DETECTOR
# ============================================================

@st.cache_resource
def load_detector():
    return Detector()


# ============================================================
# HEADER
# ============================================================

st.title("Deepfake Detection")

try:
    detector = load_detector()

except Exception as exc:
    st.error(f"Failed to initialize the detector: {exc}")
    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

available_models = detector.available_models()

if not available_models:
    st.error(
        "No classification models were loaded. "
        "Check your model checkpoint files and architecture."
    )

    errors = detector.get_model_errors()

    if errors:
        with st.expander("Model loading errors", expanded=True):
            st.json(errors)

    st.stop()


# Display model selection.
default_model = (
    "M1_original"
    if "M1_original" in available_models
    else available_models[0]
)

model_options = available_models

selected_model = st.selectbox(
    "Select detection model",
    options=model_options,
    index=model_options.index(default_model),
    format_func=lambda model: {
        "M1_original": "Original Image Classifier",
        "M2_gan_enhanced": "GAN-Enhanced Image Classifier",
        "M3_feature_fusion": "Feature Fusion Classifier",
    }.get(model, model),
)


# Display GAN status.
if detector.gan is None:
    st.warning(
        "The GAN enhancement model is unavailable. "
        "GAN-enhanced and feature-fusion predictions may not work."
    )


# Show failed models, if any.
model_errors = detector.get_model_errors()

if model_errors:
    with st.expander("Model loading status"):
        st.write("Successfully loaded models:")
        st.write(available_models)

        st.write("Models with errors:")
        st.json(model_errors)


# ============================================================
# UPLOAD IMAGE OR VIDEO
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm",
    ],
    accept_multiple_files=False,
)


# ============================================================
# PREVIEW
# ============================================================

if uploaded_file is not None:

    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension not in SUPPORTED_EXTENSIONS:
        st.error("Unsupported file format.")
        st.stop()

    uploaded_data = uploaded_file.getvalue()

    if len(uploaded_data) == 0:
        st.error("The uploaded file is empty.")
        st.stop()

    if len(uploaded_data) > MAX_FILE_SIZE:
        st.error("File size exceeds the 100 MB limit.")
        st.stop()

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    if file_extension in IMAGE_EXTENSIONS:

        try:
            from io import BytesIO

            with Image.open(BytesIO(uploaded_data)) as img:
                img.verify()

            with Image.open(BytesIO(uploaded_data)) as img:
                preview_image = img.convert("RGB")

            st.image(
                preview_image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

        except (UnidentifiedImageError, OSError, ValueError):
            st.error("The uploaded file is not a valid image.")
            st.stop()

    # --------------------------------------------------------
    # VIDEO PREVIEW
    # --------------------------------------------------------

    elif file_extension in VIDEO_EXTENSIONS:

        st.video(uploaded_data)

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    ):

        temporary_video_path = None

        try:
            with st.spinner("Analyzing media..."):

                # ============================================
                # IMAGE PREDICTION
                # ============================================

                if file_extension in IMAGE_EXTENSIONS:

                    from io import BytesIO

                    with Image.open(
                        BytesIO(uploaded_data)
                    ) as img:
                        image = img.convert("RGB")

                    result = detector.predict_image(
                        image,
                        model_name=selected_model,
                    )

                # ============================================
                # VIDEO PREDICTION
                # ============================================

                elif file_extension in VIDEO_EXTENSIONS:

                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        suffix=file_extension,
                        delete=False,
                    ) as temp_file:

                        temp_file.write(uploaded_data)
                        temporary_video_path = temp_file.name

                    result = detector.predict_video(
                        video_path=temporary_video_path,
                        model_name=selected_model,
                    )

                else:
                    st.error("Unsupported file format.")
                    st.stop()

            # ================================================
            # VALIDATE PREDICTION RESULT
            # ================================================

            prediction = result.get("prediction")

            if prediction not in {"FAKE", "REAL"}:
                st.error(
                    "The model returned an invalid prediction. "
                    "Check the output handling in inference.py."
                )

                with st.expander("Prediction details"):
                    st.json(result)

                st.stop()

            fake_probability = result.get("fake_probability")
            real_probability = result.get("real_probability")

            # ================================================
            # DISPLAY RESULT
            # ================================================

            st.divider()
            st.subheader("Detection Result")

            if prediction == "FAKE":
                st.error("## FAKE")

            else:
                st.success("## REAL")

            st.caption(
                f"Model: {selected_model}"
            )

            # ================================================
            # DISPLAY PROBABILITIES
            # ================================================

            col1, col2 = st.columns(2)

            if fake_probability is not None:
                col1.metric(
                    "Fake Probability",
                    f"{float(fake_probability) * 100:.2f}%",
                )

            if real_probability is not None:
                col2.metric(
                    "Real Probability",
                    f"{float(real_probability) * 100:.2f}%",
                )

            # ================================================
            # DISPLAY VIDEO DETAILS
            # ================================================

            if "frames_analyzed" in result:
                st.write(
                    f"**Frames analyzed:** "
                    f"{result['frames_analyzed']}"
                )

            if "frame_results" in result:

                with st.expander("Frame analysis details"):

                    for frame_result in result["frame_results"]:

                        frame_prediction = frame_result.get(
                            "prediction",
                            "UNKNOWN",
                        )

                        frame_fake_probability = frame_result.get(
                            "fake_probability"
                        )

                        if frame_fake_probability is not None:
                            probability_text = (
                                f"{float(frame_fake_probability) * 100:.2f}%"
                            )
                        else:
                            probability_text = "Unavailable"

                        st.write(
                            f"**Frame {frame_result.get('frame', '?')}:** "
                            f"{frame_prediction} — "
                            f"Fake probability: {probability_text}"
                        )

            # ================================================
            # ADDITIONAL INFORMATION
            # ================================================

            if "note" in result:
                st.caption(result["note"])

            st.caption(
                "Predictions are model estimates, not a guarantee "
                "of whether media is authentic."
            )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

        finally:
            if (
                temporary_video_path
                and os.path.exists(temporary_video_path)
            ):
                try:
                    os.remove(temporary_video_path)
                except OSError:
                    pass