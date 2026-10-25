import json, tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
from inference import Detector

st.set_page_config(page_title="Deepfake Detector", page_icon="🔎", layout="wide")
st.title("🔎 Deepfake Detection Lab")
st.caption("Custom GAN enhancement + three EfficientNet-B0 classifier variants")

@st.cache_resource
def load_detector():
    return Detector()

detector = load_detector()
with st.sidebar:
    st.header("Settings")
    available = detector.available_models()
    if not available:
        st.error("No classifier checkpoints found.")
        st.markdown("Train the Kaggle notebook, then copy its three `*_best.pth` files into the local `models/` folder.")
        st.stop()
    selected = st.selectbox("Classifier", available, format_func=lambda x: {
        "M1_original":"M1 — Original image",
        "M2_gan_enhanced":"M2 — GAN-enhanced image",
        "M3_feature_fusion":"M3 — Feature fusion"
    }.get(x,x))
    st.write("Device:", "GPU" if __import__("torch").cuda.is_available() else "CPU")
    st.info("The app uses the checkpoint(s) that exist in the models folder. M2/M3 also require the GAN checkpoint.")

kind = st.radio("Input type", ["Image", "Video"], horizontal=True)
uploaded = st.file_uploader("Upload an image or video", type=["jpg","jpeg","png","webp","mp4","mov","avi","mkv"])
if uploaded:
    if kind == "Image":
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1,1])
        with left: st.image(image, caption="Uploaded input", use_container_width=True)
        if st.button("Analyze image", type="primary"):
            try:
                result = detector.predict_image(image, selected)
                with right:
                    st.subheader(f"Prediction: {result['prediction']}")
                    st.metric("Fake probability", f"{result['fake_probability']:.1%}")
                    st.metric("Real probability", f"{result['real_probability']:.1%}")
                    st.progress(result["fake_probability"])
                    st.caption(result["note"])
                    st.json(result)
            except Exception as e:
                st.error(str(e))
    else:
        st.video(uploaded)
        if st.button("Analyze video", type="primary"):
            suffix = Path(uploaded.name).suffix or ".mp4"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue()); temp_path = tmp.name
                with st.spinner("Sampling frames and running inference..."):
                    result = detector.predict_video(temp_path, selected)
                st.subheader(f"Video-level estimate: {result['prediction']}")
                st.metric("Mean sampled-frame fake probability", f"{result['mean_fake_probability']:.1%}")
                st.write(f"Frames analyzed: {result['frames_analyzed']}")
                st.dataframe(result["frame_results"], use_container_width=True)
                st.download_button("Download result JSON", json.dumps(result, indent=2), "prediction.json", "application/json")
            except Exception as e:
                st.error(str(e))
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)

st.divider()

