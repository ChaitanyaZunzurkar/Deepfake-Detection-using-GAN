import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from inference import Detector

st.set_page_config(
    page_title="Deepfake Lens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#eaf0ff; --muted:#9ca9c4; --panel:#111827; --line:#26334b; --accent:#8b9cff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 20% 0%, #17233c 0%, #0b1020 42%, #080c16 100%); color:var(--ink); }
.block-container { max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
.hero { padding: 1.6rem 1.7rem; border:1px solid #293653; border-radius:22px; background:linear-gradient(120deg,rgba(30,43,72,.94),rgba(15,22,39,.85)); margin-bottom:1.3rem; }
.eyebrow { color:#a9b6ff; font-weight:700; text-transform:uppercase; font-size:.75rem; letter-spacing:.16em; }
.hero h1 { font-size:2.55rem; line-height:1.1; margin:.5rem 0 .65rem; }
.hero p { color:#b8c4dc; font-size:1.02rem; max-width:680px; margin:0; }
.panel { border:1px solid var(--line); border-radius:18px; background:rgba(17,24,39,.82); padding:1.2rem; }
.kicker { color:#9daecc; text-transform:uppercase; font-size:.72rem; font-weight:700; letter-spacing:.12em; }
.result { border:1px solid #34425f; border-radius:18px; padding:1.4rem; background:linear-gradient(135deg,#141e32,#101726); }
.result-title { font-family:'Space Grotesk'; font-size:2.2rem; font-weight:700; }
.small-muted { color:#9ca9c4; font-size:.9rem; }
div[data-testid="stFileUploader"] { border:1px dashed #465777; border-radius:14px; padding:.5rem; background:rgba(17,24,39,.6); }
.stButton>button { border-radius:10px; font-weight:700; min-height:2.8rem; }
section[data-testid="stSidebar"] { background:#0c1220; border-right:1px solid #26334b; }
hr { border-color:#26334b; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    try:
        return Detector(), None
    except Exception as exc:
        return None, str(exc)

detector, load_error = load_detector()

with st.sidebar:
    st.markdown("## Deepfake Lens")
    st.caption("Image & video authenticity analysis")
    st.divider()
    st.markdown("### Analysis settings")
    if detector:
        available = detector.available_models()
    else:
        available = []
    model_labels = {
        "M1_original": "M1 · Original image",
        "M2_gan_enhanced": "M2 · GAN-enhanced",
        "M3_feature_fusion": "M3 · Feature fusion",
    }
    if available:
        selected = st.selectbox("Detection model", available, format_func=lambda x: model_labels.get(x, x))
    else:
        selected = "M1_original"
        st.selectbox("Detection model", ["M1 · Original image", "M2 · GAN-enhanced", "M3 · Feature fusion"], disabled=True)
    import torch
    st.markdown(f"**Compute:** {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    st.markdown(f"**Model status:** {'Ready' if available else 'Waiting for checkpoints'}")
    st.divider()
    st.caption("Results are model estimates, not forensic proof.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI media analysis · Research tool</div>
  <h1>See beyond what looks real.</h1>
  <p>Upload an image or video to estimate whether its visual content may be AI-manipulated. Analyze single images or sample frames across a video.</p>
</div>
""", unsafe_allow_html=True)

tab_image, tab_video, tab_about = st.tabs(["▧  Image analysis", "▶  Video analysis", "ⓘ  About"])

with tab_image:
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("### Upload an image")
        img_file = st.file_uploader("Choose JPG, PNG or WEBP", type=["jpg","jpeg","png","webp","bmp"], key="image_upload")
        image = None
        if img_file:
            try:
                image = Image.open(img_file).convert("RGB")
                st.image(image, caption=img_file.name, use_container_width=True)
                st.caption(f"{image.width} × {image.height} pixels")
            except Exception as exc:
                st.error(f"Unable to read image: {exc}")
        else:
            st.markdown('<div class="panel"><div class="kicker">Image input</div><h3>Drop an image here</h3><p class="small-muted">The selected image will appear here before analysis.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Analysis result")
        if not available:
            st.markdown('<div class="result"><div class="kicker">Awaiting model files</div><div class="result-title">Ready when you are</div><p class="small-muted">The application interface is ready. Add your compatible checkpoints to the <code>models/</code> folder to enable predictions.</p></div>', unsafe_allow_html=True)
            if load_error:
                st.error(f"Model loading error: {load_error}")
            st.info("You can start and explore the app now. Prediction controls become active after the model files are available.")
        elif image is not None:
            if st.button("Analyze image", type="primary", use_container_width=True, key="analyze_image"):
                try:
                    with st.spinner("Analyzing image…"):
                        result = detector.predict_image(image, selected)
                    pred = result["prediction"]
                    st.markdown(f'<div class="result"><div class="kicker">Model prediction</div><div class="result-title">{pred}</div><p class="small-muted">Estimated confidence distribution from the selected classifier.</p></div>', unsafe_allow_html=True)
                    a,b = st.columns(2)
                    a.metric("Fake probability", f'{result["fake_probability"]:.1%}')
                    b.metric("Real probability", f'{result["real_probability"]:.1%}')
                    st.progress(float(result["fake_probability"]))
                    st.caption(result["note"])
                    st.download_button("Download result JSON", json.dumps(result, indent=2), "image_analysis.json", "application/json", use_container_width=True)
                except Exception as exc:
                    st.error(f"Analysis could not be completed: {exc}")
        else:
            st.markdown('<div class="result"><div class="kicker">No analysis yet</div><div class="result-title">Waiting for image</div><p class="small-muted">Upload an image on the left, then start the analysis.</p></div>', unsafe_allow_html=True)

with tab_video:
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("### Upload a video")
        video_file = st.file_uploader("Choose MP4, MOV, AVI or MKV", type=["mp4","mov","avi","mkv","webm"], key="video_upload")
        if video_file:
            st.video(video_file)
            st.caption(video_file.name)
        else:
            st.markdown('<div class="panel"><div class="kicker">Video input</div><h3>Drop a video here</h3><p class="small-muted">The app samples frames and estimates a video-level result.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Analysis result")
        if not available:
            st.markdown('<div class="result"><div class="kicker">Awaiting model files</div><div class="result-title">Video analysis ready</div><p class="small-muted">Add compatible checkpoints to enable frame analysis.</p></div>', unsafe_allow_html=True)
        elif video_file:
            max_frames = st.slider("Frames to sample", min_value=4, max_value=32, value=12, step=4)
            if st.button("Analyze video", type="primary", use_container_width=True, key="analyze_video"):
                temp_path = None
                try:
                    suffix = Path(video_file.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(video_file.getvalue())
                        temp_path = tmp.name
                    with st.spinner(f"Sampling up to {max_frames} frames and analyzing…"):
                        result = detector.predict_video(temp_path, selected, max_frames=max_frames)
                    st.markdown(f'<div class="result"><div class="kicker">Video-level estimate</div><div class="result-title">{result["prediction"]}</div><p class="small-muted">Mean fake probability across sampled frames; not a calibrated video probability.</p></div>', unsafe_allow_html=True)
                    st.metric("Mean sampled-frame fake probability", f'{result["mean_fake_probability"]:.1%}')
                    st.metric("Frames analyzed", result["frames_analyzed"])
                    st.dataframe(result["frame_results"], use_container_width=True, hide_index=True)
                    st.download_button("Download result JSON", json.dumps(result, indent=2), "video_analysis.json", "application/json", use_container_width=True)
                except Exception as exc:
                    st.error(f"Analysis could not be completed: {exc}")
                finally:
                    if temp_path:
                        Path(temp_path).unlink(missing_ok=True)
        else:
            st.markdown('<div class="result"><div class="kicker">No analysis yet</div><div class="result-title">Waiting for video</div><p class="small-muted">Upload a video on the left, then start the analysis.</p></div>', unsafe_allow_html=True)

with tab_about:
    st.markdown("### How it works")
    st.markdown("""
    1. **Input:** accepts a still image or a video file.
    2. **Preprocessing:** converts the image to RGB and prepares the frame for the trained model.
    3. **Detection:** runs the selected classifier; M2 and M3 also use the GAN enhancement checkpoint.
    4. **Video mode:** samples frames throughout the clip and aggregates frame-level fake probabilities.
    """)
    st.warning("This is a research prototype. Accuracy depends on the training data and preprocessing. Compression, lighting, unseen manipulation methods, and images without clear faces can affect results. Do not use this output as definitive evidence.")
    st.markdown("### Supported formats")
    st.write("Images: JPG, JPEG, PNG, WEBP, BMP · Videos: MP4, MOV, AVI, MKV, WEBM")

st.divider()
st.markdown('<p class="small-muted" style="text-align:center">DEEPFAKE LENS · LOCAL INFERENCE · RESEARCH PROTOTYPE</p>', unsafe_allow_html=True)
