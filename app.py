import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from core.processor import FaceFilterProcessor

st.set_page_config(
    page_title="Face Filter",
    page_icon="🧙",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Raleway:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Raleway', sans-serif;
        background-color: #0a0a0f;
        color: #e8e0d0;
    }

    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #1a0a2e 0%, #0a0a0f 60%),
                    radial-gradient(ellipse at 80% 20%, #0d1a2e 0%, transparent 50%);
        background-color: #0a0a0f;
    }

    h1, h2, h3 {
        font-family: 'Cinzel', serif;
        letter-spacing: 0.05em;
    }

    .title-container {
        text-align: center;
        padding: 2rem 0 1rem;
    }

    .title-main {
        font-family: 'Cinzel', serif;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #c9a84c 0%, #f5d78e 50%, #c9a84c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0;
    }

    .title-sub {
        font-family: 'Raleway', sans-serif;
        font-size: 0.9rem;
        color: #7a6e5e;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #c9a84c44, transparent);
        margin: 1.5rem 0;
    }

    .filter-card {
        background: linear-gradient(135deg, #12101e 0%, #1a1528 100%);
        border: 1px solid #2a2240;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .filter-label {
        font-family: 'Cinzel', serif;
        font-size: 0.75rem;
        color: #c9a84c;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stRadio"] > label {
        font-family: 'Cinzel', serif;
        color: #c9a84c;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
    }

    div[data-testid="stRadio"] div[role="radio"] {
        background: #1a1528;
        border: 1px solid #2a2240;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #c9a84c, #f5d78e);
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'Raleway', sans-serif;
        letter-spacing: 0.1em;
        font-weight: 500;
    }

    .status-active {
        background: #1a2e1a;
        color: #5fd45f;
        border: 1px solid #2a4a2a;
    }

    .status-inactive {
        background: #2e1a1a;
        color: #d45f5f;
        border: 1px solid #4a2a2a;
    }

    .upload-hint {
        font-size: 0.78rem;
        color: #5a5060;
        font-style: italic;
        margin-top: 0.3rem;
    }

    section[data-testid="stSidebar"] {
        background: #08080d;
        border-right: 1px solid #1a1528;
    }

    .stFileUploader {
        background: #12101e;
        border: 1px dashed #2a2240;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <p class="title-main">🧙 Face Filter</p>
    <p class="title-sub">Mediapipe · Real-time AR · FaceMesh</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

def load_asset(uploaded_file):
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

col_feed, col_controls = st.columns([3, 1])

with col_controls:
    st.markdown('<div class="filter-label">Filter Mode</div>', unsafe_allow_html=True)
    filter_mode = st.radio(
        "",
        ["None", "Hat", "Sunglasses", "Both"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="filter-label">Hat Asset (PNG/WEBP)</div>', unsafe_allow_html=True)
    hat_file = st.file_uploader("", type=["png", "webp"], key="hat", label_visibility="collapsed")
    if hat_file:
        preview = Image.open(hat_file).convert("RGBA").resize((120, 80))
        st.image(preview)
        hat_file.seek(0)
    else:
        st.markdown('<p class="upload-hint">Upload a transparent PNG/WEBP</p>', unsafe_allow_html=True)

    st.markdown('<div class="filter-label" style="margin-top:1rem">Hat Scale</div>', unsafe_allow_html=True)
    hat_scale = st.slider("", 0.8, 2.0, 1.3, 0.05, key="hat_scale", label_visibility="collapsed")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="filter-label">Sunglasses Asset (PNG/WEBP)</div>', unsafe_allow_html=True)
    glasses_file = st.file_uploader("", type=["png", "webp"], key="glasses", label_visibility="collapsed")
    if glasses_file:
        preview = Image.open(glasses_file).convert("RGBA").resize((120, 60))
        st.image(preview)
        glasses_file.seek(0)
    else:
        st.markdown('<p class="upload-hint">Upload a transparent PNG/WEBP</p>', unsafe_allow_html=True)

    st.markdown('<div class="filter-label" style="margin-top:1rem">Glasses Scale</div>', unsafe_allow_html=True)
    glasses_scale = st.slider("", 0.8, 2.0, 1.1, 0.05, key="glasses_scale", label_visibility="collapsed")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="filter-label">Mesh Overlay</div>', unsafe_allow_html=True)
    show_mesh = st.toggle("Show FaceMesh", value=False)

with col_feed:
    ctx = webrtc_streamer(
        key="face-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=FaceFilterProcessor,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.active_filter = filter_mode.lower()
        ctx.video_processor.show_mesh = show_mesh
        ctx.video_processor.hat_scale = hat_scale
        ctx.video_processor.glasses_scale = glasses_scale

        if hat_file:
            hat_asset = load_asset(hat_file)
            if hat_asset is not None:
                ctx.video_processor.hat_asset = hat_asset

        if glasses_file:
            glasses_asset = load_asset(glasses_file)
            if glasses_asset is not None:
                ctx.video_processor.glasses_asset = glasses_asset

        is_active = ctx.state.playing
        badge_class = "status-active" if is_active else "status-inactive"
        badge_text = "● LIVE" if is_active else "○ STOPPED"
        st.markdown(f'<span class="status-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: #12101e;
            border: 1px solid #2a2240;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <p style="font-family: 'Cinzel', serif; color: #3a3050; font-size: 1rem;">
                Click START to activate camera
            </p>
        </div>
        """, unsafe_allow_html=True)
