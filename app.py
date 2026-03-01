import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from core.processor import FaceFilterProcessor

st.set_page_config(page_title="Face Filter", page_icon="🧙", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Raleway:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Raleway', sans-serif; background-color: #0a0a0f; color: #e8e0d0; }
    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #1a0a2e 0%, #0a0a0f 60%),
                    radial-gradient(ellipse at 80% 20%, #0d1a2e 0%, transparent 50%);
        background-color: #0a0a0f;
    }
    .title-main {
        font-family: 'Cinzel', serif; font-size: 2.5rem; font-weight: 900;
        background: linear-gradient(135deg, #c9a84c 0%, #f5d78e 50%, #c9a84c 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; letter-spacing: 0.1em; text-transform: uppercase;
        text-align: center; margin: 1rem 0 0.25rem;
    }
    .title-sub {
        font-family: 'Raleway', sans-serif; font-size: 0.8rem; color: #7a6e5e;
        letter-spacing: 0.3em; text-transform: uppercase; text-align: center; margin-bottom: 1.5rem;
    }
    .divider { height: 1px; background: linear-gradient(90deg, transparent, #c9a84c44, transparent); margin: 1.5rem 0; }
    div[data-testid="stButton"] button {
        width: 100%; background: linear-gradient(135deg, #c9a84c, #f5d78e);
        color: #0a0a0f; font-family: 'Cinzel', serif; font-weight: 700;
        letter-spacing: 0.15em; border: none; border-radius: 8px;
        padding: 0.6rem 1rem; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-main">🧙 Face Filter</p>', unsafe_allow_html=True)
st.markdown('<p class="title-sub">Real-time AR Face Filter</p>', unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    def load_png(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            st.error(f"Could not load: {path}")
            st.stop()
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    def load_gif(path):
        pil_gif = Image.open(path)
        frames = []
        try:
            while True:
                frame = pil_gif.convert("RGBA")
                bgra = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
                frames.append(bgra)
                pil_gif.seek(pil_gif.tell() + 1)
        except EOFError:
            pass
        return frames

    return {
        "hat":      load_png("assets/wizard-hat.png"),
        "mustache": load_png("assets/churos-mustache.png"),
        "milky":    load_gif("assets/milky-pour.gif"),
    }


assets = load_assets()

FILTER_OPTIONS = {
    "🧙 Wizard Hat":      "hat",
    "🥸 Churos Mustache": "mustache",
    "🥛 Milky Pour":      "milky",
}

selected_label = st.radio(
    "",
    list(FILTER_OPTIONS.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
selected_mode = FILTER_OPTIONS[selected_label]

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="face-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=lambda: FaceFilterProcessor(assets),
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    async_processing=True,
    desired_playing_state=True,
)

if ctx.video_processor:
    ctx.video_processor.mode = selected_mode

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if ctx.video_processor:
    if st.button("📸 Take Picture"):
        snapshot = ctx.video_processor.get_snapshot()
        if snapshot is not None:
            rgb_snap = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_snap)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            st.image(pil_img, caption="Your snapshot", use_container_width=True)
            st.download_button(
                label="⬇ Download",
                data=buf,
                file_name=f"{selected_mode}_photo.png",
                mime="image/png",
            )
        else:
            st.warning("No frame captured yet. Make sure the camera is running.")