# live_stream.py
import av
import os
import sys
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import tempfile

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro

st.title('AI Fitness Trainer: Squats Analysis (Live)')

# Mode selection
mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)

thresholds = get_thresholds_beginner() if mode == 'Beginner' else get_thresholds_pro()

# Session state for download
if 'download' not in st.session_state:
    st.session_state['download'] = False

output_video_file = "output_live.mp4"

# Initialize pose
pose = get_mediapipe_pose()

# -----------------------------
# OpenCV + VideoTransformer
# -----------------------------
class PoseTransformer(VideoTransformerBase):
    def __init__(self):
        self.process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
        self.pose = pose
        self.recording = False
        self.out = None

    def start_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video_file, fourcc, 20.0, (640, 480))
        self.recording = True

    def stop_recording(self):
        if self.out:
            self.out.release()
            self.out = None
        self.recording = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out_img, _ = self.process_frame.process(img, self.pose)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        # Record if active
        if self.recording and self.out:
            resized = cv2.resize(out_bgr, (640, 480))
            self.out.write(resized)

        return out_bgr


# -----------------------------
# WebRTC Streamer
# -----------------------------
transformer = PoseTransformer()

ctx = webrtc_streamer(
    key="live-squat-analysis",
    video_transformer_factory=PoseTransformer,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
    ),
    async_processing=False,
)

# -----------------------------
# Recording Controls
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Recording", disabled=ctx.video_transformer is None):
        if ctx.video_transformer:
            ctx.video_transformer.start_recording()
            st.success("Recording started...")

with col2:
    if st.button("Stop Recording", disabled=ctx.video_transformer is None):
        if ctx.video_transformer:
            ctx.video_transformer.stop_recording()
            st.success("Recording saved!")

# -----------------------------
# Download Button
# -----------------------------
download_button = st.empty()

if os.path.exists(output_video_file):
    with open(output_video_file, "rb") as f:
        download = download_button.download_button(
            label="Download Recorded Video",
            data=f,
            file_name="squat_analysis_live.mp4",
            mime="video/mp4"
        )
    if download:
        st.session_state['download'] = True

if st.session_state.get('download', False) and os.path.exists(output_video_file):
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()
    st.rerun()
