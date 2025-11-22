import logging
import os

import av
import cv2
import numpy as np
import streamlit as st
# Import RTCConfiguration along with VideoProcessorBase and webrtc_streamer
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration 

# Basic logger for Cloud Run logs
logger = logging.getLogger("reflexa")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

st.header("Live Stream (Reflexa)")

# Try to import tflite runtime or fallback to tensorflow.lite if available
_tflite_interpreter = None
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
    _tflite_interpreter = tflite.Interpreter
    logger.info("Using tflite_runtime.interpreter")
except Exception:
    try:
        from tensorflow.lite import Interpreter as tflite  # type: ignore
        _tflite_interpreter = tflite
        logger.info("Using tensorflow.lite.Interpreter")
    except Exception:
        logger.info("No TFLite interpreter available; inference disabled")

# Optional: load a model if present at /app/model.tflite
TFLITE_MODEL_PATH = "/app/model.tflite"
interpreter = None
if _tflite_interpreter and os.path.exists(TFLITE_MODEL_PATH):
    try:
        interpreter = _tflite_interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        logger.info("Loaded TFLite model from %s", TFLITE_MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load TFLite model: %s", e)
        interpreter = None
else:
    if _tflite_interpreter:
        logger.info("TFLite interpreter available but no model found at %s", TFLITE_MODEL_PATH)

class VideoProcessor(VideoProcessorBase):
    """
    Example video processor that:
    - logs frame throughput
    - does simple processing (grayscale)
    - demonstrates where to run TFLite inference if available
    Implements recv(self, frame: av.VideoFrame) -> av.VideoFrame
    """
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Simple processing: grayscale -> back to BGR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Placeholder for TFLite inference block (customize for your model)
            if interpreter is not None:
                try:
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()

                    # Example assumptions — adapt to your model's input shape/format
                    inp_shape = input_details[0]["shape"]
                    # Typical shape: [1, h, w, c] or [1, h, w]
                    if len(inp_shape) >= 3:
                        inp_h, inp_w = inp_shape[1], inp_shape[2]
                        resized = cv2.resize(processed, (inp_w, inp_h))
                        # Normalize if needed (adjust per your model)
                        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
                        # Match dtype if required
                        if input_details[0]["dtype"].name == "uint8":
                            input_data = (input_data * 255).astype(np.uint8)

                        interpreter.set_tensor(input_details[0]["index"], input_data)
                        interpreter.invoke()
                        preds = interpreter.get_tensor(output_details[0]["index"])
                        # Overlay inference result placeholder
                        cv2.putText(processed, "TFLite OK", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception:
                    logger.exception("TFLite inference error; continuing without inference")

            # Overlay frame count to verify processing visually
            cv2.putText(processed, f"Frames: {self.frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Convert back to av.VideoFrame and preserve timing metadata
            new_frame = av.VideoFrame.from_ndarray(processed, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        except Exception as e:
            logger.exception("Error processing frame: %s", e)
            # Return original frame on error to keep the pipeline alive
            return frame

# --- CORRECTED RTC CONFIGURATION ---
# We are now using RTCConfiguration class for better type safety and including 
# multiple public STUN servers to increase the chance of successful connection.
rtc_configuration = RTCConfiguration(
    iceServers=[
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:3478"]},
        {"urls": ["stun:stun1.l.google.com:5349"]},
        {"urls": ["stun:stun2.l.google.com:5349"]},
        {"urls": ["stun:stun3.l.google.com:3478"]},
        {"urls": ["stun:stun3.l.google.com:5349"]},
        # Add a public Twilio STUN server as an alternative fallback
        {"urls": ["stun:global.stun.twilio.com:3478"]},
        # If necessary, a TURN server with credentials would go here:
        {"urls": ["turn:relay1.expressturn.com:3480"], "username": "000000002079128151", "credential": "JBKtuvPGM+lrWxz4vUCblI5wBrs="},
        #turn
        {
        urls: "stun:stun.relay.metered.ca:80",
      },
      {
        urls: "turn:asia.relay.metered.ca:80",
        username: "706e30892f13a766e8ca9d04",
        credential: "g+mbb7HpKhlJ5ukA",
      },
      {
        urls: "turn:asia.relay.metered.ca:80?transport=tcp",
        username: "706e30892f13a766e8ca9d04",
        credential: "g+mbb7HpKhlJ5ukA",
      },
      {
        urls: "turn:asia.relay.metered.ca:443",
        username: "706e30892f13a766e8ca9d04",
        credential: "g+mbb7HpKhlJ5ukA",
      },
      {
        urls: "turns:asia.relay.metered.ca:443?transport=tcp",
        username: "706e30892f13a766e8ca9d04",
        credential: "g+mbb7HpKhlJ5ukA",
      },
    ]
)
# -----------------------------------

# Request video only (no audio)
media_stream_constraints = {"video": True, "audio": False}

webrtc_streamer(
    key="reflexa-live",
    rtc_configuration=rtc_configuration,
    media_stream_constraints=media_stream_constraints,
    video_processor_factory=lambda: VideoProcessor(),
    async_processing=True,
)
