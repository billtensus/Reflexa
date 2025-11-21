# Full-featured Dockerfile for Cloud Run (x86_64) with broad multimedia & build deps.
# This image intentionally includes the common dev/runtime libs that are often present
# on Raspberry Pi development images so your app has maximum compatibility when built
# in Cloud Build / Cloud Run. It is larger than a slim image but avoids apt package
# mismatch problems across Debian releases.
#
# Notes:
# - This uses python:3.11-bullseye (stable Debian base). If you prefer a different Debian
#   release, use the same tag consistently.
# - For TFLite: either install a matching tflite-runtime wheel for x86_64, or use
#   `tensorflow` (much larger) in requirements.txt.
FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Install system-level build & runtime dependencies commonly needed by:
# - python-av / ffmpeg bindings
# - aiortc / libsrtp / libopus
# - OpenCV build/runtime dependencies and common image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    ffmpeg \
    # libav* dev packages for building/compatibility
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libswresample-dev \
    # WebRTC / audio / codecs
    libsrtp2-dev \
    libopus-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libxvidcore-dev \
    # SSL / imaging
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps.
# Note: If you want to use tflite-runtime, uncomment it in requirements.txt and ensure
# a compatible manylinux/x86_64 wheel is available to pip. Otherwise use full tensorflow.
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Expose Cloud Run port
EXPOSE 8080

# Default start command (Streamlit). Change entrypoint if your project uses a different launcher.
CMD ["streamlit", "run", "app/pages/1_📷️_Live_Stream.py", "--server.port=8080", "--server.address=0.0.0.0"]
