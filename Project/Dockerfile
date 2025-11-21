# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# --- SYSTEM DEPENDENCIES ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    v4l-utils \
    pkg-config \
    # --- COMPILER + FFMPEG DEV (for fallback) ---
    build-essential \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# --- PYTHON DEPENDENCIES ---
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip

# Install av FIRST (ARM64 wheel, avoids Cython bug)
RUN pip install --no-cache-dir "av==12.3.0"

# Install rest (av now satisfied)
RUN pip install --no-cache-dir -r requirements.txt

# --- APP ---
COPY . /app

EXPOSE 8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true

ENTRYPOINT ["streamlit", "run", "Demo.py"]