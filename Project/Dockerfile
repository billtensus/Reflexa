# Full-featured Dockerfile for Cloud Run (x86_64) with broad multimedia & build deps.
FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

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
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libswresample-dev \
    libsrtp2-dev \
    libopus-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libxvidcore-dev \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /app/requirements.txt
    # CRITICAL CHANGE: Remove all build-only dependencies after Python packages are compiled and installed.
    # This significantly reduces the final image size and attack surface (best practice for production).
RUN apt-get purge -y --autoremove \
    build-essential cmake pkg-config git \
    libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libswscale-dev \
    libavfilter-dev libswresample-dev libsrtp2-dev libopus-dev libvpx-dev \
    libx264-dev libx265-dev libxvidcore-dev libssl-dev libffi-dev \
    libjpeg-dev zlib1g-dev libpng-dev \
    apt-get clean \
    rm -rf /var/lib/apt/lists/*

COPY . /app

EXPOSE 8080

# Use bash -c so runtime $PORT is respected; run your demo.py (adjust path if needed)
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true

ENTRYPOINT ["streamlit", "run", "Demo.py"]





