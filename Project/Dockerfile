# Multi-stage Dockerfile to build Python packages (including wheels) in a builder stage
# and produce a smaller runtime image with only runtime system libs + installed packages.
# Adjust apt package names if you use a different base image.

########################################
# Builder stage: install build deps and Python packages
########################################
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install build-time system dependencies required to compile wheels for av/aiortc, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    ca-certificates \
    ffmpeg \
    libssl-dev \
    libffi-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libsrtp2-dev \
    libopus-dev \
    libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install into /install prefix to copy into the final image
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --prefix=/install -r /app/requirements.txt

########################################
# Final stage: smaller runtime image
########################################
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install only runtime system packages (smaller set; no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    libsrtp2-1 \
    libopus0 \
    libvpx6 \
    libavcodec58 \
    libavformat58 \
    libavdevice58 \
    libavutil56 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY . /app

EXPOSE 8080

# Run streamlit on Cloud Run expected port/address
CMD ["streamlit", "run", "app/pages/1_📷️_Live_Stream.py", "--server.port=8080", "--server.address=0.0.0.0"]
