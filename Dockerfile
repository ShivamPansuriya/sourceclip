# syntax=docker/dockerfile:1
# ============================================================
# Multi-stage Dockerfile for Video Dubber
# Stage 1: Builder — install deps and compile
# Stage 2: Runtime — minimal image with only what's needed
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# FFmpeg runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY dub/ dub/
COPY pyproject.toml .

# Install the package itself
RUN pip install --no-deps -e .

# Create working directories
RUN mkdir -p /data/input /data/output /tmp/dub_work

# Default: run the CLI
ENTRYPOINT ["dub"]
CMD ["--help"]
