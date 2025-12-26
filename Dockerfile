# ---- Build stage ----
FROM python:3.11-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build

# Copy only requirements first to leverage cache
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Python packages from builder
COPY --from=builder /install /usr/local

# Playwright + Chromium, then clean caches
RUN python -m pip install --no-cache-dir playwright \
 && python -m playwright install --with-deps chromium \
 && rm -rf /root/.cache/ms-playwright /var/lib/apt/lists/*

# Project files (make sure .dockerignore is set properly)
COPY . /app

EXPOSE 8000
CMD ["sh","-c","uvicorn app_combined:app --host 0.0.0.0 --port ${PORT:-8000}"]