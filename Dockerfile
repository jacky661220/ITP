FROM python:3.11-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
# 最小系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 複製已安裝的 site-packages
COPY --from=builder /install /usr/local

# 安裝 Playwright + Chromium，並在同一層清掉快取與不必要資產
RUN python -m pip install --no-cache-dir playwright \
 && python -m playwright install --with-deps chromium \
 && rm -rf /root/.cache/ms-playwright \
 && rm -rf /var/lib/apt/lists/*

# 複製專案（確保 .dockerignore 生效）
COPY . /app

EXPOSE 8000
CMD ["sh","-c","uvicorn app_combined:app --host 0.0.0.0 --port ${PORT:-8000}"]