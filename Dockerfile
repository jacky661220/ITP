# ---------- Builder 階段：安裝 Python 依賴 ----------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1
WORKDIR /build

COPY requirements.txt /build/requirements.txt
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt

# ---------- Runtime 階段：最小可執行環境 ----------
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 僅最小系統依賴；如需影像/音訊，保留 ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 將 builder 安裝好的 site-packages 複製進來
COPY --from=builder /install /usr/local

# 若需要 Playwright + Chromium，放在最後並清理快取
# 若不需要，整段移除可顯著變小
RUN python -m playwright install --with-deps chromium

# 複製專案（透過 .dockerignore 排除大型檔）
COPY . /app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app_combined:app --host 0.0.0.0 --port ${PORT:-8000}"]