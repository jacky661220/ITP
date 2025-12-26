# ========= Stage 1: Build Python deps 到 /install =========
FROM python:3.11-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build

# 只先複製 requirements.txt，最大化快取命中
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt

# ========= Stage 2: Runtime 映像 =========
FROM python:3.11-slim-bookworm

# 基本環境
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 最小系統依賴；同層清理 apt 快取
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 複製已安裝的 site-packages（來自 builder）
COPY --from=builder /install /usr/local

# 安裝 Playwright + Chromium（含系統依賴），並在同一層清掉快取
# 注意：--with-deps 會自動補齊 Playwright 需要的系統套件
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir playwright \
 && python -m playwright install --with-deps chromium \
 && rm -rf /root/.cache/ms-playwright \
 && rm -rf /var/lib/apt/lists/*

# 複製專案（確保 .dockerignore 有排除不必要檔案）
COPY . /app

# Render 會注入 PORT；若本地執行則預設 8000
EXPOSE 8000
CMD ["sh","-c","uvicorn app_combined:app --host 0.0.0.0 --port ${PORT:-8000}"]