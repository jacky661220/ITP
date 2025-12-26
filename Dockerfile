# 使用較穩定且小的基底
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 僅安裝極少量系統套件；其餘交給 Playwright --with-deps 處理
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 先裝 Python 依賴（利用快取）
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 安裝 Playwright 與 Chromium 及其系統依賴（一次完成）
# 若你的 requirements.txt 已含 playwright，這行只需安裝瀏覽器與依賴
RUN python -m playwright install --with-deps chromium

# 複製專案（請用 .dockerignore 避免大型檔案進來）
COPY . /app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app_combined:app --host 0.0.0.0 --port ${PORT:-8000}"]