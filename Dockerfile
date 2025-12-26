FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /ITP

# 系統依賴（Playwright/Chromium 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
ca-certificates curl git wget gnupg \
libglib2.0-0 libnss3 libgdk-pixbuf-2.0-0 libgtk-3-0 libx11-6 \
libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libasound2 libatk1.0-0 \
libatk-bridge2.0-0 libgbm1 libxshmfence1 libxext6 libxcursor1 \
libxrender1 libxi6 libxtst6 libpangocairo-1.0-0 libpango-1.0-0 \
libatspi2.0-0 libdrm2 libxkbcommon0 libnspr4 \
fonts-liberation libappindicator3-1 libu2f-udev \
ffmpeg \
&& rm -rf /var/lib/apt/lists/*

# 先安裝 Python 依賴（利用快取）
COPY requirements.txt /ITP/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 安裝 Playwright 的瀏覽器（Chromium）
RUN python -m playwright install --with-deps chromium

# 複製整個專案（包含你的模型資料夾）
COPY . /ITP

EXPOSE 8000

# 建議可用 Railway 的 PORT 環境變數（沒有就用 8000）
CMD ["sh", "-c", "uvicorn app_combined:app --host 0.0.0.0 --port ${PORT:-8000}"]