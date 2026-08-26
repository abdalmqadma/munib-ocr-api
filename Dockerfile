FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAX_UPLOAD_BYTES=4194304
ENV OCR_TIMEOUT_SECONDS=20
ENV RATE_LIMIT_REQUESTS=6
ENV RATE_LIMIT_WINDOW_SECONDS=60

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY secure_app.py .
COPY fast_ocr.py .

EXPOSE 8000

CMD ["uvicorn", "secure_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "10"]
