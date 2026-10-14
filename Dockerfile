# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    VECTORDB_PATH=/data/vectordb \
    SMARTDOCS_API_URL=http://localhost:8000

# System deps: Tesseract (+ English/Persian packs) and Poppler for OCR, libmagic/GL for `unstructured`, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-fas \
        poppler-utils libmagic1 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch by default (much smaller image). For GPU:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install torch --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY .streamlit ./.streamlit

# Run as a non-root user; /data holds the Chroma index and the HuggingFace model cache
RUN useradd --uid 10001 --create-home app \
    && mkdir -p /data && chown -R app:app /data /app
USER app
VOLUME ["/data"]

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# API by default (ONE worker: job state and the session API key live in process memory).
# UI: docker run ... <image> streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
