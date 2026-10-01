FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2 ONNX_THREADS=2

WORKDIR /app

# Serving stack is torch-free: all neural models run through onnxruntime.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY models ./models
COPY mcp_server ./mcp_server

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
