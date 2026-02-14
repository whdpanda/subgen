# Dockerfile
FROM python:3.11-slim

# --- System deps ---
# ffmpeg: burn subtitles / audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- App setup ---
WORKDIR /app
COPY . /app

# --- Python deps ---
# 1) Upgrade pip
# 2) Install PyTorch CPU wheels explicitly (prevents pulling CUDA builds)
# 3) Install demucs
# 4) Install this project
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch torchaudio && \
    pip install --no-cache-dir demucs && \
    pip install --no-cache-dir .

# --- Runtime env defaults ---
ENV SUBGEN_DATA_ROOT=/data
ENV SUBGEN_ALLOWED_ROOTS=/data
ENV SUBGEN_ALLOW_CREATE_OUTPUT_DIR=1
ENV SUBGEN_API_HOST=0.0.0.0
ENV SUBGEN_API_PORT=8000

EXPOSE 8000

# --- Start server ---
CMD ["uvicorn", "subgen.api.main:app", "--host", "0.0.0.0", "--port", "8000"]