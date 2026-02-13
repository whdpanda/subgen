# Dockerfile
FROM python:3.11-slim

# --- System deps (ffmpeg for burn/subtitles, plus basic build utils) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- App setup ---
WORKDIR /app

# Copy project first (simple approach; if you want better layer caching later, split requirements)
COPY . /app

# Upgrade pip and install project
RUN pip install --no-cache-dir -U pip && \
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
