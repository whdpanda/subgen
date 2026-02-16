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

# --- Build args ---
# 1) INSTALL_EXTRAS controls project extras:
#    - agent       : LangChain agent runtime
#    - agent_rag   : agent + rag
#    - all         : agent + rag + dev
#
# 2) TORCH_VARIANT controls torch wheels channel:
#    - cpu   : CPU-only wheels (no GPU)
#    - cu121 : CUDA 12.1 wheels (GPU)
#    - cu118 : CUDA 11.8 wheels (GPU)
ARG INSTALL_EXTRAS=agent
ARG TORCH_VARIANT=cpu

# --- Python deps ---
# 1) Upgrade pip
# 2) Install PyTorch wheels based on TORCH_VARIANT
# 3) Install demucs
# 4) Install this project with extras
RUN pip install --no-cache-dir -U pip && \
    if [ "$TORCH_VARIANT" = "cpu" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio ; \
    else \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} torch torchaudio ; \
    fi && \
    pip install --no-cache-dir demucs && \
    pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"

# --- Runtime env defaults ---
ENV SUBGEN_DATA_ROOT=/data
ENV SUBGEN_ALLOWED_ROOTS=/data
ENV SUBGEN_ALLOW_CREATE_OUTPUT_DIR=1
ENV SUBGEN_API_HOST=0.0.0.0
ENV SUBGEN_API_PORT=8000

EXPOSE 8000

# --- Start server ---
CMD ["uvicorn", "subgen.api.main:app", "--host", "0.0.0.0", "--port", "8000"]