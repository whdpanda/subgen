FROM python:3.11-slim

# --- System deps ---
# ffmpeg: burn subtitles / audio extraction
# fontconfig + fonts-noto-cjk: make libass able to render Chinese
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    fontconfig \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# --- App setup ---
WORKDIR /app
COPY . /app

ARG INSTALL_EXTRAS=agent
ARG TORCH_VARIANT=cpu

RUN pip install --no-cache-dir -U pip && \
    if [ "$TORCH_VARIANT" = "cpu" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio ; \
    else \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} torch torchaudio ; \
    fi && \
    pip install --no-cache-dir demucs && \
    pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"

ENV SUBGEN_DATA_ROOT=/data
ENV SUBGEN_ALLOWED_ROOTS=/data
ENV SUBGEN_ALLOW_CREATE_OUTPUT_DIR=1
ENV SUBGEN_API_HOST=0.0.0.0
ENV SUBGEN_API_PORT=8000

EXPOSE 8000
CMD ["uvicorn", "subgen.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
