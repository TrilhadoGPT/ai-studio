# ============================================
# PROJETO AI-STUDIO - Dockerfile Unificado
# FLUX.2-dev + LTX-2 para H200 NVL 140GB
# ============================================

FROM nvidia/cuda:12.4-devel-ubuntu22.04

LABEL maintainer="AI-STUDIO"
LABEL description="Unified API for FLUX.2 Image + LTX-2 Video Generation"
LABEL version="1.0"

# ============================================
# ENVIRONMENT SETUP
# ============================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# ============================================
# SYSTEM DEPENDENCIES
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================
# PYTHON ENVIRONMENT
# ============================================
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.5.0 \
    torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Core ML libraries
RUN pip install --no-cache-dir \
    transformers==4.47.0 \
    accelerate==1.2.1 \
    safetensors==0.4.5 \
    huggingface_hub==0.27.0 \
    diffusers==0.32.1 \
    sentencepiece \
    protobuf

# API and utilities
RUN pip install --no-cache-dir \
    flask==3.1.0 \
    flask-cors==5.0.0 \
    gunicorn==23.0.0 \
    pillow==11.0.0 \
    opencv-python-headless \
    numpy \
    pydantic \
    python-multipart \
    aiohttp \
    fastapi \
    uvicorn

# GGUF support for quantized models
RUN pip install --no-cache-dir \
    gguf \
    optimum

# ============================================
# DIRECTORY STRUCTURE
# ============================================
WORKDIR /app

# Create directories
RUN mkdir -p /app/models/flux \
    /app/models/ltx \
    /app/outputs/images \
    /app/outputs/videos \
    /app/uploads \
    /app/logs \
    /app/api \
    /app/workflows

# ============================================
# DOWNLOAD MODELS (Optional - can be done at runtime)
# ============================================
# Uncomment to pre-download models (increases image size significantly)
# ENV HF_TOKEN=your_token_here
# RUN huggingface-cli download black-forest-labs/FLUX.2-dev --local-dir /app/models/flux
# RUN huggingface-cli download Lightricks/LTX-2 --local-dir /app/models/ltx

# ============================================
# COPY APPLICATION FILES
# ============================================
COPY api/ /app/api/
COPY workflows/ /app/workflows/
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# ============================================
# NGINX CONFIGURATION
# ============================================
COPY config/nginx.conf /etc/nginx/nginx.conf
COPY config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ============================================
# PORTS
# ============================================
EXPOSE 80 8000 8001 8188

# ============================================
# HEALTH CHECK
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:8001/health || exit 1

# ============================================
# STARTUP SCRIPT
# ============================================
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
