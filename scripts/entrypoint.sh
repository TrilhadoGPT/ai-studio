#!/bin/bash
# ============================================
# AI-STUDIO Entrypoint Script
# ============================================

set -e

echo "🚀 Iniciando AI-STUDIO..."
echo "================================"

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/outputs/images
mkdir -p /app/outputs/videos
mkdir -p /app/uploads
mkdir -p /app/models/flux
mkdir -p /app/models/ltx

# Check GPU
echo "📊 Verificando GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Download models if not present
if [ ! -f "/app/models/flux/model_index.json" ] && [ -n "$HF_TOKEN" ]; then
    echo "⬇️ Baixando FLUX.2-dev..."
    huggingface-cli download black-forest-labs/FLUX.2-dev \
        --local-dir /app/models/flux \
        --token $HF_TOKEN
fi

if [ ! -f "/app/models/ltx/model_index.json" ] && [ -n "$HF_TOKEN" ]; then
    echo "⬇️ Baixando LTX-2..."
    huggingface-cli download Lightricks/LTX-2 \
        --local-dir /app/models/ltx \
        --token $HF_TOKEN
fi

# Set environment variables
export FLUX_MODEL_PATH=${FLUX_MODEL_PATH:-/app/models/flux}
export LTX_MODEL_PATH=${LTX_MODEL_PATH:-/app/models/ltx}
export FLUX_PRECISION=${FLUX_PRECISION:-fp8}
export LTX_PRECISION=${LTX_PRECISION:-fp8}

# Start services
echo "🔧 Iniciando serviços..."

# Start nginx
echo "  → Nginx (porta 80)"
nginx

# Start Image API
echo "  → Image API (porta 8000)"
cd /app && python api/image_api.py &
IMAGE_PID=$!

# Wait for Image API to start
sleep 10

# Start Video API
echo "  → Video API (porta 8001)"
cd /app && python api/video_api.py &
VIDEO_PID=$!

echo "================================"
echo "✅ AI-STUDIO iniciado!"
echo ""
echo "📡 Endpoints:"
echo "  • Health:    http://localhost/health"
echo "  • Imagem:    http://localhost/api/image/"
echo "  • Vídeo:     http://localhost/api/video/"
echo "  • Outputs:   http://localhost/outputs/"
echo ""
echo "🔧 PIDs:"
echo "  • Image API: $IMAGE_PID"
echo "  • Video API: $VIDEO_PID"
echo ""

# Keep container running and monitor processes
while true; do
    if ! kill -0 $IMAGE_PID 2>/dev/null; then
        echo "⚠️ Image API died, restarting..."
        cd /app && python api/image_api.py &
        IMAGE_PID=$!
    fi
    
    if ! kill -0 $VIDEO_PID 2>/dev/null; then
        echo "⚠️ Video API died, restarting..."
        cd /app && python api/video_api.py &
        VIDEO_PID=$!
    fi
    
    sleep 30
done
