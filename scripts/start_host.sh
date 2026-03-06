#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${ENV_FILE:-${ROOT_DIR}/.env}"

if [ -f "${ENV_PATH}" ]; then
  set -a
  . "${ENV_PATH}"
  set +a
fi

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
if [ ! -x "${VENV_DIR}/bin/gunicorn" ]; then
  echo "❌ gunicorn não encontrado em ${VENV_DIR}. Rode scripts/install_host.sh primeiro."
  exit 1
fi

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/uploads" "${ROOT_DIR}/outputs/images" "${ROOT_DIR}/outputs/videos"

export FLUX_PRECISION="${FLUX_PRECISION:-bf16}"
export LTX_PRECISION="${LTX_PRECISION:-bf16}"
export FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-${ROOT_DIR}/models/flux}"
export LTX_MODEL_PATH="${LTX_MODEL_PATH:-${ROOT_DIR}/models/ltx}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/images}"
export VIDEO_OUTPUT_DIR="${VIDEO_OUTPUT_DIR:-${ROOT_DIR}/outputs/videos}"
export IMAGE_OUTPUT_DIR="${IMAGE_OUTPUT_DIR:-${ROOT_DIR}/outputs/images}"
export UPLOAD_DIR="${UPLOAD_DIR:-${ROOT_DIR}/uploads}"
export PYTHONPATH="${ROOT_DIR}"

pkill -f 'api.image_api:app' >/dev/null 2>&1 || true
pkill -f 'api.video_api:app' >/dev/null 2>&1 || true

echo "🚀 Iniciando Image API em 127.0.0.1:8000"
nohup "${VENV_DIR}/bin/gunicorn" \
  --bind 127.0.0.1:8000 \
  --workers "${IMAGE_WORKERS:-1}" \
  --timeout "${IMAGE_TIMEOUT:-600}" \
  --threads "${IMAGE_THREADS:-4}" \
  api.image_api:app > "${ROOT_DIR}/logs/image_api.log" 2>&1 &

ENABLE_VIDEO="${ENABLE_VIDEO:-false}"
if [ "${ENABLE_VIDEO}" = "true" ]; then
  echo "🚀 Iniciando Video API em 127.0.0.1:8001"
  nohup "${VENV_DIR}/bin/gunicorn" \
    --bind 127.0.0.1:8001 \
    --workers "${VIDEO_WORKERS:-1}" \
    --timeout "${VIDEO_TIMEOUT:-1800}" \
    --threads "${VIDEO_THREADS:-2}" \
    api.video_api:app > "${ROOT_DIR}/logs/video_api.log" 2>&1 &
fi

sleep 3
echo "🩺 Health local (imagem):"
curl -sS http://127.0.0.1:8000/health || true
echo

if [ "${ENABLE_VIDEO}" = "true" ]; then
  echo "🩺 Health local (vídeo):"
  curl -sS http://127.0.0.1:8001/health || true
  echo
fi

echo "✅ Serviços iniciados."
