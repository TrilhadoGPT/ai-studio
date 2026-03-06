#!/usr/bin/env bash
set -euo pipefail

SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  SUDO="sudo"
fi

install_docker_if_possible() {
  if command -v docker >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    return 1
  fi

  echo "📦 Docker não encontrado. Tentando instalar via apt..."
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y docker.io

  if command -v systemctl >/dev/null 2>&1; then
    ${SUDO} systemctl enable docker >/dev/null 2>&1 || true
    ${SUDO} systemctl start docker >/dev/null 2>&1 || true
  else
    ${SUDO} service docker start >/dev/null 2>&1 || true
  fi

  command -v docker >/dev/null 2>&1
}

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

if [ -n "${ENV_FILE:-}" ] && [ -f "${ENV_FILE}" ]; then
  set -a
  . "${ENV_FILE}"
  set +a
fi

IMAGE="${AI_STUDIO_IMAGE:-ghcr.io/SEU_USER/ai-studio:latest}"
CONTAINER_NAME="${AI_STUDIO_CONTAINER_NAME:-ai-studio}"
HOST_PORT_HTTP="${HOST_PORT_HTTP:-80}"
HOST_PORT_IMAGE="${HOST_PORT_IMAGE:-8000}"
HOST_PORT_VIDEO="${HOST_PORT_VIDEO:-8001}"
DATA_ROOT="${AI_STUDIO_DATA_ROOT:-/opt/ai-studio}"
MODELS_DIR="${MODELS_DIR:-${DATA_ROOT}/models}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${DATA_ROOT}/outputs}"
UPLOADS_DIR="${UPLOADS_DIR:-${DATA_ROOT}/uploads}"
LOGS_DIR="${LOGS_DIR:-${DATA_ROOT}/logs}"

if ! command -v docker >/dev/null 2>&1; then
  if ! install_docker_if_possible; then
    echo "❌ Docker não encontrado e não foi possível instalar automaticamente."
    echo "   Em algumas VPS (container já isolado), Docker-in-Docker não é suportado."
    echo "   Nesse caso, suba a instância Vast já usando a imagem final: ${IMAGE}"
    exit 1
  fi
fi

if ! docker info >/dev/null 2>&1; then
  echo "❌ Docker daemon não está ativo."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠️ nvidia-smi não encontrado. Confirme que a VPS tem GPU NVIDIA."
fi

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -r /dev/tty ]; then
    echo "🔑 HF_TOKEN não definido."
    read -r -s -p "Digite seu HF_TOKEN (input oculto): " HF_TOKEN < /dev/tty
    echo
    export HF_TOKEN
  fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "⚠️ HF_TOKEN não definido."
  echo "   Sem HF_TOKEN, o download automático dos modelos pode falhar."
fi

if [ -n "${GHCR_USERNAME:-}" ] && [ -n "${GHCR_TOKEN:-}" ]; then
  echo "🔐 Login no GHCR..."
  echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USERNAME}" --password-stdin
fi

echo "📦 Pull da imagem: ${IMAGE}"
docker pull "${IMAGE}"

echo "📁 Criando diretórios persistentes em ${DATA_ROOT}"
${SUDO} mkdir -p "${MODELS_DIR}/flux" "${MODELS_DIR}/ltx" "${OUTPUTS_DIR}/images" "${OUTPUTS_DIR}/videos" "${UPLOADS_DIR}" "${LOGS_DIR}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  echo "♻️ Removendo container existente: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}"
fi

echo "🚀 Subindo container ${CONTAINER_NAME}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --gpus all \
  -p "${HOST_PORT_HTTP}:80" \
  -p "${HOST_PORT_IMAGE}:8000" \
  -p "${HOST_PORT_VIDEO}:8001" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e FLUX_PRECISION="${FLUX_PRECISION:-fp8}" \
  -e LTX_PRECISION="${LTX_PRECISION:-fp8}" \
  -e LOW_VRAM="${LOW_VRAM:-false}" \
  -v "${MODELS_DIR}:/app/models" \
  -v "${OUTPUTS_DIR}:/app/outputs" \
  -v "${UPLOADS_DIR}:/app/uploads" \
  -v "${LOGS_DIR}:/app/logs" \
  "${IMAGE}"

echo "✅ Deploy concluído."
echo "🌐 Health: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${HOST_PORT_HTTP}/health"
