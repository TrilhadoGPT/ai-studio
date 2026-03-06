#!/usr/bin/env bash
set -euo pipefail

SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  SUDO="sudo"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${ENV_FILE:-${ROOT_DIR}/.env}"

if [ ! -f "${ENV_PATH}" ] && [ -f "${ROOT_DIR}/.env.example" ]; then
  cp "${ROOT_DIR}/.env.example" "${ENV_PATH}"
fi

if [ -f "${ENV_PATH}" ]; then
  set -a
  . "${ENV_PATH}"
  set +a
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ python3 não encontrado."
  exit 1
fi

echo "📦 Instalando dependências do sistema..."
${SUDO} apt-get update
${SUDO} apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  ffmpeg \
  libsm6 \
  libxext6 \
  libgl1 \
  libglib2.0-0 \
  curl \
  git \
  wget

echo "🧹 Limpando cache do apt..."
${SUDO} apt-get clean
${SUDO} apt-get autoclean -y

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
if [ ! -d "${VENV_DIR}" ]; then
  echo "🐍 Criando virtualenv em ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

echo "📦 Instalando dependências Python..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install --no-cache-dir \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${VENV_DIR}/bin/pip" install --no-cache-dir -r "${ROOT_DIR}/requirements-host.txt"

echo "🧹 Limpando cache do pip..."
"${VENV_DIR}/bin/pip" cache purge || true

mkdir -p \
  "${ROOT_DIR}/logs" \
  "${ROOT_DIR}/uploads" \
  "${ROOT_DIR}/outputs/images" \
  "${ROOT_DIR}/outputs/videos" \
  "${ROOT_DIR}/models/flux" \
  "${ROOT_DIR}/models/ltx"

if [ -z "${HF_TOKEN:-}" ] && [ -r /dev/tty ]; then
  echo "🔑 Informe seu HF_TOKEN (input oculto):"
  read -r -s HF_TOKEN < /dev/tty
  echo
fi

if [ -n "${HF_TOKEN:-}" ]; then
  if grep -q '^HF_TOKEN=' "${ENV_PATH}" 2>/dev/null; then
    sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${HF_TOKEN}|" "${ENV_PATH}"
  else
    echo "HF_TOKEN=${HF_TOKEN}" >> "${ENV_PATH}"
  fi

  if [ ! -f "${ROOT_DIR}/models/flux/model_index.json" ]; then
    echo "⬇️ Baixando FLUX.2-dev..."
    "${VENV_DIR}/bin/huggingface-cli" download \
      black-forest-labs/FLUX.2-dev \
      --local-dir "${ROOT_DIR}/models/flux" \
      --token "${HF_TOKEN}"
  else
    echo "✅ FLUX já presente em ${ROOT_DIR}/models/flux"
  fi

  DOWNLOAD_LTX="${DOWNLOAD_LTX:-false}"
  if [ "${DOWNLOAD_LTX}" = "true" ] && [ ! -f "${ROOT_DIR}/models/ltx/model_index.json" ]; then
    echo "⬇️ Baixando LTX-2..."
    "${VENV_DIR}/bin/huggingface-cli" download \
      Lightricks/LTX-2 \
      --local-dir "${ROOT_DIR}/models/ltx" \
      --token "${HF_TOKEN}"
  fi
else
  echo "⚠️ HF_TOKEN não definido. O download automático de modelos foi ignorado."
fi

echo "✅ Instalação host concluída."
echo "Próximo passo: ENV_FILE=${ENV_PATH} ${ROOT_DIR}/scripts/start_host.sh"
