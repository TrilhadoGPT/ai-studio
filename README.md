# 🎨 AI-STUDIO - API Unificada Imagem + Vídeo

## 📋 Visão Geral

O **AI-STUDIO** é uma solução completa para geração de imagens e vídeos com IA, combinando:

- **FLUX.2-dev** para geração de imagens de alta qualidade
- **LTX-2 19B** para geração de vídeos com áudio sincronizado

Otimizado para **NVIDIA H200 NVL 140GB**, mas adaptável para outras GPUs.

## 🚀 Deploy Rápido (5 minutos)

### Estado Atual (Mar/2026)

- API de vídeo atualizada para **LTX2Pipeline** (text-to-video) e **LTX2ImageToVideoPipeline** (image-to-video)
- Suporte a fila assíncrona em memória para vídeo:
  - `POST /jobs/image-to-video`
  - `GET /jobs/<job_id>`
  - `GET /jobs/<job_id>/download`
- Modo host com opção **video-only** (`ENABLE_IMAGE=false`, `ENABLE_VIDEO=true`)

### Opção Host (sem Docker) - VPS limpa

```bash
cd /workspace
git clone https://github.com/TrilhadoGPT/ai-studio.git
cd ai-studio

cp .env.example .env
chmod +x scripts/install_host.sh scripts/start_host.sh scripts/stop_host.sh

# Instala dependências host
ENV_FILE=/workspace/ai-studio/.env ./scripts/install_host.sh

# Exemplo video-only no .env:
# ENABLE_IMAGE=false
# ENABLE_VIDEO=true
# DOWNLOAD_FLUX=false
# DOWNLOAD_LTX=true

# Sobe APIs habilitadas no .env
ENV_FILE=/workspace/ai-studio/.env ./scripts/start_host.sh

# Teste local vídeo
curl -s http://127.0.0.1:8001/health
```

> Em `huggingface_hub>=1.x`, o binário de CLI é `hf` (não `huggingface-cli`).

### Onde configurar tokens

- Caminho do arquivo `.env`: `/workspace/ai-studio/.env`
- Token do Hugging Face: `HF_TOKEN=...`
- Token do GitHub (para GHCR, opcional em modo host): `GHCR_TOKEN=...`

### Publicar no seu GitHub (novo repositório)

```bash
cd /workspace/ai-studio
git checkout -b feat/host-install
git add .
git commit -m "feat: host installer + start/stop scripts + flux2 reference conditioning"

# depois de criar repo vazio no seu GitHub:
git remote remove origin
git remote add origin https://github.com/SEU_USUARIO/SEU_REPO.git
git push -u origin feat/host-install
```

### Opção 0: Bootstrap em 1 comando (`curl | bash`)

```bash
curl -fsSL https://raw.githubusercontent.com/SEU_USER/SEU_REPO/main/install.sh -o /tmp/ai-studio-install.sh
cp .env.example .env
# edite .env e preencha: GHCR_USERNAME, GHCR_TOKEN, HF_TOKEN, AI_STUDIO_IMAGE
ENV_FILE=.env bash /tmp/ai-studio-install.sh
```

> Se preferir pipe direto: `curl -fsSL https://raw.githubusercontent.com/SEU_USER/SEU_REPO/main/install.sh | HF_TOKEN=hf_xxx GHCR_USERNAME=... GHCR_TOKEN=... AI_STUDIO_IMAGE=ghcr.io/... bash`

### Opção 1: Vast.ai (Recomendado)

```bash
# 1. Fazer login no Docker Hub
docker login

# 2. Build da imagem
docker build -t SEU_USUARIO/ai-studio:latest .

# 3. Push para Docker Hub
docker push SEU_USUARIO/ai-studio:latest

# 4. No Vast.ai:
#    - Create Instance
#    - Image: SEU_USUARIO/ai-studio:latest
#    - GPU: H200 NVL
#    - Ports: 80, 8000, 8001
#    - Environment: HF_TOKEN=seu_token
```

### Opção 2: Usar imagem pronta

```bash
# No Vast.ai:
vastai create instance \
  --image SEU_USUARIO/ai-studio:latest \
  --gpu "H200" \
  --port 80:80 \
  --port 8000:8000 \
  --port 8001:8001 \
  --env HF_TOKEN=seu_token_huggingface
```

## 📡 Endpoints

### API de Imagem (FLUX.2)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/image/health` | GET | Status da API |
| `/api/image/generate` | POST | Gerar imagem de texto |
| `/api/image/multi-reference` | POST | Gerar com referências |
| `/api/image/image-to-image` | POST | Transformar imagem |
| `/api/image/avatar/generate` | POST | Gerar variações de avatar |

### API de Vídeo (LTX-2)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/video/health` | GET | Status da API |
| `/api/video/generate` | POST | Gerar vídeo de texto |
| `/api/video/image-to-video` | POST | Animar imagem |
| `/api/video/avatar/animate` | POST | Animar avatar |
| `/api/video/batch` | POST | Gerar múltiplos vídeos |

### API de Vídeo (Host direto em `:8001`)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/health` | GET | Status da API de vídeo |
| `/generate` | POST | Text-to-video (sincrono) |
| `/image-to-video` | POST | Image-to-video (sincrono) |
| `/jobs/image-to-video` | POST | Cria job assíncrono |
| `/jobs/<job_id>` | GET | Consulta status do job |
| `/jobs/<job_id>/download` | GET | Baixa MP4 do job |

## 📝 Exemplos de Uso

### 1. Gerar Imagem

```bash
curl -X POST http://SEU_IP/api/image/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic lion in golden hour savanna",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "guidance_scale": 3.5
  }'
```

### 2. Gerar Avatar Consistente

```bash
curl -X POST http://SEU_IP/api/image/avatar/generate \
  -H "Content-Type: application/json" \
  -d '{
    "avatar_image": "BASE64_DA_IMAGEM",
    "prompt": "Person in business suit, professional headshot",
    "num_variations": 4,
    "consistency_strength": 0.8
  }'
```

### 3. Gerar Vídeo

```bash
curl -X POST http://SEU_IP/api/video/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing in a sunny garden",
    "width": 1280,
    "height": 720,
    "frames": 121,
    "fps": 24,
    "enable_audio": true
  }'
```

### 4. Animar Imagem

```bash
curl -X POST http://SEU_IP/api/video/image-to-video \
  -H "Content-Type: application/json" \
  -d '{
    "image": "BASE64_DA_IMAGEM",
    "prompt": "The person waves hello",
    "duration": 4,
    "fps": 24,
    "motion_strength": 0.7
  }'
```

### 5. Pipeline Completo (Avatar → Vídeo)

```python
import requests
import base64

BASE_URL = "http://SEU_IP"

# 1. Carregar avatar
with open("avatar.jpg", "rb") as f:
    avatar_b64 = base64.b64encode(f.read()).decode()

# 2. Gerar variações do avatar
avatar_response = requests.post(f"{BASE_URL}/api/image/avatar/generate", json={
    "avatar_image": avatar_b64,
    "prompt": "Person speaking in a podcast studio",
    "num_variations": 1,
    "consistency_strength": 0.85
})

variation = avatar_response.json()["variations"][0]["image_base64"]

# 3. Animar avatar
video_response = requests.post(f"{BASE_URL}/api/video/avatar/animate", json={
    "avatar_image": variation,
    "prompt": "Person speaking naturally",
    "speech_text": "Hello, welcome to my channel!",
    "duration": 5,
    "fps": 24
})

# 4. Salvar vídeo
video_b64 = video_response.json()["video"]["video_base64"]
with open("output.mp4", "wb") as f:
    f.write(base64.b64decode(video_b64))

print("✅ Vídeo gerado: output.mp4")
```

## ⚙️ Configuração

### Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `HF_TOKEN` | - | Token do Hugging Face (obrigatório) |
| `FLUX_PRECISION` | fp8 | Precisão: fp8, bf16, fp16 |
| `LTX_PRECISION` | fp8 | Precisão: fp8, bf16, fp16 |
| `FLUX_MODEL_PATH` | /app/models/flux | Caminho do modelo |
| `LTX_MODEL_PATH` | /app/models/ltx | Caminho do modelo |
| `LOW_VRAM` | false | Ativar otimizações de VRAM |

### Requisitos por GPU

| GPU | VRAM | Config Recomendada |
|-----|------|-------------------|
| RTX 4090/5090 | 24GB | FP8 para ambos |
| A100 | 80GB | BF16 completo |
| H200 NVL | 140GB | BF16 + Batch |

## 📊 Performance (H200 NVL)

### Imagem (FLUX.2)

| Configuração | Tempo | Throughput |
|--------------|-------|------------|
| 1024x1024, 20 steps | ~2-3s | ~20-30 img/s |
| Batch 8 | ~6-8s | ~1 img/s total |

### Vídeo (LTX-2)

| Configuração | Tempo | Throughput |
|--------------|-------|------------|
| 720p, 4s, 24fps | ~30-60s | ~60-120 vídeo/h |
| 4K, 10s, 50fps | ~2-3 min | ~20-30 vídeo/h |

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Reduzir precisão
export FLUX_PRECISION=fp8
export LTX_PRECISION=fp8

# Ativar otimizações
export LOW_VRAM=true
```

### Modelo não carrega

```bash
# Verificar HF_TOKEN
echo $HF_TOKEN

# Download manual
huggingface-cli login
huggingface-cli download black-forest-labs/FLUX.2-dev --local-dir /app/models/flux
huggingface-cli download Lightricks/LTX-2 --local-dir /app/models/ltx
```

## 📁 Estrutura de Arquivos

```
/app/
├── api/
│   ├── image_api.py      # API de imagem
│   └── video_api.py      # API de vídeo
├── config/
│   ├── nginx.conf        # Configuração do proxy
│   └── supervisord.conf  # Gestão de processos
├── models/
│   ├── flux/             # Modelos FLUX.2
│   └── ltx/              # Modelos LTX-2
├── outputs/
│   ├── images/           # Imagens geradas
│   └── videos/           # Vídeos gerados
├── scripts/
│   └── entrypoint.sh     # Script de inicialização
├── Dockerfile
└── README.md
```

## 📜 Licença

- **FLUX.2**: FLUX.2-dev License (não comercial)
- **LTX-2**: Open Source

---

**Projeto AI-STUDIO** - Guia completo disponível em:
- `Projeto_FLUX-H200.pdf` (Imagens)
- `Projeto_LTX-VIDEO.pdf` (Vídeos)
