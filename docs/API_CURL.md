# API CURL (Imagem + Vídeo)

Base URL padrão (trocar depois pelo IP real):

`https://<my-vps-ip>`

Se o certificado HTTPS for self-signed, adicione `-k` no `curl`.

---

## 1) Geração de imagem com 3 imagens de referência

Endpoint:

`POST https://<my-vps-ip>/api/image/generate/multi-reference`

### Preparar base64 das 3 imagens

```bash
IMG1_B64=$(base64 -w 0 ref1.png)
IMG2_B64=$(base64 -w 0 ref2.png)
IMG3_B64=$(base64 -w 0 ref3.png)
```

### JSON da requisição

```json
{
  "prompt": "Professional headshot, studio lighting, realistic skin, 85mm lens",
  "reference_images": ["<BASE64_REF_1>", "<BASE64_REF_2>", "<BASE64_REF_3>"],
  "reference_strength": 0.8,
  "width": 1024,
  "height": 1024,
  "steps": 28
}
```

### CURL

```bash
curl -X POST "https://<my-vps-ip>/api/image/generate/multi-reference" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg prompt "Professional headshot, studio lighting, realistic skin, 85mm lens" \
    --arg img1 "$IMG1_B64" \
    --arg img2 "$IMG2_B64" \
    --arg img3 "$IMG3_B64" \
    '{
      prompt: $prompt,
      reference_images: [$img1, $img2, $img3],
      reference_strength: 0.8,
      width: 1024,
      height: 1024,
      steps: 28
    }')"
```

---

## 2) Fallback: geração de imagem com 1 imagem de referência

Endpoint:

`POST https://<my-vps-ip>/api/image/generate/multi-reference`

### Preparar base64 da imagem

```bash
IMG1_B64=$(base64 -w 0 ref1.png)
```

### JSON da requisição

```json
{
  "prompt": "Professional headshot, neutral background",
  "reference_images": ["<BASE64_REF_1>"],
  "reference_strength": 0.7,
  "width": 1024,
  "height": 1024,
  "steps": 28
}
```

### CURL

```bash
curl -X POST "https://<my-vps-ip>/api/image/generate/multi-reference" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg prompt "Professional headshot, neutral background" \
    --arg img1 "$IMG1_B64" \
    '{
      prompt: $prompt,
      reference_images: [$img1],
      reference_strength: 0.7,
      width: 1024,
      height: 1024,
      steps: 28
    }')"
```

---

## 3) Geração de vídeo com envio de imagem (image-to-video)

Endpoint:

`POST https://<my-vps-ip>/api/video/image-to-video`

### Preparar base64 da imagem

```bash
SOURCE_IMG_B64=$(base64 -w 0 source.png)
```

### JSON da requisição

```json
{
  "image": "<BASE64_SOURCE_IMAGE>",
  "prompt": "The person smiles and waves naturally",
  "duration": 4,
  "fps": 24,
  "motion_strength": 0.7
}
```

### CURL

```bash
curl -X POST "https://<my-vps-ip>/api/video/image-to-video" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg image "$SOURCE_IMG_B64" \
    --arg prompt "The person smiles and waves naturally" \
    '{
      image: $image,
      prompt: $prompt,
      duration: 4,
      fps: 24,
      motion_strength: 0.7
    }')"
```

---

## 4) Health checks rápidos

```bash
curl "https://<my-vps-ip>/health"
curl "https://<my-vps-ip>/api/image/health"
curl "https://<my-vps-ip>/api/video/health"
```
