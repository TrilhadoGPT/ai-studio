#!/usr/bin/env python3
"""
API de Geração de Imagem - FLUX.2-dev
Projeto AI-STUDIO
"""

import os
import io
import base64
import uuid
import json
import inspect
from datetime import datetime
from typing import Optional, List
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
import torch

# Allow loading trusted FP8 checkpoints on PyTorch versions that default to
# weights_only=True and stricter safe globals.
try:
    torch.serialization.add_safe_globals([
        torch._C.StorageBase,
        torch.storage._LegacyStorage,
        torch.storage.TypedStorage,
        torch.UntypedStorage,
    ])
except Exception:
    pass

_original_torch_load = torch.load

def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load_compat

# ============================================
# CONFIGURATION
# ============================================
app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("FLUX_MODEL_PATH", "/app/models/flux/FLUX.2-dev")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs/images")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/uploads")
MAX_REF_IMAGES = 10

# ============================================
# MODEL LOADING
# ============================================
print("🔥 Carregando FLUX.2-dev...")
pipe = None

def load_model():
    global pipe
    if pipe is None:
        from diffusers import FluxPipeline, Flux2Pipeline
        
        # Detect precision from environment
        precision = os.environ.get("FLUX_PRECISION", "fp8")
        
        if precision == "fp8":
            torch_dtype = torch.float8_e4m3fn
        else:
            torch_dtype = torch.bfloat16
        
        pipeline_cls = FluxPipeline
        model_index_path = os.path.join(MODEL_PATH, "model_index.json")
        if os.path.exists(model_index_path):
            try:
                with open(model_index_path, "r", encoding="utf-8") as f:
                    model_index = json.load(f)
                if model_index.get("_class_name") == "Flux2Pipeline":
                    pipeline_cls = Flux2Pipeline
            except Exception:
                pass

        pipe = pipeline_cls.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
        )
        pipe.to("cuda")
        
        # Enable memory optimizations
        pipe.enable_model_cpu_offload() if os.environ.get("LOW_VRAM") else None
        
    return pipe

# ============================================
# HELPER FUNCTIONS
# ============================================
def save_image(image: Image.Image, prefix: str = "flux") -> str:
    """Save image and return path"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath, "PNG")
    return filepath

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))

def compose_reference_canvas(ref_images: List[Image.Image], width: int, height: int) -> Image.Image:
    """
    Combine multiple references into a single conditioning canvas.
    This gives the model a deterministic visual anchor when multi-reference
    adapters are not explicitly configured.
    """
    if len(ref_images) == 1:
        return ImageOps.fit(ref_images[0].convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    count = len(ref_images)
    cols = 2 if count > 1 else 1
    rows = (count + cols - 1) // cols
    cell_w = width // cols
    cell_h = height // rows

    for idx, ref in enumerate(ref_images):
        r = idx // cols
        c = idx % cols
        fitted = ImageOps.fit(ref.convert("RGB"), (cell_w, cell_h), method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, (c * cell_w, r * cell_h))

    return canvas

# ============================================
# API ENDPOINTS
# ============================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": "FLUX.2-dev",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_used": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A",
        "vram_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A"
    })

@app.route("/generate", methods=["POST"])
def generate_image():
    """
    Generate image from text prompt
    
    Request body:
    {
        "prompt": "A beautiful landscape",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "guidance_scale": 3.5,
        "seed": null,
        "num_images": 1
    }
    """
    try:
        data = request.json
        prompt = data.get("prompt", "")
        width = data.get("width", 1024)
        height = data.get("height", 1024)
        steps = data.get("steps", 20)
        guidance_scale = data.get("guidance_scale", 3.5)
        seed = data.get("seed")
        num_images = data.get("num_images", 1)
        
        # Load model
        pipeline = load_model()
        
        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate
        images = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images
        ).images
        
        # Convert to base64
        results = []
        for img in images:
            filepath = save_image(img)
            results.append({
                "image_base64": image_to_base64(img),
                "filepath": filepath,
                "width": img.width,
                "height": img.height
            })
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "images": results,
            "generation_time": "N/A"  # TODO: add timing
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/generate/multi-reference", methods=["POST"])
def generate_multi_reference():
    """
    Generate image with multiple reference images (for avatar consistency)
    
    Request body:
    {
        "prompt": "Person working in office",
        "reference_images": ["base64_img1", "base64_img2", ...],
        "reference_strength": 0.7,
        "width": 1024,
        "height": 1024,
        "steps": 28
    }
    """
    try:
        data = request.json
        prompt = data.get("prompt", "")
        ref_images_b64 = data.get("reference_images", [])
        ref_strength = data.get("reference_strength", 0.7)
        width = data.get("width", 1024)
        height = data.get("height", 1024)
        steps = data.get("steps", 28)
        
        # Validate reference images
        if len(ref_images_b64) > MAX_REF_IMAGES:
            return jsonify({
                "success": False, 
                "error": f"Maximum {MAX_REF_IMAGES} reference images allowed"
            }), 400
        
        # Process reference images
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        ref_images = []
        ref_pil_images = []
        for i, b64 in enumerate(ref_images_b64):
            img = base64_to_image(b64)
            filepath = os.path.join(UPLOAD_DIR, f"ref_{uuid.uuid4().hex[:8]}_{i}.png")
            img.save(filepath)
            ref_images.append(filepath)
            ref_pil_images.append(img)

        enhanced_prompt = f"{prompt} [reference strength: {ref_strength}]"
        ref_canvas = compose_reference_canvas(ref_pil_images, width, height)

        pipeline = load_model()
        call_params = inspect.signature(pipeline.__call__).parameters
        pipeline_kwargs = {
            "prompt": enhanced_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
        }

        # Use native image-conditioning when available for stronger identity carry-over.
        conditioning_mode = "prompt_only_fallback"
        if "image" in call_params:
            pipeline_kwargs["image"] = ref_canvas
            conditioning_mode = "image_conditioning"
        elif "ip_adapter_image" in call_params:
            pipeline_kwargs["ip_adapter_image"] = ref_pil_images
            conditioning_mode = "ip_adapter_image_conditioning"

        try:
            image = pipeline(**pipeline_kwargs).images[0]
        except Exception:
            # Ensure endpoint still works even if the runtime pipeline rejects
            # conditioning args for the loaded checkpoint.
            image = pipeline(
                prompt=enhanced_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
            ).images[0]
            conditioning_mode = "prompt_only_fallback"
        
        filepath = save_image(image, "multiref")
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "reference_count": len(ref_images),
            "conditioning_mode": conditioning_mode,
            "image": {
                "image_base64": image_to_base64(image),
                "filepath": filepath
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/image-to-image", methods=["POST"])
def image_to_image():
    """
    Transform existing image with new prompt
    
    Request body:
    {
        "image": "base64_image",
        "prompt": "Make it snowy",
        "strength": 0.75,
        "steps": 20
    }
    """
    try:
        data = request.json
        image_b64 = data.get("image")
        prompt = data.get("prompt", "")
        strength = data.get("strength", 0.75)
        steps = data.get("steps", 20)
        
        # Decode input image
        input_image = base64_to_image(image_b64)
        
        # TODO: Implement proper img2img with FLUX
        # For now, generate new image with prompt
        pipeline = load_model()
        
        output_image = pipeline(
            prompt=prompt,
            width=input_image.width,
            height=input_image.height,
            num_inference_steps=steps,
        ).images[0]
        
        filepath = save_image(output_image, "img2img")
        
        return jsonify({
            "success": True,
            "image": {
                "image_base64": image_to_base64(output_image),
                "filepath": filepath
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/avatar/generate", methods=["POST"])
def generate_avatar():
    """
    Generate avatar variations with consistent identity
    
    Request body:
    {
        "avatar_image": "base64_image",
        "prompt": "Person in business suit",
        "num_variations": 4,
        "consistency_strength": 0.8
    }
    """
    try:
        data = request.json
        avatar_b64 = data.get("avatar_image")
        prompt = data.get("prompt", "")
        num_variations = data.get("num_variations", 4)
        consistency = data.get("consistency_strength", 0.8)
        
        # Save avatar
        avatar_img = base64_to_image(avatar_b64)
        avatar_path = os.path.join(UPLOAD_DIR, f"avatar_{uuid.uuid4().hex[:8]}.png")
        avatar_img.save(avatar_path)
        
        pipeline = load_model()
        
        # Generate variations
        variations = []
        for i in range(num_variations):
            enhanced_prompt = f"{prompt} [identity consistency: {consistency}]"
            
            img = pipeline(
                prompt=enhanced_prompt,
                width=1024,
                height=1024,
                num_inference_steps=28,
                seed=torch.randint(0, 2**32, (1,)).item()
            ).images[0]
            
            filepath = save_image(img, f"avatar_var{i}")
            variations.append({
                "image_base64": image_to_base64(img),
                "filepath": filepath,
                "variation_index": i
            })
        
        return jsonify({
            "success": True,
            "avatar_prompt": prompt,
            "variations": variations
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Pre-load model
    print("Pré-carregando modelo...")
    load_model()
    print("✅ Modelo carregado!")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
