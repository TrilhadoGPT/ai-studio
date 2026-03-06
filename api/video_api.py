#!/usr/bin/env python3
"""
API de Geração de Vídeo - LTX-2
Projeto AI-STUDIO
"""

import os
import io
import base64
import uuid
import json
import subprocess
from datetime import datetime
from typing import Optional, List
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("LTX_MODEL_PATH", "/app/models/ltx")
OUTPUT_DIR = os.environ.get("VIDEO_OUTPUT_DIR", "/app/outputs/videos")
IMAGE_DIR = os.environ.get("IMAGE_OUTPUT_DIR", "/app/outputs/images")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/uploads")

# ============================================
# MODEL LOADING
# ============================================
print("🎬 Carregando LTX-2...")
video_pipe = None
audio_pipe = None

def load_video_model():
    global video_pipe
    if video_pipe is None:
        from diffusers import LTXPipeline
        
        precision = os.environ.get("LTX_PRECISION", "fp8")
        
        if precision == "fp8":
            torch_dtype = torch.float8_e4m3fn
        elif precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
        
        video_pipe = LTXPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
        )
        video_pipe.to("cuda")
        
    return video_pipe

# ============================================
# HELPER FUNCTIONS
# ============================================
def save_video(frames: List, prefix: str = "ltx", fps: int = 24) -> str:
    """Save frames as video file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save frames as temporary images
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame)
        else:
            img = frame
        img.save(os.path.join(temp_dir, f"frame_{i:05d}.png"))
    
    # Use ffmpeg to create video
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        filepath
    ], check=True, capture_output=True)
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    return filepath

def video_to_base64(filepath: str) -> str:
    """Convert video file to base64"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))

# ============================================
# API ENDPOINTS
# ============================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": "LTX-2 19B",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_used": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A",
        "vram_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A"
    })

@app.route("/generate", methods=["POST"])
def generate_video():
    """
    Generate video from text prompt
    
    Request body:
    {
        "prompt": "A cat playing in the garden",
        "width": 1280,
        "height": 720,
        "frames": 121,
        "fps": 24,
        "steps": 28,
        "guidance_scale": 3.0,
        "seed": null,
        "enable_audio": true
    }
    """
    try:
        data = request.json
        prompt = data.get("prompt", "")
        width = data.get("width", 1280)
        height = data.get("height", 720)
        num_frames = data.get("frames", 121)
        fps = data.get("fps", 24)
        steps = data.get("steps", 28)
        guidance_scale = data.get("guidance_scale", 3.0)
        seed = data.get("seed")
        enable_audio = data.get("enable_audio", False)
        
        # Load model
        pipeline = load_video_model()
        
        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate video
        output = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        frames = output.frames[0]
        
        # Save video
        video_path = save_video(frames, fps=fps)
        
        # TODO: Add audio if enabled
        audio_path = None
        if enable_audio:
            # audio_path = generate_audio(video_path, prompt)
            pass
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "video": {
                "filepath": video_path,
                "video_base64": video_to_base64(video_path),
                "width": width,
                "height": height,
                "frames": num_frames,
                "fps": fps,
                "duration": num_frames / fps
            },
            "audio": audio_path
        })
        
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/image-to-video", methods=["POST"])
def image_to_video():
    """
    Animate an existing image
    
    Request body:
    {
        "image": "base64_image",
        "prompt": "Person walking in the park",
        "duration": 4,
        "fps": 24,
        "motion_strength": 0.7
    }
    """
    try:
        data = request.json
        image_b64 = data.get("image")
        prompt = data.get("prompt", "")
        duration = data.get("duration", 4)  # seconds
        fps = data.get("fps", 24)
        motion_strength = data.get("motion_strength", 0.7)
        
        # Decode input image
        input_image = base64_to_image(image_b64)
        
        # Save uploaded image
        upload_path = os.path.join(UPLOAD_DIR, f"img2vid_{uuid.uuid4().hex[:8]}.png")
        input_image.save(upload_path)
        
        # Load model
        pipeline = load_video_model()
        
        num_frames = duration * fps
        
        # Generate video from image
        output = pipeline(
            prompt=prompt,
            image=input_image,
            width=input_image.width,
            height=input_image.height,
            num_frames=num_frames,
            num_inference_steps=28,
            strength=motion_strength,
        )
        
        frames = output.frames[0]
        video_path = save_video(frames, prefix="img2vid", fps=fps)
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "source_image": upload_path,
            "video": {
                "filepath": video_path,
                "video_base64": video_to_base64(video_path),
                "duration": duration,
                "fps": fps,
                "frames": num_frames
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/avatar/animate", methods=["POST"])
def animate_avatar():
    """
    Animate an avatar image into video
    
    Request body:
    {
        "avatar_image": "base64_image",
        "prompt": "Person speaking naturally",
        "speech_text": "Hello, welcome to my channel!",
        "duration": 5,
        "fps": 24
    }
    """
    try:
        data = request.json
        avatar_b64 = data.get("avatar_image")
        prompt = data.get("prompt", "Person speaking naturally")
        speech_text = data.get("speech_text", "")
        duration = data.get("duration", 5)
        fps = data.get("fps", 24)
        
        # Decode avatar
        avatar_img = base64_to_image(avatar_b64)
        
        # Create enhanced prompt for animation
        if speech_text:
            enhanced_prompt = f"{prompt}. The person is saying: '{speech_text}'"
        else:
            enhanced_prompt = prompt
        
        # Load model
        pipeline = load_video_model()
        
        num_frames = duration * fps
        
        # Generate animated video
        output = pipeline(
            prompt=enhanced_prompt,
            image=avatar_img,
            width=min(avatar_img.width, 1280),
            height=min(avatar_img.height, 720),
            num_frames=num_frames,
            num_inference_steps=30,
        )
        
        frames = output.frames[0]
        video_path = save_video(frames, prefix="avatar_animate", fps=fps)
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "speech_text": speech_text,
            "video": {
                "filepath": video_path,
                "video_base64": video_to_base64(video_path),
                "duration": duration,
                "fps": fps
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/batch", methods=["POST"])
def batch_generate():
    """
    Generate multiple videos in batch
    
    Request body:
    {
        "prompts": ["A cat playing", "A dog running", ...],
        "config": {
            "width": 720,
            "height": 720,
            "duration": 4
        }
    }
    """
    try:
        data = request.json
        prompts = data.get("prompts", [])
        config = data.get("config", {})
        
        if len(prompts) > 10:
            return jsonify({
                "success": False,
                "error": "Maximum 10 videos per batch"
            }), 400
        
        results = []
        pipeline = load_video_model()
        
        for i, prompt in enumerate(prompts):
            num_frames = config.get("duration", 4) * config.get("fps", 24)
            
            output = pipeline(
                prompt=prompt,
                width=config.get("width", 720),
                height=config.get("height", 720),
                num_frames=num_frames,
                num_inference_steps=config.get("steps", 25),
            )
            
            frames = output.frames[0]
            video_path = save_video(frames, prefix=f"batch_{i}", fps=config.get("fps", 24))
            
            results.append({
                "index": i,
                "prompt": prompt,
                "video_path": video_path
            })
        
        return jsonify({
            "success": True,
            "total_videos": len(results),
            "videos": results
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    print("Pré-carregando modelo de vídeo...")
    load_video_model()
    print("✅ Modelo de vídeo carregado!")
    
    app.run(host="0.0.0.0", port=8001, debug=False)
