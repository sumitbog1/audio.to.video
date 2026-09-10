"""Local Image Generator using cached Dreamshaper-8 (SD1.5) on CUDA GPU."""

import os
import gc
import time
from typing import List, Optional, Tuple
from PIL import Image

_pipe = None
MODEL_ID = "Lykon/dreamshaper-8"


def get_image_pipeline():
    """Lazily loads local Dreamshaper-8 diffusion pipeline in float16 on CUDA."""
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[image_gen] Loading local '{MODEL_ID}' on {device} ({dtype})...")
        try:
            _pipe = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                safety_checker=None,
                torch_dtype=dtype,
                local_files_only=True
            ).to(device)
        except Exception:
            # Fallback if local_files_only needs HF cache search
            _pipe = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                safety_checker=None,
                torch_dtype=dtype
            ).to(device)

        if device == "cuda":
            # Enable memory efficient attention if available
            try:
                _pipe.enable_attention_slicing()
            except Exception:
                pass

    return _pipe


def unload_image_model():
    """Unloads diffusion pipeline from VRAM to keep GPU free for Whisper or Video encoding."""
    global _pipe
    if _pipe is not None:
        del _pipe
        _pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[image_gen] Diffusion model unloaded from VRAM.")


def generate_single_image(
    prompt: str,
    output_path: str,
    negative_prompt: str = "ugly, blurry, low quality, distorted, extra limbs, bad anatomy",
    width: int = 768,
    height: int = 512,
    num_steps: int = 20,
    guidance_scale: float = 7.0
) -> str:
    """Generates a single image from text prompt and saves to output_path."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pipe = get_image_pipeline()

    # Append quality booster tags
    clean_prompt = prompt.strip()
    if not any(k in clean_prompt.lower() for k in ["photorealistic", "cinematic", "8k", "masterpiece", "detailed"]):
        clean_prompt += ", cinematic lighting, detailed, 8k"

    image = pipe(
        prompt=clean_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale
    ).images[0]

    image.save(output_path)
    return output_path


def generate_bulk_images(
    prompts_text: str,
    output_dir: str,
    negative_prompt: str = "ugly, blurry, low quality, distorted, bad anatomy",
    width: int = 768,
    height: int = 512,
    num_steps: int = 20,
    progress_callback = None
) -> List[str]:
    """
    Generates images in bulk from simple text lines (1 prompt per line).
    Automatically saves them as 001.png, 002.png, 003.png...
    matching Tab 2's scene image expectations!
    """
    lines = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]
    if not lines:
        raise ValueError("No prompt lines provided.")

    os.makedirs(output_dir, exist_ok=True)
    pipe = get_image_pipeline()

    created_paths = []
    total = len(lines)

    for idx, prompt in enumerate(lines):
        scene_num = str(idx + 1).zfill(3)
        file_path = os.path.join(output_dir, f"{scene_num}.png")

        if progress_callback:
            progress_callback(idx / total, f"Generating image {idx + 1}/{total} ({scene_num}.png)...")

        clean_prompt = prompt
        if not any(k in clean_prompt.lower() for k in ["photorealistic", "cinematic", "8k", "masterpiece"]):
            clean_prompt += ", cinematic lighting, detailed, 8k"

        img = pipe(
            prompt=clean_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=7.0
        ).images[0]

        img.save(file_path)
        created_paths.append(file_path)

    if progress_callback:
        progress_callback(1.0, "All images generated successfully!")

    return created_paths
