"""
Minimalist Editorial Line-Art Image Generator (Dan Koe / Atomic Habits style).
Fast local diffusion pipeline on CUDA GPU producing clean black-ink editorial illustrations
on flat warm cream canvas background.
"""

import os
import gc
import time
from typing import List, Optional, Tuple
from PIL import Image

_pipe = None
MODEL_ID = "Lykon/dreamshaper-8"

# Tailored aesthetic prompt & negative prompt for Dan Koe / Atomic Habits style
EDITORIAL_STYLE_SUFFIX = (
    ", minimalist editorial line art illustration, hand-drawn black ink pen sketch, "
    "flat warm cream canvas background, Dan Koe aesthetic, Atomic Habits book illustration, "
    "clean conceptual diagram, high contrast monochrome drawing"
)

DEFAULT_NEGATIVE_PROMPT = (
    "photorealistic, 3d render, cgi, realistic photo, colorful gradients, "
    "vibrant saturated colors, neon, blurry, dark background, wooden table, "
    "desk, frame, margins, borders, low quality, distorted anatomy, extra limbs"
)


def get_image_pipeline():
    """Lazily loads local Dreamshaper-8 diffusion pipeline in float16 on CUDA."""
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[image_gen] Loading local line-art model '{MODEL_ID}' on {device} ({dtype})...")
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
        print("[image_gen] Line-art diffusion model unloaded from VRAM.")


def prepare_editorial_prompt(prompt: str) -> str:
    """Ensures prompt adheres to minimalist editorial line art aesthetic on warm cream background."""
    p = prompt.strip()
    keywords = ["minimalist", "editorial", "dan koe", "line art", "line-art", "atomic habits"]
    if not any(k in p.lower() for k in keywords):
        p = f"{p}{EDITORIAL_STYLE_SUFFIX}"
    elif "cream" not in p.lower() and "canvas" not in p.lower() and "background" not in p.lower():
        p = f"{p}, flat warm cream canvas background, clean black ink line art"
    return p


def generate_single_image(
    prompt: str,
    output_path: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 768,
    height: int = 512,
    num_steps: int = 25,
    guidance_scale: float = 7.5
) -> str:
    """Generates a single minimalist editorial line-art image from text prompt."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pipe = get_image_pipeline()

    final_prompt = prepare_editorial_prompt(prompt)

    image = pipe(
        prompt=final_prompt,
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
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 768,
    height: int = 512,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    progress_callback = None
) -> List[str]:
    """
    Generates minimalist editorial line-art images in bulk from simple text lines (1 prompt per line).
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
            progress_callback(idx / total, f"Generating line-art illustration {idx + 1}/{total} ({scene_num}.png)...")

        final_prompt = prepare_editorial_prompt(prompt)

        img = pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]

        img.save(file_path)
        created_paths.append(file_path)

    if progress_callback:
        progress_callback(1.0, "All line-art scene images generated successfully!")

    return created_paths
