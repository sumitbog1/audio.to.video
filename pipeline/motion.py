"""Cinematic Motion Engine: Ken Burns Pan & Zoom on 16:9 widescreen images."""

import os
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def load_and_fit_image(image_path: str, target_size: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)) -> Image.Image:
    """Loads image and fits it to 16:9 widescreen (1920x1080) with high quality lanczos downsampling."""
    if not os.path.exists(image_path):
        # Create a neutral dark placeholder if missing
        img = Image.new("RGB", target_size, color=(20, 25, 40))
        return img

    img = Image.open(image_path).convert("RGB")
    # Fit into 16:9 with center crop
    fitted = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
    return fitted


def create_ken_burns_clip(
    image_path: str,
    duration: float,
    mode: str = "zoom_in",
    fps: int = 24
):
    """
    Creates a MoviePy VideoClip applying smooth Ken Burns motion (zoom-in, zoom-out, pan).
    Duration matches the scene audio duration exactly.
    """
    try:
        from moviepy.editor import VideoClip
    except ImportError:
        from moviepy import VideoClip

    base_img = load_and_fit_image(image_path, (TARGET_WIDTH, TARGET_HEIGHT))
    w, h = base_img.size

    # Zoom range (1.0 to 1.08)
    max_scale = 1.08

    def make_frame(t):
        progress = min(1.0, max(0.0, t / max(duration, 0.001)))

        if mode == "zoom_in":
            scale = 1.0 + (max_scale - 1.0) * progress
        elif mode == "zoom_out":
            scale = max_scale - (max_scale - 1.0) * progress
        else:
            scale = 1.04

        # Calculate crop rectangle for centered zoom
        crop_w = int(w / scale)
        crop_h = int(h / scale)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2

        cropped = base_img.crop((left, top, left + crop_w, top + crop_h))
        resized = cropped.resize((w, h), Image.Resampling.BILINEAR)
        return np.array(resized)

    clip = VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip
