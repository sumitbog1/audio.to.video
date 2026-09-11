"""Cinematic Motion Engine: Ken Burns Pan & Zoom on 16:9 widescreen images."""

import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from typing import Tuple, Any

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def load_and_fit_image(
    image_path: Any,
    scene_id: str = "",
    target_size: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)
) -> Image.Image:
    """
    Loads image and fits it to 16:9 widescreen (1920x1080) with high quality Lanczos downsampling.
    If the image is missing, generates an elegant dark cinematic card indicating the scene ID.
    Supports file path string or PIL.Image.Image instance.
    """
    img = None
    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    elif isinstance(image_path, str) and image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            img = None

    if img is None:
        img = Image.new("RGB", target_size, color=(16, 20, 30))
        draw = ImageDraw.Draw(img)
        title = f"Scene {scene_id}" if scene_id else "Scene"
        sub_label = "Missing Scene Image - Placeholder"
        try:
            f_title = ImageFont.truetype("arialbd.ttf", 64)
            f_sub = ImageFont.truetype("arial.ttf", 36)
        except Exception:
            f_title = ImageFont.load_default()
            f_sub = ImageFont.load_default()

        b1 = draw.textbbox((0, 0), title, font=f_title)
        b2 = draw.textbbox((0, 0), sub_label, font=f_sub)
        w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
        w2, h2 = b2[2] - b2[0], b2[3] - b2[1]

        cx = target_size[0] // 2
        cy = target_size[1] // 2
        draw.text((cx - w1 // 2, cy - 40), title, font=f_title, fill=(220, 230, 245))
        draw.text((cx - w2 // 2, cy + 40), sub_label, font=f_sub, fill=(130, 145, 170))
        return img

    fitted = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
    return fitted


def create_ken_burns_clip(
    image_path: str,
    duration: float,
    mode: str = "zoom_in",
    fps: int = 24,
    scene_id: str = ""
):
    """
    Creates a MoviePy VideoClip applying smooth Ken Burns motion:
      - 'zoom_in': smooth zoom towards center (1.0 -> 1.08)
      - 'zoom_out': smooth zoom out from center (1.08 -> 1.0)
      - 'pan_left': subtle horizontal drift from right to left
      - 'pan_right': subtle horizontal drift from left to right
      - 'static': no motion
    """
    try:
        from moviepy.editor import VideoClip
    except ImportError:
        from moviepy import VideoClip

    base_img = load_and_fit_image(image_path, scene_id=scene_id, target_size=(TARGET_WIDTH, TARGET_HEIGHT))
    w, h = base_img.size
    max_scale = 1.08
    pan_scale = 1.06

    def make_frame(t):
        progress = min(1.0, max(0.0, t / max(duration, 0.001)))

        if mode == "zoom_in":
            scale = 1.0 + (max_scale - 1.0) * progress
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
        elif mode == "zoom_out":
            scale = max_scale - (max_scale - 1.0) * progress
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
        elif mode == "pan_left":
            scale = pan_scale
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            max_drift = w - crop_w
            left = int(max_drift * (1.0 - progress))
            top = (h - crop_h) // 2
        elif mode == "pan_right":
            scale = pan_scale
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            max_drift = w - crop_w
            left = int(max_drift * progress)
            top = (h - crop_h) // 2
        else:
            scale = 1.0
            crop_w = w
            crop_h = h
            left = 0
            top = 0

        cropped = base_img.crop((left, top, left + crop_w, top + crop_h))
        resized = cropped.resize((w, h), Image.Resampling.BILINEAR)
        return np.array(resized)

    clip = VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip
