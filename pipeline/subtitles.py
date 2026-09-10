"""Dynamic Subtitle Engine: Punchy 3-5 word animated cards with high-contrast Universal UI typography."""

import os
import re
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont


def _split_into_cards(words: List[Dict[str, Any]], max_words_per_card: int = 5) -> List[Dict[str, Any]]:
    """Groups word timestamps into punchy 3-5 word subtitle cards."""
    if not words:
        return []

    cards = []
    current_card_words = []

    for w in words:
        current_card_words.append(w)
        # End card if reach max words or punctuation
        has_punct = any(w["word"].endswith(p) for p in [".", "!", "?", ",", ";", ":", "—", "-"])
        if len(current_card_words) >= max_words_per_card or (has_punct and len(current_card_words) >= 3):
            cards.append({
                "text": " ".join(item["word"] for item in current_card_words),
                "start": current_card_words[0]["start"],
                "end": current_card_words[-1]["end"]
            })
            current_card_words = []

    if current_card_words:
        cards.append({
            "text": " ".join(item["word"] for item in current_card_words),
            "start": current_card_words[0]["start"],
            "end": current_card_words[-1]["end"]
        })

    return cards


def draw_subtitle_card(
    frame_img: Image.Image,
    text: str,
    font_size: int = 52,
    y_position: int = 900
) -> Image.Image:
    """Draws a clean, punchy subtitle card with dark pill background & bold white text."""
    if not text or not text.strip():
        return frame_img

    draw = ImageDraw.Draw(frame_img)
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    frame_w, frame_h = frame_img.size
    x = (frame_w - text_w) // 2
    y = y_position

    # Draw rounded dark pill backdrop
    pad_x = 24
    pad_y = 12
    pill_box = [x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y]
    draw.rounded_rectangle(pill_box, radius=12, fill=(0, 0, 0, 180))

    # Draw black outline
    stroke_w = 3
    for dx in range(-stroke_w, stroke_w + 1):
        for dy in range(-stroke_w, stroke_w + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))

    # Draw crisp white text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    return frame_img


def attach_subtitles_to_clip(clip, scene_data: Dict[str, Any], scene_start_offset: float):
    """Overlays synchronized subtitles onto a scene VideoClip."""
    words = scene_data.get("words", [])
    if not words:
        return clip

    cards = _split_into_cards(words, max_words_per_card=5)
    if not cards:
        return clip

    # Adjust card times relative to scene start
    rel_cards = []
    for c in cards:
        st = max(0.0, c["start"] - scene_start_offset)
        et = max(st + 0.5, c["end"] - scene_start_offset)
        rel_cards.append({
            "text": c["text"],
            "start": st,
            "end": min(clip.duration, et)
        })

    def fl(gf, t):
        frame = gf(t)
        # Find active card at time t
        active_card = None
        for c in rel_cards:
            if c["start"] <= t <= c["end"]:
                active_card = c
                break

        if active_card:
            pil_img = Image.fromarray(frame).convert("RGBA")
            drawn = draw_subtitle_card(pil_img, active_card["text"])
            return np.array(drawn.convert("RGB"))
        return frame

    import numpy as np
    return clip.fl(fl)
