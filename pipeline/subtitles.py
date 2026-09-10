"""Dynamic Subtitle Engine: Punchy 3-5 word animated cards with high-contrast Universal UI typography."""

import os
from typing import List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _split_into_cards(words: List[Dict[str, Any]], max_words_per_card: int = 5) -> List[Dict[str, Any]]:
    """Groups word timestamps into punchy 3-5 word subtitle cards."""
    if not words:
        return []

    cards = []
    current_card_words = []

    for w in words:
        current_card_words.append(w)
        # End card if reached max words or end of sentence / clause
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


def _get_font(size: int):
    """Loads bold sans-serif font across Windows and cross-platform environments."""
    font_candidates = [
        "arialbd.ttf",
        "arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "DejaVuSans-Bold.ttf"
    ]
    for cand in font_candidates:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_subtitle_card(
    frame_img: Image.Image,
    text: str,
    base_font_size: int = 54,
    y_position: int = 910
) -> Image.Image:
    """
    Draws a clean, punchy subtitle card with glassmorphism dark pill background & bold white text.
    Uses proper alpha compositing so underlying video motion remains subtly visible through the pill.
    """
    if not text or not text.strip():
        return frame_img

    frame_w, frame_h = frame_img.size
    max_allowed_text_w = frame_w - 180

    font_size = base_font_size
    font = _get_font(font_size)

    # Temporary measure draw
    dummy_draw = ImageDraw.Draw(frame_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Auto-scale font if text is too wide
    while text_w > max_allowed_text_w and font_size > 28:
        font_size -= 4
        font = _get_font(font_size)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x = (frame_w - text_w) // 2
    y = y_position

    # Pill dimensions
    pad_x = 28
    pad_y = 14
    pill_box = [x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y]

    # Semi-transparent backdrop overlay via alpha composite
    overlay = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    # Dark rounded pill with smooth alpha
    overlay_draw.rounded_rectangle(pill_box, radius=14, fill=(12, 14, 20, 195), outline=(255, 255, 255, 40), width=1)

    frame_rgba = frame_img.convert("RGBA")
    composited = Image.alpha_composite(frame_rgba, overlay)
    draw = ImageDraw.Draw(composited)

    # Black stroke outline for crisp legibility
    stroke_w = 3
    for dx in range(-stroke_w, stroke_w + 1):
        for dy in range(-stroke_w, stroke_w + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))

    # Bold crisp white text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    return composited


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
        et = max(st + 0.4, c["end"] - scene_start_offset)
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

    return clip.fl(fl)
