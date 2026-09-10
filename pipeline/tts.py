"""TTS Engine: High-quality AI speech synthesis using Edge-TTS neural voices."""

import os
import re
import time
import asyncio
import tempfile
from typing import List, Dict, Any, Optional
import edge_tts

try:
    from moviepy.editor import AudioFileClip, concatenate_audioclips
except ImportError:
    from moviepy import AudioFileClip, concatenate_audioclips

VOICE_MAP = {
    "Ryan – Rich, Smooth & Engaging (British)": "en-GB-RyanNeural",
    "Brian – Deep Cinematic Documentary (American)": "en-US-BrianNeural",
    "Eric – Deep Authoritative Baritone (American)": "en-US-EricNeural",
    "Guy – Passionate Storyteller (American)": "en-US-GuyNeural",
    "Christopher – Warm Explainer (American)": "en-US-ChristopherNeural",
    "Jenny – Expressive Female Narrator (American)": "en-US-JennyNeural",
    "Madhur – Hindi Male Storyteller (मधुर)": "hi-IN-MadhurNeural",
    "Swara – Hindi Female Explainer (स्वरा)": "hi-IN-SwaraNeural",
}

DEFAULT_VOICE = "Brian – Deep Cinematic Documentary (American)"


def get_voice_choices() -> List[str]:
    """Returns list of curated voice display names."""
    return list(VOICE_MAP.keys())


def resolve_voice_id(voice_name: str) -> str:
    """Maps display voice name or raw voice ID to Microsoft Neural voice ID."""
    if voice_name in VOICE_MAP:
        return VOICE_MAP[voice_name]
    for k, v in VOICE_MAP.items():
        if voice_name.lower() in k.lower() or voice_name.lower() in v.lower():
            return v
    return "en-US-BrianNeural"


def _clean_text_for_tts(text: str) -> str:
    """Strips scene prefixes (e.g. '001:', 'Scene 001 - ') and clean formatting."""
    t = re.sub(r'^\s*(?:scene\s*)?(?:\[\s*)?\d{1,4}(?:\s*\])?(?:\s*[:\-\.\)\_]\s*|\s+)', '', text, flags=re.IGNORECASE)
    # Normalize double spaces and quotes
    t = re.sub(r'\s+', ' ', t).strip()
    return t


async def _synthesize_clip_async(text: str, voice_id: str, rate_str: str, out_file: str):
    communicate = edge_tts.Communicate(text, voice_id, rate=rate_str)
    await communicate.save(out_file)


def synthesize_text_to_mp3(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed_percent: int = 0,
    output_path: Optional[str] = None
) -> str:
    """Synthesizes text into a single MP3 audio file."""
    if not output_path:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "generated_narration.mp3")

    voice_id = resolve_voice_id(voice)
    cleaned = _clean_text_for_tts(text)
    rate_str = f"{'+' if speed_percent >= 0 else ''}{speed_percent}%"

    try:
        asyncio.run(_synthesize_clip_async(cleaned, voice_id, rate_str, output_path))
    except RuntimeError:
        # If loop is already running in current thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(lambda: asyncio.run(_synthesize_clip_async(cleaned, voice_id, rate_str, output_path)))
            fut.result(timeout=30.0)

    return output_path


def synthesize_scenes_to_master_mp3(
    scenes: List[Dict[str, Any]],
    voice: str = DEFAULT_VOICE,
    speed_percent: int = 0,
    output_path: Optional[str] = None,
    progress_callback = None
) -> str:
    """
    Synthesizes each scene individually and concatenates them into a single Master MP3.
    Ensures natural cadence and distinct boundaries between scenes.
    """
    if not scenes:
        raise ValueError("No scenes provided for synthesis.")

    if not output_path:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"master_narration_{int(time.time())}.mp3")

    voice_id = resolve_voice_id(voice)
    rate_str = f"{'+' if speed_percent >= 0 else ''}{speed_percent}%"

    temp_clips = []
    audio_clips = []
    temp_dir = tempfile.mkdtemp(prefix="tts_scenes_")

    try:
        total = len(scenes)
        for idx, s in enumerate(scenes):
            if progress_callback:
                progress_callback(idx / total, f"Synthesizing voice for Scene {s['id']} ({idx + 1}/{total})...")

            text = _clean_text_for_tts(s["text"])
            if not text:
                continue

            scene_audio_path = os.path.join(temp_dir, f"scene_{s['id']}.mp3")
            try:
                asyncio.run(_synthesize_clip_async(text, voice_id, rate_str, scene_audio_path))
            except RuntimeError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    fut = executor.submit(lambda: asyncio.run(_synthesize_clip_async(text, voice_id, rate_str, scene_audio_path)))
                    fut.result(timeout=30.0)

            temp_clips.append(scene_audio_path)
            audio_clips.append(AudioFileClip(scene_audio_path))

        if not audio_clips:
            raise ValueError("No valid audio was synthesized.")

        if progress_callback:
            progress_callback(0.9, "Concatenating scenes into master MP3 audio...")

        # Concat all audio clips
        final_audio = concatenate_audioclips(audio_clips)
        final_audio.write_audiofile(output_path, fps=44100, logger=None)
        final_audio.close()

        for c in audio_clips:
            try:
                c.close()
            except Exception:
                pass

        if progress_callback:
            progress_callback(1.0, "Master MP3 synthesized successfully!")

        print(f"[tts] Master MP3 saved to: {output_path}")
        return output_path

    finally:
        # Cleanup temp scene audio files
        for p in temp_clips:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass
