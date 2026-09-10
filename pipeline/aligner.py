"""Audio Alignment Engine using Whisper word-level timestamps and scene text matching."""

import os
import re
import difflib
import subprocess
from typing import List, Dict, Any, Tuple


_model = None


def get_audio_duration(audio_path: str) -> float:
    """Gets audio duration in seconds using ffprobe/ffmpeg."""
    try:
        from moviepy.editor import AudioFileClip
        clip = AudioFileClip(audio_path)
        dur = float(clip.duration)
        clip.close()
        return dur
    except Exception:
        pass

    # Fallback to ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def load_whisper_model(model_size: str = "base"):
    """Loads faster-whisper model (GPU if CUDA available, else CPU)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[aligner] Loading faster-whisper '{model_size}' model on {device} ({compute_type})...")
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model


def transcribe_audio_words(audio_path: str, model_size: str = "base") -> List[Dict[str, Any]]:
    """Transcribes audio and returns a list of {word, start, end} dicts."""
    model = load_whisper_model(model_size)
    segments, _info = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                cleaned = w.word.strip()
                if cleaned:
                    words.append({
                        "word": cleaned,
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3)
                    })
    return words


def parse_scene_script(script_text: str) -> List[Dict[str, Any]]:
    """
    Parses user script into individual scene blocks.
    Supports formats like:
      001: Narration text...
      001 - Narration text...
      001 Narration text...
      Scene 001: Narration text...
      [001] Narration text...
    """
    if not script_text or not script_text.strip():
        return []

    lines = script_text.strip().split("\n")
    scenes = []
    current_id = None
    current_lines = []

    header_pattern = re.compile(
        r'^\s*(?:scene\s*)?(?:\[\s*)?(\d{1,4})(?:\s*\])?(?:\s*[:\-\.\)\_]\s*|\s+)(.*)$',
        re.IGNORECASE
    )

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        match = header_pattern.match(line_str)
        if match:
            # New scene header found
            raw_id = match.group(1)
            rest_text = match.group(2).strip()

            if current_id is not None:
                scene_text = " ".join(current_lines).strip()
                if scene_text:
                    scenes.append({
                        "id": current_id,
                        "text": scene_text
                    })
                current_lines = []

            # Format ID as 3 digits (e.g. 001)
            current_id = str(int(raw_id)).zfill(3)
            if rest_text:
                current_lines.append(rest_text)
        else:
            if current_id is None:
                # If no initial ID, start with 001
                current_id = "001"
            current_lines.append(line_str)

    if current_id is not None:
        scene_text = " ".join(current_lines).strip()
        if scene_text:
            scenes.append({
                "id": current_id,
                "text": scene_text
            })

    return scenes


def _normalize_text(text: str) -> str:
    """Lowercases and strips non-alphanumeric characters for fuzzy matching."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def align_scenes_to_audio(
    scenes: List[Dict[str, Any]],
    audio_path: str,
    model_size: str = "base"
) -> List[Dict[str, Any]]:
    """
    Aligns each scene's text with the master audio transcription.
    Calculates exact start and end timestamps for every scene.
    Ensures seamless contiguous boundaries: scene[i].end == scene[i+1].start.
    """
    audio_dur = get_audio_duration(audio_path)
    if not scenes:
        return []

    if audio_dur <= 0:
        audio_dur = 10.0

    print(f"[aligner] Transcribing audio '{audio_path}' (duration: {audio_dur:.2f}s)...")
    whisper_words = transcribe_audio_words(audio_path, model_size=model_size)
    total_whisper_words = len(whisper_words)

    if total_whisper_words == 0:
        # Fallback: distribute evenly if whisper returns no words
        print("[aligner] Warning: No words detected by Whisper. Distributing durations evenly.")
        per_scene = audio_dur / len(scenes)
        aligned = []
        for i, s in enumerate(scenes):
            st = round(i * per_scene, 3)
            et = round(audio_dur if i == len(scenes) - 1 else (i + 1) * per_scene, 3)
            aligned.append({
                "id": s["id"],
                "text": s["text"],
                "start": st,
                "end": et,
                "duration": round(et - st, 3),
                "words": []
            })
        return aligned

    # Extract clean word list from Whisper
    w_clean_list = [_normalize_text(w["word"]) for w in whisper_words]

    # Find starting word index for each scene in the whisper word stream
    scene_start_indices = []
    current_search_idx = 0

    for s_idx, scene in enumerate(scenes):
        scene_words = [_normalize_text(w) for w in scene["text"].split() if _normalize_text(w)]
        if not scene_words:
            scene_start_indices.append(current_search_idx)
            continue

        if s_idx == 0:
            scene_start_indices.append(0)
            # Advance search index by roughly half of scene words
            current_search_idx = min(len(scene_words), total_whisper_words - 1)
            continue

        # Look for the first 3-5 words of the scene in remaining whisper words
        probe_len = min(4, len(scene_words))
        probe = " ".join(scene_words[:probe_len])

        best_idx = current_search_idx
        best_ratio = 0.0

        # Slide search window from current_search_idx
        search_range = range(
            current_search_idx,
            min(total_whisper_words - probe_len + 1, current_search_idx + 150)
        )

        for w_i in search_range:
            candidate = " ".join(w_clean_list[w_i:w_i + probe_len])
            ratio = difflib.SequenceMatcher(None, probe, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = w_i
                if ratio > 0.85:
                    break

        scene_start_indices.append(best_idx)
        current_search_idx = max(current_search_idx + 1, best_idx + len(scene_words))

    # Construct boundaries
    aligned_scenes = []
    num_scenes = len(scenes)

    for i in range(num_scenes):
        s_data = scenes[i]
        start_w_idx = scene_start_indices[i]

        if i == 0:
            start_time = 0.0
        else:
            prev_end = aligned_scenes[i - 1]["end"]
            detected_start = whisper_words[start_w_idx]["start"] if start_w_idx < total_whisper_words else prev_end
            start_time = max(prev_end, detected_start)

        if i == num_scenes - 1:
            end_time = audio_dur
        else:
            next_start_w = scene_start_indices[i + 1]
            if next_start_w < total_whisper_words:
                end_time = whisper_words[next_start_w]["start"]
            else:
                end_time = audio_dur

        # Ensure start < end and no gaps
        if end_time <= start_time:
            end_time = start_time + 1.0

        # Associate word timestamps for subtitle generation
        end_w_idx = scene_start_indices[i + 1] if i + 1 < num_scenes else total_whisper_words
        scene_words_slice = whisper_words[start_w_idx:end_w_idx]

        aligned_scenes.append({
            "id": s_data["id"],
            "text": s_data["text"],
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "duration": round(end_time - start_time, 3),
            "words": scene_words_slice
        })

    # Ensure last scene reaches full audio duration
    if aligned_scenes:
        aligned_scenes[-1]["end"] = round(audio_dur, 3)
        aligned_scenes[-1]["duration"] = round(audio_dur - aligned_scenes[-1]["start"], 3)

    return aligned_scenes
