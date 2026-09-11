"""Video Composer: Builds, animates, and concatenates all scenes with Master Audio muxing."""

import os
import re
import glob
import subprocess
from typing import List, Dict, Any, Optional

try:
    from moviepy.editor import concatenate_videoclips
except ImportError:
    from moviepy import concatenate_videoclips

from .motion import create_ken_burns_clip, TARGET_WIDTH, TARGET_HEIGHT


def _get_ffmpeg_exe() -> str:
    """Finds ffmpeg executable using imageio_ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def find_matching_image(scene_id: str, images_source: Any) -> Optional[str]:
    """
    Finds the image file corresponding to scene_id (e.g. '001', '002').
    Supports:
      - Directory path containing 001.png, flow_001_*.png, scene_1.jpg etc.
      - List of file paths
      - Dict of id -> path
    """
    try:
        clean_id = str(int(scene_id))  # e.g. "1"
        pad_id = clean_id.zfill(3)     # e.g. "001"
    except (ValueError, TypeError):
        clean_id = str(scene_id)
        pad_id = str(scene_id)

    candidates = []
    if isinstance(images_source, str) and os.path.isdir(images_source):
        for ext in ["png", "jpg", "jpeg", "webp"]:
            candidates.extend(glob.glob(os.path.join(images_source, f"*.{ext}")))
            candidates.extend(glob.glob(os.path.join(images_source, f"*.{ext.upper()}")))
    elif isinstance(images_source, list):
        candidates = images_source
    elif isinstance(images_source, dict):
        if scene_id in images_source:
            return images_source[scene_id]
        if pad_id in images_source:
            return images_source[pad_id]
        if clean_id in images_source:
            return images_source[clean_id]
        candidates = list(images_source.values())

    # Priority 1: Exact match (e.g. 001.png) or starts with (001_*, 001-*, 001 *, 001.*)
    for path in candidates:
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename == pad_id or basename == clean_id:
            return path
        if re.match(rf'^{pad_id}[_\-\s\.]', basename) or re.match(rf'^{clean_id}[_\-\s\.]', basename):
            return path

    # Priority 2: Contains (001), [001], (1), or [1]
    for path in candidates:
        basename = os.path.splitext(os.path.basename(path))[0]
        if f"({pad_id})" in basename or f"[{pad_id}]" in basename or f"({clean_id})" in basename or f"[{clean_id}]" in basename:
            return path

    # Priority 3: Contains bounded token *001* (surrounded by non-digits, e.g. flow_001_render, scene-001)
    for path in candidates:
        basename = os.path.splitext(os.path.basename(path))[0]
        if re.search(rf'(?<!\d){pad_id}(?!\d)', basename):
            return path

    # Priority 4: Broad contains *001* substring anywhere in filename
    for path in candidates:
        basename = os.path.splitext(os.path.basename(path))[0]
        if pad_id in basename:
            return path

    # Priority 5: Fallback to non-padded digit token (e.g. scene_1.png)
    for path in candidates:
        basename = os.path.splitext(os.path.basename(path))[0]
        if re.search(rf'(?<!\d){clean_id}(?!\d)', basename):
            return path

    return None


def assemble_video(
    aligned_scenes: List[Dict[str, Any]],
    images_source: Any,
    audio_path: str,
    output_path: str,
    enable_ken_burns: bool = True,
    fps: int = 24,
    progress_callback = None
) -> str:
    """
    Main Assembly Pipeline:
    1. Creates animated video clip for each scene with exact audio duration.
    2. Concatenates all scenes.
    3. Muxes original Master MP3 audio using FFmpeg for pristine quality.
    """
    if not aligned_scenes:
        raise ValueError("No aligned scenes provided.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_video_path = output_path.replace(".mp4", "_temp_silent.mp4")

    clips = []
    total_scenes = len(aligned_scenes)
    # 4 dynamic cinematic movements
    modes = ["zoom_in", "pan_left", "zoom_out", "pan_right"]

    print(f"[composer] Assembling {total_scenes} scenes...")

    try:
        for idx, scene in enumerate(aligned_scenes):
            scene_id = scene["id"]
            duration = scene["duration"]
            if duration <= 0.1:
                duration = 1.0

            if progress_callback:
                progress_callback(idx / total_scenes, f"Processing scene {idx + 1}/{total_scenes} (ID: {scene_id})...")

            # Find matching image
            img_path = find_matching_image(scene_id, images_source)
            if not img_path:
                print(f"[composer] Warning: No image found for scene {scene_id}. Using cinematic placeholder slate.")
                img_path = ""

            # Create motion clip
            mode = modes[idx % len(modes)] if enable_ken_burns else "static"
            clip = create_ken_burns_clip(img_path, duration=duration, mode=mode, fps=fps, scene_id=scene_id)

            clips.append(clip)

        if progress_callback:
            progress_callback(0.85, "Rendering video stream...")

        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")

        # Render intermediate silent video
        final_video.write_videofile(
            temp_video_path,
            fps=fps,
            codec="libx264",
            preset="fast",
            audio=False,
            threads=4
        )

        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        final_video.close()

        if progress_callback:
            progress_callback(0.95, "Muxing master MP3 audio with FFmpeg...")

        ffmpeg_exe = _get_ffmpeg_exe()
        ffmpeg_cmd = [
            ffmpeg_exe, "-y",
            "-i", temp_video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path
        ]

        print(f"[composer] Running FFmpeg audio muxing: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if progress_callback:
            progress_callback(1.0, "Video generation complete!")

        print(f"[composer] Final synchronized video saved to: {output_path}")
        return output_path

    finally:
        # Guarantee cleanup of temporary video
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
