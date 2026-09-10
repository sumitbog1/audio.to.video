"""Video Composer: Builds, animates, and concatenates all scenes with Master Audio muxing."""

import os
import re
import glob
import subprocess
from typing import List, Dict, Any, Optional

try:
    from moviepy.editor import concatenate_videoclips, AudioFileClip
except ImportError:
    from moviepy import concatenate_videoclips, AudioFileClip

from .motion import create_ken_burns_clip, load_and_fit_image, TARGET_WIDTH, TARGET_HEIGHT
from .subtitles import attach_subtitles_to_clip


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
        # Read files in directory
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

    # Match candidates
    # Priority 1: Exact prefix or exact name (e.g. 001.png, 001_foo.png, flow_001_*.png)
    for path in candidates:
        basename = os.path.splitext(os.path.basename(path))[0]
        # Check patterns like "001", "flow_001_...", "scene_001", "001-..."
        if basename == pad_id or basename == clean_id:
            return path
        if re.search(rf'(^|[^0-9]){pad_id}([^0-9]|$)', basename):
            return path

    # Priority 2: Fallback to non-padded digit match
    for path in candidates:
        basename = os.path.splitext(os.path.basename(path))[0]
        if re.search(rf'(^|[^0-9]){clean_id}([^0-9]|$)', basename):
            return path

    return None


def assemble_video(
    aligned_scenes: List[Dict[str, Any]],
    images_source: Any,
    audio_path: str,
    output_path: str,
    enable_subtitles: bool = True,
    enable_ken_burns: bool = True,
    fps: int = 24,
    progress_callback = None
) -> str:
    """
    Main Assembly Pipeline:
    1. Creates animated video clip for each scene with exact audio duration.
    2. Overlays synchronized subtitles (if enabled).
    3. Concatenates all scenes.
    4. Muxes original Master MP3 audio using FFmpeg for pristine quality.
    """
    if not aligned_scenes:
        raise ValueError("No aligned scenes provided.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_video_path = output_path.replace(".mp4", "_temp_silent.mp4")

    clips = []
    total_scenes = len(aligned_scenes)
    modes = ["zoom_in", "zoom_out"]

    print(f"[composer] Assembling {total_scenes} scenes...")

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
            print(f"[composer] Warning: No image found for scene {scene_id}. Using blank placeholder.")
            img_path = ""

        # Create motion clip
        mode = modes[idx % len(modes)] if enable_ken_burns else "static"
        clip = create_ken_burns_clip(img_path, duration=duration, mode=mode, fps=fps)

        # Attach subtitles
        if enable_subtitles and scene.get("words"):
            clip = attach_subtitles_to_clip(clip, scene, scene_start_offset=scene["start"])

        clips.append(clip)

    if progress_callback:
        progress_callback(0.85, "Rendering video stream...")

    # Concatenate all clips
    final_video = concatenate_videoclips(clips, method="compose")

    # Render intermediate video
    final_video.write_videofile(
        temp_video_path,
        fps=fps,
        codec="libx264",
        preset="fast",
        audio=False,
        threads=4
    )

    # Close moviepy clips
    for c in clips:
        try:
            c.close()
        except Exception:
            pass
    final_video.close()

    if progress_callback:
        progress_callback(0.95, "Muxing master MP3 audio with FFmpeg...")

    # Fast, lossless audio muxing with FFmpeg
    # Merges original audio with video stream, trimming video or audio to exact match
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

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

    # Clean up temp video
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    if progress_callback:
        progress_callback(1.0, "Video generation complete! 🎉")

    print(f"[composer] Final synchronized video saved to: {output_path}")
    return output_path
