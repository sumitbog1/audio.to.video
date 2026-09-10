"""End-to-end verification test for Audio-to-Video Sync Studio."""

import os
import sys
import shutil
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.aligner import parse_scene_script, align_scenes_to_audio, get_audio_duration
from pipeline.composer import assemble_video, find_matching_image


def create_dummy_images(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    colors = [
        ("001.png", (41, 128, 185), "Scene 001 - Freedom"),
        ("002.png", (142, 68, 173), "Scene 002 - Certainty"),
        ("003.png", (39, 174, 96), "Scene 003 - Inner Power")
    ]
    created = []
    for filename, bg_color, title in colors:
        img = Image.new("RGB", (1920, 1080), color=bg_color)
        draw = ImageDraw.Draw(img)
        # Draw some text
        try:
            font = ImageFont.truetype("arialbd.ttf", 60)
        except Exception:
            font = ImageFont.load_default()
        draw.text((200, 500), title, font=font, fill=(255, 255, 255))
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        created.append(filepath)
    return created


def run_verification():
    audio_path = r"C:\Users\SUMIT\.gemini\antigravity\brain\c6d04366-f5a0-4c18-9f2e-21a691d9ef30\master_ryan_philosophy.wav"
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found at {audio_path}")
        return False

    temp_test_dir = os.path.join(PROJECT_ROOT, "tests", "temp_test_run")
    os.makedirs(temp_test_dir, exist_ok=True)

    try:
        print("[TEST 1] Creating test numbered images...")
        img_paths = create_dummy_images(temp_test_dir)
        print(f"  -> Created {len(img_paths)} test images.")

        print("[TEST 2] Parsing scene script...")
        script_text = """
001: Most people do not truly seek freedom.
002: They seek comfort and certainty.
003: But he who faces the abyss of his own mind finds a power that no circumstance can ever strip away.
        """
        scenes = parse_scene_script(script_text)
        assert len(scenes) == 3, f"Expected 3 scenes, got {len(scenes)}"
        print(f"  -> Successfully parsed {len(scenes)} scenes.")

        print("[TEST 3] Testing image matching resolution...")
        for s in scenes:
            matched = find_matching_image(s["id"], temp_test_dir)
            assert matched is not None, f"Failed to match image for scene {s['id']}"
            print(f"  -> Scene {s['id']} matched image: {os.path.basename(matched)}")

        print("[TEST 4] Aligning scenes to master audio via faster-whisper...")
        audio_dur = get_audio_duration(audio_path)
        print(f"  -> Audio duration: {audio_dur:.2f}s")
        aligned = align_scenes_to_audio(scenes, audio_path, model_size="base")

        print("  -> Alignment Results:")
        for s in aligned:
            print(f"     Scene {s['id']}: {s['start']:.2f}s -> {s['end']:.2f}s (duration: {s['duration']:.2f}s)")
            assert s["duration"] > 0, f"Scene {s['id']} has invalid duration {s['duration']}"

        assert abs(aligned[-1]["end"] - audio_dur) < 0.5, "Last scene end time does not match audio duration"

        print("[TEST 5] Assembling video with Ken Burns motion & punchy subtitles...")
        out_video = os.path.join(temp_test_dir, "test_output_verification.mp4")
        res = assemble_video(
            aligned_scenes=aligned,
            images_source=temp_test_dir,
            audio_path=audio_path,
            output_path=out_video,
            enable_subtitles=True,
            enable_ken_burns=True,
            fps=24
        )

        assert os.path.exists(res), "Output video file was not generated"
        file_size = os.path.getsize(res)
        print(f"  -> Rendered video size: {file_size / (1024*1024):.2f} MB")
        assert file_size > 100000, f"Video file is unexpectedly small: {file_size} bytes"

        print("\n[SUCCESS] ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
        return True

    finally:
        # Cleanup
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
