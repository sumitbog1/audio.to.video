"""Command-line interface for Audio-to-Video Sync Studio."""

import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pipeline.aligner import parse_scene_script, align_scenes_to_audio, get_audio_duration
from pipeline.composer import assemble_video, find_matching_image
from pipeline.tts import synthesize_scenes_to_master_mp3, get_voice_choices, DEFAULT_VOICE


def main():
    parser = argparse.ArgumentParser(
        description="Audio-to-Video Studio: Synchronize numbered scenes and images with master audio or AI voice."
    )
    parser.add_argument(
        "--audio", "-a", default=None,
        help="Path to existing master MP3/WAV audio file. If omitted, voice is synthesized from script."
    )
    parser.add_argument(
        "--script", "-s", required=True,
        help="Path to script text file OR string containing numbered scenes (001: ...)"
    )
    parser.add_argument(
        "--images", "-i", default="",
        help="Path to folder containing numbered images (001.png, 002.png...) or comma-separated paths"
    )
    parser.add_argument(
        "--voice", default=DEFAULT_VOICE,
        help=f"Voice name for AI speech synthesis (default: {DEFAULT_VOICE})"
    )
    parser.add_argument(
        "--speed", type=int, default=0,
        help="Voice speech speed adjustment percent (-20 to +20). Default: 0"
    )
    parser.add_argument(
        "--output", "-o", default="outputs/synced_output.mp4",
        help="Path for output MP4 video (default: outputs/synced_output.mp4)"
    )
    parser.add_argument(
        "--no-ken-burns", action="store_true",
        help="Disable cinematic Ken Burns pan/zoom motion"
    )
    parser.add_argument(
        "--preview-only", action="store_true",
        help="Only display parsed scene timestamps and matched images without rendering video"
    )
    parser.add_argument(
        "--model", default="base",
        help="Whisper model size (tiny, base, small, medium). Default: base"
    )

    args = parser.parse_args()

    # Load script text
    if os.path.exists(args.script):
        with open(args.script, "r", encoding="utf-8") as f:
            script_text = f.read()
    else:
        script_text = args.script

    scenes = parse_scene_script(script_text)
    if not scenes:
        print("❌ Error: No numbered scenes found in script. Use formats like '001: Text', '002: Text'.")
        sys.exit(1)

    # Determine audio source
    audio_path = args.audio
    if not audio_path or not os.path.exists(audio_path):
        print(f"🎙️ No master audio provided. Synthesizing AI voice using '{args.voice}'...")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        gen_audio_path = args.output.replace(".mp4", "_master_voice.mp3")
        audio_path = synthesize_scenes_to_master_mp3(
            scenes=scenes,
            voice=args.voice,
            speed_percent=args.speed,
            output_path=gen_audio_path,
            progress_callback=lambda f, desc: print(f"  [{int(f*100):3d}%] {desc}")
        )
        print(f"✅ Generated master audio: {audio_path}")

    # Images source
    if os.path.isdir(args.images):
        img_src = args.images
    elif "," in args.images:
        img_src = [p.strip() for p in args.images.split(",") if os.path.exists(p.strip())]
    elif os.path.exists(args.images):
        img_src = [args.images]
    else:
        img_src = []

    total_dur = get_audio_duration(audio_path)
    print(f"\n=======================================================")
    print(f"🎵 Audio: {os.path.basename(audio_path)} ({total_dur:.2f}s)")
    print(f"📑 Scenes parsed: {len(scenes)}")
    print(f"=======================================================\n")

    print("⏳ Aligning scenes with Whisper speech recognition...")
    aligned = align_scenes_to_audio(scenes, audio_path, model_size=args.model)

    print("\n--- Detected Scene Timestamps & Images ---")
    for s in aligned:
        matched_img = find_matching_image(s["id"], img_src)
        matched_name = os.path.basename(matched_img) if matched_img else "CINEMATIC SLATE (Placeholder)"
        print(f"Scene {s['id']}: [{s['start']:.2f}s -> {s['end']:.2f}s] (dur: {s['duration']:.2f}s) | Image: {matched_name}")
        print(f"  Narration: \"{s['text'][:60]}...\"" if len(s['text']) > 60 else f"  Narration: \"{s['text']}\"")

    if args.preview_only:
        print("\n[SUCCESS] Preview finished (--preview-only specified). Exiting.")
        return

    print("\n🎬 Rendering synchronized master 1080p video...")
    out_file = assemble_video(
        aligned_scenes=aligned,
        images_source=img_src,
        audio_path=audio_path,
        output_path=args.output,
        enable_ken_burns=not args.no_ken_burns,
        fps=24,
        progress_callback=lambda f, desc: print(f"  [{int(f*100):3d}%] {desc}")
    )

    print(f"\n[SUCCESS] Synced video rendered: {os.path.abspath(out_file)}")


if __name__ == "__main__":
    main()
