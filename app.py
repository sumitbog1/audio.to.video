"""Gradio Web Studio for Audio-to-Video Sync Studio."""

import os
import time
import gradio as gr
from pipeline.aligner import parse_scene_script, align_scenes_to_audio, get_audio_duration
from pipeline.composer import assemble_video, find_matching_image
from pipeline.tts import synthesize_scenes_to_master_mp3, get_voice_choices, DEFAULT_VOICE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Tab 1: Script to AI Voice & Video
# ----------------------------------------------------------------------

def generate_voice_only(script_text, voice, speed_percent, progress=gr.Progress()):
    """Synthesizes master MP3 narration from script text."""
    if not script_text or not script_text.strip():
        raise gr.Error("Please paste your scene narration script.")

    scenes = parse_scene_script(script_text)
    if not scenes:
        raise gr.Error("Could not parse numbered scenes (e.g. 001:, 002:).")

    progress(0.1, desc="Synthesizing AI voice narration...")
    timestamp = int(time.time())
    out_mp3 = os.path.join(OUTPUT_DIR, f"master_voice_{timestamp}.mp3")

    def tts_cb(frac, desc):
        progress(frac, desc=desc)

    synthesize_scenes_to_master_mp3(
        scenes=scenes,
        voice=voice,
        speed_percent=speed_percent,
        output_path=out_mp3,
        progress_callback=tts_cb
    )

    total_dur = get_audio_duration(out_mp3)
    return out_mp3, f"✅ Voice MP3 generated successfully! ({total_dur:.2f}s) Saved to: `{out_mp3}`"


def generate_full_from_script(
    script_text,
    voice,
    speed_percent,
    images_files,
    folder_path,
    enable_subtitles,
    enable_ken_burns,
    whisper_model,
    language_choice,
    progress=gr.Progress()
):
    """1-Click: Synthesizes AI voice narration, aligns scenes, and generates full 1080p video."""
    if not script_text or not script_text.strip():
        raise gr.Error("Please paste your scene narration script.")

    scenes = parse_scene_script(script_text)
    if not scenes:
        raise gr.Error("Could not parse numbered scenes (e.g. 001:, 002:).")

    # 1. Synthesize Master Audio
    progress(0.05, desc="Synthesizing AI speech narration...")
    timestamp = int(time.time())
    master_mp3 = os.path.join(OUTPUT_DIR, f"master_voice_{timestamp}.mp3")

    def tts_cb(frac, desc):
        progress(0.05 + frac * 0.25, desc=desc)

    synthesize_scenes_to_master_mp3(
        scenes=scenes,
        voice=voice,
        speed_percent=speed_percent,
        output_path=master_mp3,
        progress_callback=tts_cb
    )

    # 2. Gather image sources
    img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
    if not img_src and images_files:
        img_src = [f.name if hasattr(f, "name") else f for f in images_files]

    # 3. Forced Alignment with Whisper
    progress(0.35, desc=f"Aligning narration with Faster-Whisper ({whisper_model})...")
    lang = None if language_choice == "Auto-Detect" else language_choice.lower()
    aligned = align_scenes_to_audio(scenes, master_mp3, model_size=whisper_model, language=lang)

    # 4. Video Assembly
    out_video = os.path.join(OUTPUT_DIR, f"synced_video_{timestamp}.mp4")

    def comp_cb(frac, desc):
        progress(0.45 + frac * 0.50, desc=desc)

    assemble_video(
        aligned_scenes=aligned,
        images_source=img_src,
        audio_path=master_mp3,
        output_path=out_video,
        enable_subtitles=enable_subtitles,
        enable_ken_burns=enable_ken_burns,
        fps=24,
        progress_callback=comp_cb
    )

    # Build preview table
    md = [
        f"### 🎵 Generated Voice Duration: `{get_audio_duration(master_mp3):.2f}s` | Total Scenes: `{len(aligned)}`\n",
        "| Scene ID | Start Time | End Time | Duration | Matching Image | Spoken Text Preview |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for s in aligned:
        matched_img = find_matching_image(s["id"], img_src)
        img_status = f"✅ `{os.path.basename(matched_img)}`" if matched_img else "🎨 *Cinematic Slate*"
        preview_text = s['text'][:45] + ("..." if len(s['text']) > 45 else "")
        md.append(f"| **{s['id']}** | `{s['start']:.2f}s` | `{s['end']:.2f}s` | `{s['duration']:.2f}s` | {img_status} | {preview_text} |")

    return master_mp3, out_video, f"✅ Complete video & voice generated successfully! Saved to: `{out_video}`", "\n".join(md)


# ----------------------------------------------------------------------
# Tab 2: Existing Master Audio Sync
# ----------------------------------------------------------------------

def preview_alignment(audio_file, script_text, images_files, folder_path, model_size, language_choice):
    """Parses script and aligns with uploaded MP3 audio, showing detected scene timestamps."""
    if not audio_file:
        return "❌ Please upload an MP3/WAV audio file first.", None

    if not script_text or not script_text.strip():
        return "❌ Please paste your scene narration script.", None

    audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file
    scenes = parse_scene_script(script_text)
    if not scenes:
        return "❌ Could not parse any numbered scenes (e.g., 001:, 002:).", None

    img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
    if not img_src and images_files:
        img_src = [f.name if hasattr(f, "name") else f for f in images_files]

    total_dur = get_audio_duration(audio_path)
    lang = None if language_choice == "Auto-Detect" else language_choice.lower()
    aligned = align_scenes_to_audio(scenes, audio_path, model_size=model_size, language=lang)

    md = [
        f"### 🎵 Master Audio Duration: `{total_dur:.2f}s` | Total Scenes: `{len(aligned)}`\n",
        "| Scene ID | Start Time | End Time | Duration | Matching Image | Spoken Text Preview |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for s in aligned:
        matched_img = find_matching_image(s["id"], img_src)
        img_status = f"✅ `{os.path.basename(matched_img)}`" if matched_img else "🎨 *Cinematic Slate*"
        preview_text = s['text'][:45] + ("..." if len(s['text']) > 45 else "")
        md.append(f"| **{s['id']}** | `{s['start']:.2f}s` | `{s['end']:.2f}s` | `{s['duration']:.2f}s` | {img_status} | {preview_text} |")

    return "\n".join(md), aligned


def build_full_video_existing_audio(
    audio_file,
    script_text,
    images_files,
    folder_path,
    enable_subtitles,
    enable_ken_burns,
    model_size,
    language_choice,
    progress=gr.Progress()
):
    """Generates the full master 1080p video synchronized to the uploaded MP3 audio."""
    if not audio_file:
        raise gr.Error("Please upload an MP3/WAV audio file.")

    if not script_text or not script_text.strip():
        raise gr.Error("Please paste your scene narration script.")

    audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file

    progress(0.05, desc="Parsing script...")
    scenes = parse_scene_script(script_text)
    if not scenes:
        raise gr.Error("No valid numbered scenes found in script.")

    img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
    if not img_src and images_files:
        img_src = [f.name if hasattr(f, "name") else f for f in images_files]

    progress(0.15, desc=f"Aligning scenes with Faster-Whisper ({model_size})...")
    lang = None if language_choice == "Auto-Detect" else language_choice.lower()
    aligned = align_scenes_to_audio(scenes, audio_path, model_size=model_size, language=lang)

    timestamp = int(time.time())
    output_video_path = os.path.join(OUTPUT_DIR, f"synced_video_{timestamp}.mp4")

    def p_cb(fraction, desc):
        progress(0.25 + fraction * 0.70, desc=desc)

    assemble_video(
        aligned_scenes=aligned,
        images_source=img_src,
        audio_path=audio_path,
        output_path=output_video_path,
        enable_subtitles=enable_subtitles,
        enable_ken_burns=enable_ken_burns,
        fps=24,
        progress_callback=p_cb
    )

    return output_video_path, f"✅ Video generated successfully! Saved to: `{output_video_path}`"


# ----------------------------------------------------------------------
# Gradio Studio Layout
# ----------------------------------------------------------------------

custom_css = """
.gradio-container { max-width: 1250px !important; margin: auto; }
.header-box { text-align: center; margin-bottom: 20px; }
.action-btn-primary { background: linear-gradient(90deg, #2563eb, #3b82f6) !important; color: white !important; font-weight: 600; }
"""

sample_script = """001: Most people do not truly seek freedom.
002: They seek comfort, security, and certainty in an unpredictable world.
003: But he who dares to face the silence of his own mind unlocks an eternal power."""

with gr.Blocks(title="Audio.to.Video Studio", css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # 🎬 Audio-to-Video Studio
        ### Professional 1080p Video Creation from Text Scripts, AI Voices, and Numbered Images (`001.png`...)
        """,
        elem_classes=["header-box"]
    )

    with gr.Tabs():
        # TAB 1: SCRIPT TO AI VOICE & VIDEO
        with gr.TabItem("🎙️ 1. Script to AI Voice & Video (Full Auto)"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 📝 Script & Voice Settings")
                    t1_script = gr.Textbox(
                        label="Numbered Scene Script",
                        lines=8,
                        placeholder="001: Scene one text...\n002: Scene two text...\n003: Scene three text...",
                        value=sample_script
                    )

                    with gr.Row():
                        t1_voice = gr.Dropdown(
                            label="Narrator Voice",
                            choices=get_voice_choices(),
                            value=DEFAULT_VOICE
                        )
                        t1_speed = gr.Slider(
                            label="Speech Speed (%)",
                            minimum=-25,
                            maximum=25,
                            step=1,
                            value=0
                        )

                    with gr.Accordion("🖼️ Scene Images (001.png, 002.png...)", open=True):
                        t1_folder = gr.Textbox(
                            label="📁 Local Images Folder Path (Optional)",
                            placeholder=r"e.g. D:\Projects\images"
                        )
                        t1_images = gr.File(
                            label="📤 Upload Numbered Image Files",
                            file_count="multiple",
                            file_types=["image"]
                        )

                    with gr.Row():
                        t1_ken_burns = gr.Checkbox(label="🎥 Ken Burns Motion (Pan & Zoom)", value=True)
                        t1_subtitles = gr.Checkbox(label="💬 Universal UI Subtitles", value=True)

                    with gr.Row():
                        t1_voice_btn = gr.Button("🎙️ 1. Generate Voice MP3 Only", variant="secondary")
                        t1_full_btn = gr.Button("⚡ 2. Generate Complete 1080p Video", variant="primary")

                with gr.Column(scale=5):
                    gr.Markdown("### 🎞️ Output & Preview")
                    t1_audio_out = gr.Audio(label="🎵 Master Voice Narration (MP3)", interactive=False)
                    t1_video_out = gr.Video(label="🎬 Final Synchronized Video (1080p)", interactive=False)
                    t1_status = gr.Markdown("Status: *Ready to generate.*")

                    with gr.Accordion("📊 Scene Timing Inspector", open=False):
                        t1_timing_md = gr.Markdown("Scene timings will appear here after generation.")

            # Tab 1 Events
            t1_voice_btn.click(
                fn=generate_voice_only,
                inputs=[t1_script, t1_voice, t1_speed],
                outputs=[t1_audio_out, t1_status]
            )

            t1_full_btn.click(
                fn=generate_full_from_script,
                inputs=[
                    t1_script,
                    t1_voice,
                    t1_speed,
                    t1_images,
                    t1_folder,
                    t1_subtitles,
                    t1_ken_burns,
                    gr.State("base"),
                    gr.State("Auto-Detect")
                ],
                outputs=[t1_audio_out, t1_video_out, t1_status, t1_timing_md]
            )

        # TAB 2: EXISTING AUDIO SYNC
        with gr.TabItem("🎵 2. Sync Existing Audio File (Master MP3)"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 🎵 Audio & Script Input")
                    t2_audio = gr.Audio(label="Upload Master MP3/WAV Audio", type="filepath")
                    t2_script = gr.Textbox(
                        label="Numbered Scene Script",
                        lines=8,
                        placeholder="001: Scene one text...\n002: Scene two text...",
                        value=sample_script
                    )

                    with gr.Accordion("🖼️ Scene Images (001.png, 002.png...)", open=True):
                        t2_folder = gr.Textbox(
                            label="📁 Local Images Folder Path (Optional)",
                            placeholder=r"e.g. D:\Projects\images"
                        )
                        t2_images = gr.File(
                            label="📤 Upload Numbered Image Files",
                            file_count="multiple",
                            file_types=["image"]
                        )

                    with gr.Row():
                        t2_ken_burns = gr.Checkbox(label="🎥 Ken Burns Motion (Pan & Zoom)", value=True)
                        t2_subtitles = gr.Checkbox(label="💬 Universal UI Subtitles", value=True)

                    with gr.Row():
                        t2_preview_btn = gr.Button("🔍 Preview Scene Timestamps", variant="secondary")
                        t2_generate_btn = gr.Button("🚀 Generate Synced Video", variant="primary")

                with gr.Column(scale=5):
                    gr.Markdown("### 🎞️ Output & Timing Inspector")
                    t2_video_out = gr.Video(label="🎬 Synchronized 1080p Video", interactive=False)
                    t2_status = gr.Markdown("Status: *Ready.*")

                    with gr.Accordion("📊 Detected Scene Timestamps & Matching", open=True):
                        t2_timing_md = gr.Markdown("Click **'Preview Scene Timestamps'** to inspect audio alignment.")

            # Tab 2 Events
            t2_preview_btn.click(
                fn=preview_alignment,
                inputs=[t2_audio, t2_script, t2_images, t2_folder, gr.State("base"), gr.State("Auto-Detect")],
                outputs=[t2_timing_md, gr.State()]
            )

            t2_generate_btn.click(
                fn=build_full_video_existing_audio,
                inputs=[
                    t2_audio,
                    t2_script,
                    t2_images,
                    t2_folder,
                    t2_subtitles,
                    t2_ken_burns,
                    gr.State("base"),
                    gr.State("Auto-Detect")
                ],
                outputs=[t2_video_out, t2_status]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
