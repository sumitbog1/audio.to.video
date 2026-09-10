"""Gradio Web Studio for Audio-to-Video Sync Studio."""

import os
import time
import gradio as gr
from pipeline.aligner import parse_scene_script, align_scenes_to_audio, get_audio_duration
from pipeline.composer import assemble_video, find_matching_image
from pipeline.tts import (
    synthesize_scenes_to_master_mp3,
    synthesize_text_to_mp3,
    get_voice_choices,
    DEFAULT_VOICE
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Tab 1: Script to MP3 Voiceover Only
# ----------------------------------------------------------------------

def convert_script_to_mp3(script_text, voice, speed_percent, progress=gr.Progress()):
    """Converts simple script text lines into a continuous Master MP3 voiceover."""
    if not script_text or not script_text.strip():
        raise gr.Error("Please enter your narration script text.")

    progress(0.2, desc="Synthesizing narration voiceover...")
    timestamp = int(time.time())
    out_mp3 = os.path.join(OUTPUT_DIR, f"master_voice_{timestamp}.mp3")

    synthesize_text_to_mp3(
        text=script_text,
        voice=voice,
        speed_percent=speed_percent,
        output_path=out_mp3
    )

    total_dur = get_audio_duration(out_mp3)
    status_msg = f"""
### ✅ Master Voice MP3 Ready!
- **Duration**: `{total_dur:.2f}s`
- **File**: `{os.path.basename(out_mp3)}`
- **Path**: `{out_mp3}`

💡 *You can listen to or download the MP3 above, or go to **Tab 2** to sync it with your images into a 1080p video!*
"""
    return out_mp3, status_msg


# ----------------------------------------------------------------------
# Tab 2: Audio-to-Video Sync Studio (Audio + Images -> Video)
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
.gradio-container { max-width: 1200px !important; margin: auto; }
.header-box { text-align: center; margin-bottom: 20px; }
"""

sample_script = """Most people do not truly seek freedom.
They seek comfort, security, and certainty in an unpredictable world.
But he who dares to face the silence of his own mind unlocks an eternal power."""

with gr.Blocks(title="Audio.to.Video Studio", css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # 🎬 Audio-to-Video Studio
        ### Convert Scripts to AI Voiceover MP3 & Sync Master Audio with Numbered Images (`001.png`...)
        """,
        elem_classes=["header-box"]
    )

    with gr.Tabs():
        # TAB 1: SCRIPT TO MP3 CONVERTER
        with gr.TabItem("🎙️ 1. Script to MP3 Converter"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 📝 Narration Script Input (Simple Text Lines)")
                    t1_script = gr.Textbox(
                        label="Narration Script",
                        lines=10,
                        placeholder="Paste your story or narration as simple text lines...\nEach line on its own row...\nNo numbering or splitting needed...",
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

                    t1_convert_btn = gr.Button("🎙️ Convert Script to MP3", variant="primary")

                with gr.Column(scale=5):
                    gr.Markdown("### 🎵 Output MP3 Audio Player")
                    t1_audio_out = gr.Audio(label="Master Voice Narration (MP3)", interactive=False)
                    t1_status = gr.Markdown("Status: *Ready. Paste your script lines and click 'Convert Script to MP3'.*")

            # Tab 1 Event
            t1_convert_btn.click(
                fn=convert_script_to_mp3,
                inputs=[t1_script, t1_voice, t1_speed],
                outputs=[t1_audio_out, t1_status]
            )

        # TAB 2: AUDIO-TO-VIDEO SYNC STUDIO
        with gr.TabItem("🎬 2. Audio-to-Video Sync Studio (Audio + Images -> 1080p Video)"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 🎵 Master Audio & Script")
                    t2_audio = gr.Audio(label="Upload Master MP3/WAV Audio", type="filepath")
                    t2_script = gr.Textbox(
                        label="Scene Script (Simple Text Lines - 1 line per scene image)",
                        lines=8,
                        placeholder="Line 1 narration...\nLine 2 narration...\nLine 3 narration...",
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
                    gr.Markdown("### 🎞️ Output Video & Inspector")
                    t2_video_out = gr.Video(label="🎬 Synchronized 1080p Video", interactive=False)
                    t2_status = gr.Markdown("Status: *Ready.*")

                    with gr.Accordion("📊 Detected Scene Timestamps & Image Matching", open=True):
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
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
