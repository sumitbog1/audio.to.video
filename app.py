"""Gradio Web Studio for Audio-to-Video Sync Generator."""

import os
import time
import gradio as gr
from pipeline.aligner import parse_scene_script, align_scenes_to_audio, get_audio_duration
from pipeline.composer import assemble_video, find_matching_image

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def preview_alignment(audio_file, script_text, images_files, folder_path, model_size, language_choice):
    """Parses script and aligns with MP3 audio, showing detected scene timestamps."""
    if not audio_file:
        return "❌ Please upload an MP3/WAV audio file first.", None

    if not script_text or not script_text.strip():
        return "❌ Please paste your scene narration script.", None

    audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file
    scenes = parse_scene_script(script_text)
    if not scenes:
        return "❌ Could not parse any numbered scenes (e.g., 001:, 002:).", None

    # Collect images source
    img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
    if not img_src and images_files:
        img_src = [f.name if hasattr(f, "name") else f for f in images_files]

    total_dur = get_audio_duration(audio_path)
    lang = None if language_choice == "Auto-Detect" else language_choice.lower()
    aligned = align_scenes_to_audio(scenes, audio_path, model_size=model_size, language=lang)

    # Build markdown summary
    md = [
        f"### 🎵 Master Audio Duration: `{total_dur:.2f}s` | Total Scenes: `{len(aligned)}`\n",
        "| Scene ID | Start Time | End Time | Duration | Matching Image | Spoken Text Preview |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for s in aligned:
        matched_img = find_matching_image(s["id"], img_src)
        img_status = f"✅ `{os.path.basename(matched_img)}`" if matched_img else "⚠️ *Missing (uses placeholder)*"
        preview_text = s['text'][:45] + ("..." if len(s['text']) > 45 else "")
        md.append(f"| **{s['id']}** | `{s['start']:.2f}s` | `{s['end']:.2f}s` | `{s['duration']:.2f}s` | {img_status} | {preview_text} |")

    return "\n".join(md), aligned


def build_full_video(
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
    """Generates the full master 1080p video synchronized to the MP3 audio."""
    if not audio_file:
        raise gr.Error("Please upload an MP3/WAV audio file.")

    if not script_text or not script_text.strip():
        raise gr.Error("Please paste your scene narration script.")

    audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file

    progress(0.05, desc="Parsing script...")
    scenes = parse_scene_script(script_text)
    if not scenes:
        raise gr.Error("No valid numbered scenes found in script.")

    # Image source
    img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
    if not img_src and images_files:
        img_src = [f.name if hasattr(f, "name") else f for f in images_files]

    if not img_src:
        raise gr.Error("Please upload numbered images or provide an images folder path.")

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


custom_css = """
.gradio-container { max-width: 1200px !important; margin: auto; }
.header-box { text-align: center; margin-bottom: 20px; }
.preview-box { background: #131722; padding: 14px; border-radius: 8px; border: 1px solid #2a324b; }
"""

with gr.Blocks(title="Audio.to.Video Sync Studio", css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # 🎬 Audio-to-Video Sync Studio
        ### Turn Master MP3 Narration + Numbered Images (`001.png`, `002.png`...) into a Seamless 1080p Video
        """,
        elem_classes=["header-box"]
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 1️⃣ Audio & Script Input")
            audio_input = gr.Audio(label="🎵 Master MP3 Audio File", type="filepath")

            script_input = gr.Textbox(
                label="📝 Scene Narrations (Numbered)",
                lines=9,
                placeholder="001: In the quiet dawn of human curiosity, questions began to stir...\n002: Philosophers looked up at the stars and wondered about our purpose...\n003: Today, we continue that timeless journey of exploration...",
                value="001: Most people do not truly seek freedom.\n002: They seek comfort and certainty.\n003: But he who faces the abyss of his own mind finds a power that no circumstance can ever strip away."
            )

            with gr.Accordion("🖼️ 2️⃣ Images Input (Upload files OR enter Folder Path)", open=True):
                folder_input = gr.Textbox(
                    label="📁 Local Images Directory Path (Optional)",
                    placeholder=r"e.g. D:\Projects\images_folder"
                )
                images_input = gr.File(
                    label="📤 Upload Numbered Image Files (001.png, 002.png...)",
                    file_count="multiple",
                    file_types=["image"]
                )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    whisper_model = gr.Dropdown(
                        label="Whisper Speech Model",
                        choices=["tiny", "base", "small", "medium"],
                        value="base"
                    )
                    language_select = gr.Dropdown(
                        label="Audio Language",
                        choices=["Auto-Detect", "en", "hi", "es", "fr", "de"],
                        value="Auto-Detect"
                    )

            with gr.Row():
                ken_burns_toggle = gr.Checkbox(label="🎥 Ken Burns Motion (Pan & Zoom)", value=True)
                subtitles_toggle = gr.Checkbox(label="💬 Universal UI Subtitles", value=True)

            with gr.Row():
                preview_btn = gr.Button("🔍 Preview Scene Timestamps", variant="secondary")
                generate_btn = gr.Button("🚀 Generate Synced Video", variant="primary")

        with gr.Column(scale=5):
            gr.Markdown("### 3️⃣ Video Output & Timing Inspector")
            video_output = gr.Video(label="🎞️ Generated Synchronized Video", interactive=False)
            status_text = gr.Markdown("Status: *Ready to build.*")

            with gr.Accordion("📊 Detected Scene Timestamps & Image Matching", open=True):
                alignment_preview = gr.Markdown("Click **'Preview Scene Timestamps'** to inspect audio alignment.")

    # Event Handlers
    preview_btn.click(
        fn=preview_alignment,
        inputs=[audio_input, script_input, images_input, folder_input, whisper_model, language_select],
        outputs=[alignment_preview, gr.State()]
    )

    generate_btn.click(
        fn=build_full_video,
        inputs=[
            audio_input,
            script_input,
            images_input,
            folder_input,
            subtitles_toggle,
            ken_burns_toggle,
            whisper_model,
            language_select
        ],
        outputs=[video_output, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
