"""Gradio Web Studio for Audio-to-Video Sync Studio."""

import os
import time
import html
import traceback
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
# Modern UI HTML Components (Large Table, Loader, Error Banners)
# ----------------------------------------------------------------------

def render_timing_table_html(aligned_scenes, audio_dur: float, images_source) -> str:
    """Renders a spacious, high-visibility, modern dark-mode table for scene timestamps & image matching."""
    if not aligned_scenes:
        return """
        <div class="empty-table-box">
            <div style="font-size: 32px; margin-bottom: 8px;">📊</div>
            <div style="font-weight: 600; font-size: 16px; color: #94a3b8;">No scene timestamps detected yet.</div>
            <div style="font-size: 14px; color: #64748b; margin-top: 4px;">Upload your audio and click <strong>'🔍 Preview Scene Timestamps'</strong> to inspect alignment.</div>
        </div>
        """

    total_scenes = len(aligned_scenes)
    matched_count = 0
    rows_html = []

    for s in aligned_scenes:
        matched_img = find_matching_image(s["id"], images_source)
        if matched_img:
            matched_count += 1
            img_badge = f'<span class="badge badge-img-ok">✅ {html.escape(os.path.basename(matched_img))}</span>'
        else:
            img_badge = '<span class="badge badge-img-missing">⚠️ Missing Image (Cinematic Slate)</span>'

        esc_text = html.escape(s["text"])
        rows_html.append(f"""
        <tr>
            <td style="text-align: center;"><span class="badge badge-scene">Scene {html.escape(s['id'])}</span></td>
            <td style="text-align: center;"><span class="badge badge-time">{s['start']:.2f}s</span></td>
            <td style="text-align: center;"><span class="badge badge-time">{s['end']:.2f}s</span></td>
            <td style="text-align: center;"><span class="badge badge-dur">{s['duration']:.2f}s</span></td>
            <td>{img_badge}</td>
            <td class="scene-narration-cell">{esc_text}</td>
        </tr>
        """)

    all_matched = matched_count == total_scenes
    match_summary_class = "summary-pill-ok" if all_matched else "summary-pill-warn"
    match_summary_text = f"✅ Images Matched: {matched_count}/{total_scenes}" if all_matched else f"⚠️ Images Matched: {matched_count}/{total_scenes} (Missing will use slates)"

    return f"""
    <div class="large-inspector-card">
        <div class="inspector-summary-bar">
            <div class="summary-pill">🎵 Master Audio Duration: <strong>{audio_dur:.2f}s</strong></div>
            <div class="summary-pill">📑 Total Scenes: <strong>{total_scenes}</strong></div>
            <div class="summary-pill {match_summary_class}">{match_summary_text}</div>
            <div class="summary-pill status-ready">✨ Contiguous 0-Gap Alignment</div>
        </div>
        <div class="table-scroll-wrapper">
            <table class="inspector-table">
                <thead>
                    <tr>
                        <th style="width: 110px; text-align: center;">Scene</th>
                        <th style="width: 100px; text-align: center;">Start Time</th>
                        <th style="width: 100px; text-align: center;">End Time</th>
                        <th style="width: 100px; text-align: center;">Duration</th>
                        <th style="width: 260px;">Matching Image File</th>
                        <th>Spoken Narration Text</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
    </div>
    """


def render_loader_html(title: str, subtitle: str) -> str:
    """Renders an animated glowing spinner card for running processes."""
    return f"""
    <div class="loader-banner">
        <div class="spinner-ring"></div>
        <div class="loader-text-wrap">
            <div class="loader-title">⚡ {html.escape(title)}</div>
            <div class="loader-subtitle">{html.escape(subtitle)}</div>
        </div>
    </div>
    """


def render_error_html(title: str, detail: str, hint: str = "") -> str:
    """Renders a prominent, high-visibility red error banner."""
    hint_html = f'<div class="error-hint">💡 <strong>Suggestion:</strong> {html.escape(hint)}</div>' if hint else ""
    return f"""
    <div class="error-banner">
        <div class="error-header">
            <span class="error-icon">❌</span>
            <span class="error-title">{html.escape(title)}</span>
        </div>
        <div class="error-detail">{html.escape(detail)}</div>
        {hint_html}
    </div>
    """


def render_success_html(title: str, file_path: str, duration: float = 0.0) -> str:
    """Renders a prominent green success banner."""
    dur_str = f" | Duration: `{duration:.2f}s`" if duration > 0 else ""
    return f"""
    <div class="success-banner">
        <div class="success-header">
            <span class="success-icon">🎉</span>
            <span class="success-title">{html.escape(title)}</span>
        </div>
        <div class="success-detail">File: <code>{html.escape(os.path.basename(file_path))}</code>{dur_str}</div>
        <div class="success-path">Saved to: <code>{html.escape(file_path)}</code></div>
    </div>
    """


# ----------------------------------------------------------------------
# Tab 1: Script to MP3 Converter Logic
# ----------------------------------------------------------------------

def convert_script_to_mp3(script_text, voice, progress=gr.Progress()):
    """Converts simple script text lines into a continuous Master MP3 voiceover with robust error handling."""
    if not script_text or not script_text.strip():
        return None, render_error_html(
            title="Script Text Missing",
            detail="The narration script box is empty. Please enter your script lines before converting.",
            hint="Type or paste simple sentences or paragraphs in the text area on the left."
        )

    try:
        progress(0.15, desc="Preparing voice synthesis...")
        timestamp = int(time.time())
        out_mp3 = os.path.join(OUTPUT_DIR, f"master_voice_{timestamp}.mp3")

        progress(0.40, desc=f"Synthesizing speech with '{voice}'...")
        synthesize_text_to_mp3(
            text=script_text,
            voice=voice,
            speed_percent=0,
            output_path=out_mp3
        )

        total_dur = get_audio_duration(out_mp3)
        progress(1.0, desc="Master MP3 ready!")
        return out_mp3, render_success_html(
            title="Master Voice MP3 Generated Successfully!",
            file_path=out_mp3,
            duration=total_dur
        )

    except Exception as e:
        print("[ERROR in convert_script_to_mp3]:", traceback.format_exc())
        return None, render_error_html(
            title="Voice Synthesis Failed",
            detail=str(e),
            hint="Make sure your internet connection is active for neural speech synthesis, or try selecting another voice."
        )


# ----------------------------------------------------------------------
# Tab 2: Audio-to-Video Sync Studio Logic
# ----------------------------------------------------------------------

def preview_alignment_flow(audio_file, script_text, images_files, folder_path, model_size, language_choice, progress=gr.Progress()):
    """Parses script and aligns with uploaded MP3 audio, displaying the large inspector table."""
    if not audio_file:
        error_card = render_error_html(
            title="Audio File Missing",
            detail="Please upload a Master MP3 or WAV audio file in Step 1 before previewing timestamps.",
            hint="Click on the 'Upload Master MP3/WAV Audio' box to select your audio file."
        )
        return error_card, render_error_html("Audio Missing", "Please upload audio file first.")

    if not script_text or not script_text.strip():
        error_card = render_error_html(
            title="Script Text Missing",
            detail="Please enter your scene narration script lines in Step 1.",
            hint="Each line represents one scene narration (e.g. Line 1 for image 001, Line 2 for image 002)."
        )
        return error_card, render_error_html("Script Empty", "Please enter narration text.")

    audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file

    try:
        progress(0.10, desc="Parsing script lines...")
        scenes = parse_scene_script(script_text)
        if not scenes:
            return render_error_html(
                title="Could Not Parse Scenes",
                detail="No readable lines were found in your script.",
                hint="Make sure you have text lines separated by Enter/Return."
            ), render_error_html("Parse Failed", "No valid scenes found.")

        # Resolve image source
        img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
        if not img_src and images_files:
            img_src = [f.name if hasattr(f, "name") else f for f in images_files]

        progress(0.30, desc=f"Loading Faster-Whisper ({model_size}) on GPU/CPU...")
        total_dur = get_audio_duration(audio_path)
        lang = None if language_choice == "Auto-Detect" else language_choice.lower()

        progress(0.60, desc="Aligning speech word timestamps...")
        aligned = align_scenes_to_audio(scenes, audio_path, model_size=model_size, language=lang)

        progress(1.0, desc="Alignment complete!")
        table_html = render_timing_table_html(aligned, total_dur, img_src)
        status_html = f"""
        <div class="info-status-card">
            ✅ <strong>Alignment Preview Ready:</strong> Successfully aligned <strong>{len(aligned)} scenes</strong> with audio duration <strong>{total_dur:.2f}s</strong>. Review the table below, then click <strong>'🚀 Generate Synced Video'</strong>!
        </div>
        """
        return table_html, status_html

    except Exception as e:
        print("[ERROR in preview_alignment]:", traceback.format_exc())
        return render_error_html(
            title="Speech Alignment Error",
            detail=str(e),
            hint="Verify that the uploaded file is a valid audio file (MP3/WAV) and contains audible speech."
        ), render_error_html("Alignment Failed", str(e))


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
    """Generates the full master 1080p video with live progress loader and comprehensive error handling."""
    if not audio_file:
        error_html = render_error_html(
            title="Master Audio Missing",
            detail="Cannot generate video without an audio file. Please upload an MP3 or WAV audio track.",
            hint="Upload your audio file in the top left box."
        )
        return None, error_html, render_timing_table_html([], 0.0, [])

    if not script_text or not script_text.strip():
        error_html = render_error_html(
            title="Scene Script Missing",
            detail="Cannot generate video without scene narration script lines.",
            hint="Enter your narration lines in the script box."
        )
        return None, error_html, render_timing_table_html([], 0.0, [])

    audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file

    try:
        progress(0.05, desc="Parsing script lines...")
        scenes = parse_scene_script(script_text)
        if not scenes:
            raise ValueError("No valid scene narration lines found.")

        # Images source
        img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
        if not img_src and images_files:
            img_src = [f.name if hasattr(f, "name") else f for f in images_files]

        if not img_src:
            gr.Warning("No image files uploaded or folder not found. Cinematic placeholder slates will be used.")

        progress(0.20, desc=f"Aligning speech with Faster-Whisper ({model_size})...")
        lang = None if language_choice == "Auto-Detect" else language_choice.lower()
        aligned = align_scenes_to_audio(scenes, audio_path, model_size=model_size, language=lang)
        total_dur = get_audio_duration(audio_path)

        timestamp = int(time.time())
        output_video_path = os.path.join(OUTPUT_DIR, f"synced_video_{timestamp}.mp4")

        def p_cb(fraction, desc):
            progress(0.35 + fraction * 0.60, desc=desc)

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

        progress(1.0, desc="Video complete!")
        table_html = render_timing_table_html(aligned, total_dur, img_src)
        success_card = render_success_html(
            title="1080p Synchronized Video Generated Successfully!",
            file_path=output_video_path,
            duration=total_dur
        )

        return output_video_path, success_card, table_html

    except Exception as e:
        print("[ERROR in build_full_video_existing_audio]:", traceback.format_exc())
        error_card = render_error_html(
            title="Video Generation Failed",
            detail=str(e),
            hint="Check that your images are readable image formats (PNG, JPG, WEBP) and the audio file is not corrupt."
        )
        return None, error_card, render_timing_table_html([], 0.0, [])


# ----------------------------------------------------------------------
# Modern Studio Layout & CSS
# ----------------------------------------------------------------------

custom_css = """
.gradio-container { max-width: 1400px !important; margin: auto; }
.header-box { text-align: center; margin-bottom: 24px; }

/* Large Inspector Card & Table */
.large-inspector-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    margin-top: 10px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}
.inspector-summary-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
    align-items: center;
}
.summary-pill {
    background: #1e293b;
    color: #e2e8f0;
    padding: 8px 16px;
    border-radius: 9999px;
    font-size: 14px;
    border: 1px solid #334155;
}
.summary-pill-ok {
    background: #064e3b;
    color: #34d399;
    border-color: #059669;
    font-weight: 600;
}
.summary-pill-warn {
    background: #78350f;
    color: #fcd34d;
    border-color: #d97706;
    font-weight: 600;
}
.status-ready {
    background: #1e1b4b;
    color: #a5b4fc;
    border-color: #4338ca;
}

.table-scroll-wrapper {
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid #1e293b;
}
.inspector-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15px;
    background: #090d16;
}
.inspector-table th {
    background: #131d31;
    color: #93c5fd;
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 14px 18px;
    border-bottom: 2px solid #1e293b;
    text-align: left;
}
.inspector-table td {
    padding: 14px 18px;
    border-bottom: 1px solid #172237;
    color: #f1f5f9;
    vertical-align: middle;
}
.inspector-table tr:hover td {
    background: #111a2e;
}
.scene-narration-cell {
    font-size: 14.5px;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
}
.badge-scene {
    background: #1d4ed8;
    color: #ffffff;
}
.badge-time {
    background: #1e293b;
    color: #94a3b8;
    font-family: monospace;
    font-size: 13.5px;
}
.badge-dur {
    background: #047857;
    color: #ffffff;
    font-family: monospace;
    font-size: 13.5px;
}
.badge-img-ok {
    background: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}
.badge-img-missing {
    background: #78350f;
    color: #fcd34d;
    border: 1px solid #b45309;
}

/* Animated Loader Banner */
.loader-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    background: #0f172a;
    border: 2px solid #3b82f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 12px;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.25);
    animation: pulse-border 2s infinite ease-in-out;
}
.spinner-ring {
    width: 32px;
    height: 32px;
    border: 4px solid rgba(59, 130, 246, 0.2);
    border-top: 4px solid #3b82f6;
    border-radius: 50%;
    animation: spin-loader 0.8s linear infinite;
    flex-shrink: 0;
}
.loader-text-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.loader-title {
    font-weight: 700;
    color: #60a5fa;
    font-size: 15px;
}
.loader-subtitle {
    color: #94a3b8;
    font-size: 13px;
}

/* Error Banner */
.error-banner {
    background: #450a0a;
    border: 2px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 12px;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
}
.error-header {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #fca5a5;
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 6px;
}
.error-detail {
    color: #fee2e2;
    font-size: 14px;
    margin-left: 28px;
    line-height: 1.4;
}
.error-hint {
    background: rgba(0, 0, 0, 0.25);
    padding: 8px 12px;
    border-radius: 6px;
    margin-top: 10px;
    margin-left: 28px;
    color: #fde047;
    font-size: 13px;
}

/* Success Banner */
.success-banner {
    background: #064e3b;
    border: 2px solid #10b981;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 12px;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25);
}
.success-header {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #6ee7b7;
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 6px;
}
.success-detail {
    color: #d1fae5;
    font-size: 14px;
    margin-left: 28px;
}
.success-path {
    color: #a7f3d0;
    font-size: 13px;
    margin-left: 28px;
    margin-top: 4px;
}

/* Info Status Card */
.info-status-card {
    background: #172554;
    border: 1px solid #1d4ed8;
    color: #bfdbfe;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
    margin-top: 10px;
}

.empty-table-box {
    text-align: center;
    padding: 40px 20px;
    background: #090d16;
    border-radius: 8px;
    border: 1px dashed #1e293b;
}

@keyframes spin-loader {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes pulse-border {
    0%, 100% { border-color: #3b82f6; box-shadow: 0 0 15px rgba(59, 130, 246, 0.2); }
    50% { border-color: #60a5fa; box-shadow: 0 0 25px rgba(96, 165, 250, 0.4); }
}
"""

sample_script = """Most people do not truly seek freedom.
They seek comfort, security, and certainty in an unpredictable world.
But he who dares to face the silence of his own mind unlocks an eternal power."""

with gr.Blocks(title="Audio.to.Video Studio", css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # 🎬 Audio-to-Video Studio
        ### Convert Scripts to AI Voiceover MP3 & Synchronize Master Audio with Numbered Images (`001.png`...)
        """,
        elem_classes=["header-box"]
    )

    with gr.Tabs():
        # --------------------------------------------------------------
        # TAB 1: SCRIPT TO MP3 CONVERTER
        # --------------------------------------------------------------
        with gr.TabItem("🎙️ 1. Script to MP3 Converter"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 📝 Narration Script Input (Simple Text Lines)")
                    t1_script = gr.Textbox(
                        label="Narration Script",
                        lines=10,
                        placeholder="Paste your story or narration as simple text lines...\nEach sentence or thought on its own line...\nNo numbering or splitting needed...",
                        value=sample_script
                    )

                    t1_voice = gr.Dropdown(
                        label="Narrator Voice",
                        choices=get_voice_choices(),
                        value=DEFAULT_VOICE
                    )

                    t1_convert_btn = gr.Button("🎙️ Convert Script to MP3", variant="primary")

                with gr.Column(scale=5):
                    gr.Markdown("### 🎵 Output MP3 Audio Player")
                    t1_audio_out = gr.Audio(label="Master Voice Narration (MP3)", interactive=False)
                    t1_status_html = gr.HTML(
                        """<div class="info-status-card">💡 Status: <strong>Ready.</strong> Paste your script lines and click <strong>'🎙️ Convert Script to MP3'</strong>.</div>"""
                    )

            # Tab 1 Event
            t1_convert_btn.click(
                fn=convert_script_to_mp3,
                inputs=[t1_script, t1_voice],
                outputs=[t1_audio_out, t1_status_html]
            )

        # --------------------------------------------------------------
        # TAB 2: AUDIO-TO-VIDEO SYNC STUDIO
        # --------------------------------------------------------------
        with gr.TabItem("🎬 2. Audio-to-Video Sync Studio (Audio + Images -> 1080p Video)"):
            with gr.Row():
                # LEFT COLUMN: Inputs & Controls
                with gr.Column(scale=5):
                    gr.Markdown("### 🎵 1. Master Audio & Scene Script")
                    t2_audio = gr.Audio(label="Upload Master MP3/WAV Audio Track", type="filepath")
                    t2_script = gr.Textbox(
                        label="Scene Script (Simple Text Lines - 1 line per scene image)",
                        lines=8,
                        placeholder="Line 1 narration...\nLine 2 narration...\nLine 3 narration...",
                        value=sample_script
                    )

                    with gr.Accordion("🖼️ 2. Scene Images (*001*, (001), 001.png...)", open=True):
                        t2_folder = gr.Textbox(
                            label="📁 Local Images Folder Path (Optional)",
                            placeholder=r"e.g. D:\Projects\images"
                        )
                        t2_images = gr.File(
                            label="📤 Upload Numbered Image Files",
                            file_count="multiple",
                            file_types=["image"]
                        )

                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        with gr.Row():
                            t2_model = gr.Dropdown(label="Whisper Speech Model", choices=["tiny", "base", "small", "medium"], value="base")
                            t2_lang = gr.Dropdown(label="Audio Language", choices=["Auto-Detect", "en", "hi", "es", "fr", "de"], value="Auto-Detect")

                    with gr.Row():
                        t2_ken_burns = gr.Checkbox(label="🎥 Ken Burns Motion (Pan & Zoom)", value=True)
                        t2_subtitles = gr.Checkbox(label="💬 Universal UI Subtitles", value=True)

                    with gr.Row():
                        t2_preview_btn = gr.Button("🔍 Preview Scene Timestamps", variant="secondary")
                        t2_generate_btn = gr.Button("🚀 Generate Synced Video", variant="primary")

                # RIGHT COLUMN: Video Player & Status
                with gr.Column(scale=5):
                    gr.Markdown("### 🎞️ 3. Synchronized Video Output")
                    t2_video_out = gr.Video(label="🎬 1080p Master Video Player", interactive=False)
                    t2_status_html = gr.HTML(
                        """<div class="info-status-card">Status: <strong>Ready.</strong> Upload audio and click <strong>'Preview'</strong> or <strong>'Generate'</strong>.</div>"""
                    )

            # DEDICATED FULL-WIDTH SECTION FOR LARGE TABLE
            gr.Markdown("---")
            gr.Markdown("### 📊 Detected Scene Timestamps & Image Matching Inspector")
            t2_timing_table_html = gr.HTML(render_timing_table_html([], 0.0, []))

            # Tab 2 Events
            t2_preview_btn.click(
                fn=preview_alignment_flow,
                inputs=[t2_audio, t2_script, t2_images, t2_folder, t2_model, t2_lang],
                outputs=[t2_timing_table_html, t2_status_html]
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
                    t2_model,
                    t2_lang
                ],
                outputs=[t2_video_out, t2_status_html, t2_timing_table_html]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
