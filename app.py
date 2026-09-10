"""Gradio Web Studio for Audio-to-Video Sync Studio (Clean, Minimal, Black & White)."""

import os
import time
import gradio as gr

from pipeline.aligner import parse_scene_script, align_scenes_to_audio, get_audio_duration
from pipeline.composer import assemble_video, find_matching_image
from pipeline.tts import (
    synthesize_text_to_mp3,
    get_voice_choices,
    DEFAULT_VOICE
)
from pipeline.image_gen import generate_bulk_images, generate_single_image

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "generated_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Tab 1: Script to MP3 Converter
# ----------------------------------------------------------------------

def convert_script_to_mp3(script_text, voice, progress=gr.Progress(track_tqdm=True)):
    """Converts simple script text lines into a single Master MP3 audio file."""
    if not script_text or not script_text.strip():
        return None, "Status: **Error:** Please enter your narration script text."

    try:
        progress(0.2, desc="Synthesizing voiceover...")
        timestamp = int(time.time())
        out_mp3 = os.path.join(OUTPUT_DIR, f"master_voice_{timestamp}.mp3")

        synthesize_text_to_mp3(
            text=script_text,
            voice=voice,
            speed_percent=0,
            output_path=out_mp3
        )

        total_dur = get_audio_duration(out_mp3)
        progress(1.0, desc="Done!")
        msg = f"Status: **Success!** Master MP3 generated ({total_dur:.2f}s). Saved to: `{out_mp3}`"
        return out_mp3, msg

    except Exception as e:
        return None, f"Status: **Error:** {str(e)}"


# ----------------------------------------------------------------------
# Tab 2: Audio-to-Video Sync Studio
# ----------------------------------------------------------------------

def preview_alignment(audio_file, script_text, images_files, folder_path, model_size, progress=gr.Progress(track_tqdm=True)):
    """Parses script and aligns with uploaded MP3 audio, displaying simple clean grid."""
    if not audio_file:
        return [], "Status: **Error:** Please upload an MP3/WAV audio file first."

    if not script_text or not script_text.strip():
        return [], "Status: **Error:** Please enter your scene narration script."

    audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file

    try:
        progress(0.1, desc="Parsing script lines...")
        scenes = parse_scene_script(script_text)
        if not scenes:
            return [], "Status: **Error:** Could not parse any lines from script."

        img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
        if not img_src and images_files:
            img_src = [f.name if hasattr(f, "name") else f for f in images_files]

        # Default fallback to generated_images if empty
        if not img_src and os.path.isdir(IMAGES_DIR) and len(os.listdir(IMAGES_DIR)) > 0:
            img_src = IMAGES_DIR

        progress(0.3, desc=f"Loading Whisper ({model_size})...")
        total_dur = get_audio_duration(audio_path)

        progress(0.6, desc="Aligning timestamps...")
        aligned = align_scenes_to_audio(scenes, audio_path, model_size=model_size)

        grid_data = []
        matched_count = 0
        for s in aligned:
            matched_img = find_matching_image(s["id"], img_src)
            if matched_img:
                matched_count += 1
                img_name = os.path.basename(matched_img)
            else:
                img_name = "Missing (Uses Placeholder)"

            grid_data.append([
                f"Scene {s['id']}",
                f"{s['start']:.2f}s",
                f"{s['end']:.2f}s",
                f"{s['duration']:.2f}s",
                img_name,
                s["text"]
            ])

        progress(1.0, desc="Preview complete!")
        summary = f"Status: **Alignment Ready.** Audio: `{total_dur:.2f}s` | Scenes: `{len(aligned)}` | Images matched: `{matched_count}/{len(aligned)}`"
        return grid_data, summary

    except Exception as e:
        return [], f"Status: **Error:** {str(e)}"


def build_full_video_existing_audio(
    audio_file,
    script_text,
    images_files,
    folder_path,
    enable_subtitles,
    enable_ken_burns,
    model_size,
    progress=gr.Progress(track_tqdm=True)
):
    """Generates the full master 1080p video synchronized to the audio with clean status updates."""
    if not audio_file:
        return None, "Status: **Error:** Please upload an MP3/WAV audio file.", []

    if not script_text or not script_text.strip():
        return None, "Status: **Error:** Please enter your scene narration script.", []

    audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file

    try:
        progress(0.05, desc="Parsing script...")
        scenes = parse_scene_script(script_text)
        if not scenes:
            return None, "Status: **Error:** No valid scenes found in script.", []

        img_src = folder_path.strip() if folder_path and os.path.isdir(folder_path.strip()) else []
        if not img_src and images_files:
            img_src = [f.name if hasattr(f, "name") else f for f in images_files]

        # Default fallback to generated_images if empty
        if not img_src and os.path.isdir(IMAGES_DIR) and len(os.listdir(IMAGES_DIR)) > 0:
            img_src = IMAGES_DIR

        progress(0.20, desc=f"Aligning with Faster-Whisper ({model_size})...")
        aligned = align_scenes_to_audio(scenes, audio_path, model_size=model_size)
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

        grid_data = []
        for s in aligned:
            matched_img = find_matching_image(s["id"], img_src)
            img_name = os.path.basename(matched_img) if matched_img else "Missing (Placeholder)"
            grid_data.append([
                f"Scene {s['id']}",
                f"{s['start']:.2f}s",
                f"{s['end']:.2f}s",
                f"{s['duration']:.2f}s",
                img_name,
                s["text"]
            ])

        progress(1.0, desc="Complete!")
        status_msg = f"Status: **Success!** Video generated successfully! Saved to: `{output_video_path}`"
        return output_video_path, status_msg, grid_data

    except Exception as e:
        return None, f"Status: **Error:** {str(e)}", []


# ----------------------------------------------------------------------
# Tab 3: Local Image Generator (SD1.5 Dreamshaper on GPU)
# ----------------------------------------------------------------------

def generate_local_images_flow(prompts_text, aspect_ratio, progress=gr.Progress(track_tqdm=True)):
    """Generates images in bulk from simple text lines (1 per line) saving to outputs/generated_images/ as 001.png, 002.png..."""
    if not prompts_text or not prompts_text.strip():
        return [], "Status: **Error:** Please enter at least 1 image prompt."

    try:
        if "Square" in aspect_ratio:
            w, h = 512, 512
        elif "Vertical" in aspect_ratio:
            w, h = 512, 768
        else:
            w, h = 768, 512

        def p_cb(frac, desc):
            progress(frac, desc=desc)

        created_files = generate_bulk_images(
            prompts_text=prompts_text,
            output_dir=IMAGES_DIR,
            width=w,
            height=h,
            num_steps=20,
            progress_callback=p_cb
        )

        count = len(created_files)
        msg = f"Status: **Success!** Generated {count} images (001.png... to {count:03d}.png). Saved to: `{IMAGES_DIR}`"
        return created_files, msg

    except Exception as e:
        return [], f"Status: **Error:** {str(e)}"


# ----------------------------------------------------------------------
# Gradio Studio Layout (Clean, Minimal & Black-and-White)
# ----------------------------------------------------------------------

custom_css = """
.gradio-container { max-width: 1280px !important; margin: auto; }
.header-box { text-align: center; margin-bottom: 20px; }
"""

sample_script = """Most people do not truly seek freedom.
They seek comfort, security, and certainty in an unpredictable world.
But he who dares to face the silence of his own mind unlocks an eternal power."""

sample_prompts = """Stickman sitting on a wooden stool surrounded by clocks and circular cycle arrows, unfinished task
Minimalist diagram of two paths diverging, one easy with traps and one hard leading to mastery
Stick figure focused at a simple desk with a single bright lightbulb and focus arrows"""

with gr.Blocks(title="Audio.to.Video Studio", css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # Audio-to-Video Studio
        ### Convert Scripts to Voiceover MP3, Generate Scene Images & Sync into 1080p Video
        """,
        elem_classes=["header-box"]
    )

    with gr.Tabs():
        # --------------------------------------------------------------
        # TAB 1: SCRIPT TO MP3 CONVERTER
        # --------------------------------------------------------------
        with gr.TabItem("1. Script to MP3 Converter"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### Narration Script Input")
                    t1_script = gr.Textbox(
                        label="Narration Script (Simple Text Lines)",
                        lines=9,
                        placeholder="Paste your script as simple text lines...\nEach line on its own row...",
                        value=sample_script
                    )

                    t1_voice = gr.Dropdown(
                        label="Narrator Voice",
                        choices=get_voice_choices(),
                        value=DEFAULT_VOICE
                    )

                    t1_convert_btn = gr.Button("Convert Script to MP3", variant="primary")

                with gr.Column(scale=5):
                    gr.Markdown("### Master MP3 Audio Output")
                    t1_audio_out = gr.Audio(label="Generated Voice Narration (MP3)", interactive=False)
                    t1_status = gr.Markdown("Status: *Ready. Paste your script lines and click 'Convert Script to MP3'.*")

            # Tab 1 Event
            t1_convert_btn.click(
                fn=convert_script_to_mp3,
                inputs=[t1_script, t1_voice],
                outputs=[t1_audio_out, t1_status]
            )

        # --------------------------------------------------------------
        # TAB 2: AUDIO-TO-VIDEO SYNC STUDIO
        # --------------------------------------------------------------
        with gr.TabItem("2. Audio-to-Video Sync Studio"):
            with gr.Row():
                # LEFT COLUMN
                with gr.Column(scale=5):
                    gr.Markdown("### Audio & Scene Script")
                    t2_audio = gr.Audio(label="Upload Master MP3/WAV Audio", type="filepath")
                    t2_script = gr.Textbox(
                        label="Scene Script (Simple Text Lines - 1 line per scene image)",
                        lines=7,
                        placeholder="Line 1 narration...\nLine 2 narration...\nLine 3 narration...",
                        value=sample_script
                    )

                    with gr.Accordion("Scene Images (*001*, (001), 001.png...)", open=True):
                        t2_folder = gr.Textbox(
                            label="Local Images Folder Path (Optional)",
                            placeholder=r"e.g. outputs\generated_images",
                            value=r"outputs\generated_images"
                        )
                        t2_images = gr.File(
                            label="Upload Numbered Image Files",
                            file_count="multiple",
                            file_types=["image"]
                        )

                    with gr.Accordion("Advanced Settings", open=False):
                        t2_model = gr.Dropdown(label="Whisper Speech Model", choices=["tiny", "base", "small", "medium"], value="base")

                    with gr.Row():
                        t2_ken_burns = gr.Checkbox(label="Ken Burns Motion (Pan & Zoom)", value=True)
                        t2_subtitles = gr.Checkbox(label="Universal Subtitles", value=True)

                    with gr.Row():
                        t2_preview_btn = gr.Button("Preview Scene Timestamps", variant="secondary")
                        t2_generate_btn = gr.Button("Generate Synced Video", variant="primary")

                # RIGHT COLUMN
                with gr.Column(scale=5):
                    gr.Markdown("### Video Output & Status")
                    t2_video_out = gr.Video(label="Synchronized 1080p Video", interactive=False)
                    t2_status = gr.Markdown("Status: *Ready. Upload audio and click 'Preview' or 'Generate'.*")

            # FULL-WIDTH SIMPLE GRID SECTION
            gr.Markdown("---")
            gr.Markdown("### Detected Scene Timestamps & Image Matching")
            t2_grid = gr.Dataframe(
                headers=["Scene ID", "Start Time", "End Time", "Duration", "Matched Image", "Narration Text"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )

            # Tab 2 Events
            t2_preview_btn.click(
                fn=preview_alignment,
                inputs=[t2_audio, t2_script, t2_images, t2_folder, t2_model],
                outputs=[t2_grid, t2_status]
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
                    t2_model
                ],
                outputs=[t2_video_out, t2_status, t2_grid]
            )

        # --------------------------------------------------------------
        # TAB 3: MINIMALIST LINE-ART STUDIO (DAN KOE / ATOMIC HABITS)
        # --------------------------------------------------------------
        with gr.TabItem("3. Minimalist Line-Art Studio"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### Minimalist Editorial Line-Art Prompts")
                    gr.Markdown("*Generates clean black-ink illustrations on warm cream canvas (Dan Koe / Atomic Habits style).*")
                    t3_prompts = gr.Textbox(
                        label="Scene Prompts (1 prompt per line, generates 001.png, 002.png...)",
                        lines=8,
                        placeholder="Line 1: Stickman sitting on a stool surrounded by clocks and cycle arrows\nLine 2: Minimalist diagram of two diverging habit paths\nLine 3: Stick figure focused with a bright lightbulb...",
                        value=sample_prompts
                    )

                    with gr.Row():
                        t3_aspect = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=["16:9 Widescreen (768x512)", "Square (512x512)", "9:16 Vertical (512x768)"],
                            value="16:9 Widescreen (768x512)"
                        )

                    t3_gen_btn = gr.Button("Generate Line-Art Images (001.png, 002.png...)", variant="primary")

                with gr.Column(scale=5):
                    gr.Markdown("### Generated Line-Art Gallery")
                    t3_gallery = gr.Gallery(label="Output Illustrations", columns=3, height="auto")
                    t3_status = gr.Markdown("Status: *Ready. Enter scene prompts and click 'Generate Line-Art Images'.*")
                    gr.Markdown(f"💡 *Generated illustrations are saved into `{IMAGES_DIR}` as `001.png`, `002.png`... and automatically match in Tab 2!*")

            # Tab 3 Event
            t3_gen_btn.click(
                fn=generate_local_images_flow,
                inputs=[t3_prompts, t3_aspect],
                outputs=[t3_gallery, t3_status]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
