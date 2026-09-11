# 🎬 Audio-to-Video Sync Studio (`audio.to.video`)

> Transform a **Master MP3 Audio** file + **Numbered Scene Narrations** (`001`, `002`, `003`...) + **Numbered Images** (`001.png`, `002.png`...) into a fully synchronized, professional 1080p video with cinematic motion.

---

## 🌟 Key Features

1. **Master Audio Speech Alignment (Whisper)**:
   - Uses `faster-whisper` word-level forced-alignment to detect precisely when each scene narration begins and ends.
   - Guaranteed contiguous scene boundaries without gaps, overlaps, or drifts.

2. **Automatic Numbered Image Matching**:
   - Supports images labeled `001.png`, `002.png`, `flow_001_*.png`, `scene_001.jpg`, etc.
   - Upload multiple image files directly in the Web UI, or provide a local directory path.

3. **Cinematic Motion (Ken Burns)**:
   - Dynamic pan and zoom on high-res 16:9 widescreen frames (1920x1080).
   - Smooth interpolation perfectly scaled to each scene's audio duration.

4. **Pristine Audio Quality (Lossless FFmpeg Muxing)**:
   - Retains the exact original Master MP3 audio without re-encoding degradation or sync slips.

---

## 🚀 Quick Start

### 1. Launch Web UI (Port 7861)
Simply double-click `run.bat` or run in terminal:
```bash
# If using the shared venv:
..\text.to.video\venv\Scripts\activate
python app.py
```
Open your browser at **`http://127.0.0.1:7861`**.

---

### 2. Command Line Interface (CLI)
You can also run headless batch jobs using `cli.py`:

```bash
python cli.py \
  --audio "path/to/narration.mp3" \
  --script "path/to/script.txt" \
  --images "path/to/images_folder" \
  --output "outputs/my_video.mp4"
```

#### Script Text Format Example:
```text
001: In the quiet dawn of human curiosity, ancient thinkers looked up at the night sky.
002: They contemplated existence, asking who we are and why we search for meaning.
003: Today, that same eternal flame of wonder drives our modern journey.
```

#### CLI Options:
- `--preview-only`: Preview detected scene timestamps and image matches without rendering.
- `--no-ken-burns`: Use static images without pan & zoom.
- `--model`: Whisper model size (`tiny`, `base`, `small`, `medium`). Default is `base`.

---

## 📁 Project Structure

```
audio.to.video/
├── app.py                  # Interactive Gradio Web Studio (port 7861)
├── cli.py                  # Headless command-line interface
├── run.bat                 # 1-Click Windows batch launcher
├── requirements.txt        # Python package dependencies
├── README.md               # Documentation & usage guide
└── pipeline/
    ├── __init__.py
    ├── aligner.py          # Faster-Whisper word alignment & scene boundary detection
    ├── motion.py           # 16:9 Ken Burns pan & zoom generator
    ├── image_gen.py        # Minimalist editorial line-art generator
    └── composer.py         # Clip concatenation & FFmpeg audio muxer
```
