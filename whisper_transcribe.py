import whisper
import subprocess
import os
import re
import time

# Paths (assume installed)
YT_DLP = "yt-dlp"
FFMPEG = "ffmpeg"

SUPPORTED_FORMATS = (".mp3", ".wav", ".m4a", ".mp4")

def transcribe_audio(input_source: str) -> str:
    """
    Transcribe audio from a YouTube/Vimeo URL or a local audio file.

    Args:
        input_source: str - URL or local file path

    Returns:
        str - Transcribed text
    """

    def is_video_url(url: str) -> bool:
        """Basic check for video URLs"""
        pattern = r"(youtube\.com|youtu\.be|vimeo\.com)"
        return re.search(pattern, url) is not None

    def download_audio(url: str) -> str:
        """Download audio from video URL"""
        output_file = f"audio_{int(time.time())}.mp3"
        print("📥 Downloading audio...")
        cmd = [
            YT_DLP,
            "-x",
            "--audio-format", "mp3",
            "--ffmpeg-location", FFMPEG,
            "-o", output_file,
            url
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Download finished: {output_file}")
        return output_file

    temp_file = None
    try:
        # URL case
        if input_source.startswith("http"):
            if not is_video_url(input_source):
                raise ValueError("❌ Invalid video URL")
            temp_file = download_audio(input_source)
            audio_file = temp_file
        else:
            # Local file case
            if not os.path.exists(input_source):
                raise ValueError(f"❌ File not found: {input_source}")
            if not input_source.endswith(SUPPORTED_FORMATS):
                raise ValueError(f"❌ Unsupported audio format: {input_source}")
            audio_file = input_source

        # Load Whisper model
        print("🤖 Loading Whisper model...")
        model = whisper.load_model("base")

        # Transcribe
        print("🎙️ Transcribing...")
        result = model.transcribe(audio_file)
        print("✅ Transcription complete!")
        return result["text"]

    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"🗑️ Removed temp file: {temp_file}")
