"""
Audio Loader Module - Phase A
Loads audio files using pydub, supporting .wav, .mp3, and .m4a formats.
"""

from pydub import AudioSegment
import os

# Configure FFmpeg path for pydub - add to PATH so pydub can find it
FFMPEG_DIR = r"C:\Users\madha\AppData\Local\Microsoft\WinGet\Links"
if os.path.exists(FFMPEG_DIR) and FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")


def load_audio(file_path: str) -> AudioSegment:
    """
    Load an audio file using pydub.
    
    Args:
        file_path (str): Path to the audio file (.wav, .mp3, or .m4a).
        
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
    
    try:
        # Use generic loader - let pydub auto-detect the format
        audio = AudioSegment.from_file(file_path)
        
        print(f"✓ Loaded audio: {file_path} (duration: {len(audio)/1000:.2f}s)")
        return audio
        
    except Exception as e:
        raise Exception(f"Failed to load audio file {file_path}: {str(e)}")
