"""
Transcription Engine Module - Phase B
Handles Whisper model loading and audio transcription.
"""

import os
import whisper
import torch


def configure_ffmpeg():
    """Ensure FFmpeg is available in PATH."""
    ffmpeg_dir = r"C:\Users\madha\AppData\Local\Microsoft\WinGet\Links"
    if os.path.exists(ffmpeg_dir) and ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


class TranscriptionEngine:
    """
    Whisper-based Speech-to-Text transcription engine.
    
    """
    
    def __init__(self, model_name: str = "base"):
        """Initialize the Whisper transcription engine."""
        configure_ffmpeg()
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            print("Whisper model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe an audio file to text.
        
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing: {os.path.basename(audio_path)}")
        
        result = self.model.transcribe(audio_path, language=language, verbose=False)
        
        print("Transcription complete.")
        
        return result["text"].strip()
