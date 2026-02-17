"""
Audio Normalizer Module - Phase A
Normalizes audio volume using pydub.effects.normalize.
"""

from pydub import AudioSegment
from pydub.effects import normalize


def normalize_audio(audio: AudioSegment) -> AudioSegment:
    """
    Normalize the audio volume.
    
    Args:
        audio (AudioSegment): Input audio.
        
    Returns:
        AudioSegment: Normalized audio.
    """
    normalized = normalize(audio)
    print(f"✓ Normalized audio (original dBFS: {audio.dBFS:.2f}, normalized dBFS: {normalized.dBFS:.2f})")
    return normalized
