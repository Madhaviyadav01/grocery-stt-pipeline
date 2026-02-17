"""
Silence Trimmer Module - Phase A
Removes long silent sections using pydub.silence.split_on_silence.
"""

from pydub import AudioSegment
from pydub.silence import split_on_silence


def trim_silence(audio: AudioSegment) -> AudioSegment:
    """
    Trim silence from audio by splitting on silence and rejoining chunks.
    
    Args:
        audio (AudioSegment): Input audio.
        
    Returns:
        AudioSegment: Audio with silence trimmed.
    """
    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=500,  # minimum length of silence (ms)
        silence_thresh=audio.dBFS - 14,  # silence threshold
        keep_silence=200  # keep 200ms of silence at edges
    )
    
    # If no chunks found, return original audio
    if not chunks:
        print("⚠ No silence detected, returning original audio")
        return audio
    
    # Combine chunks
    trimmed_audio = AudioSegment.empty()
    for chunk in chunks:
        trimmed_audio += chunk
    
    original_duration = len(audio) / 1000
    trimmed_duration = len(trimmed_audio) / 1000
    print(f"✓ Trimmed silence (original: {original_duration:.2f}s, trimmed: {trimmed_duration:.2f}s)")
    
    return trimmed_audio
