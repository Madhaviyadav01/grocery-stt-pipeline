"""
Simple test to verify pydub can load audio with FFmpeg in PATH.
"""

import os

# Add FFmpeg to PATH
FFMPEG_DIR = r"C:\Users\madha\AppData\Local\Microsoft\WinGet\Links"
if os.path.exists(FFMPEG_DIR):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

from pydub import AudioSegment

print("Testing audio loading with FFmpeg in PATH...")
try:
    audio = AudioSegment.from_file("data/Audio_samples/rec_1.m4a")
    print(f"✓ Successfully loaded audio!")
    print(f"  Duration: {len(audio)/1000:.2f}s")
    print(f"  Channels: {audio.channels}")
    print(f"  Sample rate: {audio.frame_rate} Hz")
    print(f"  dBFS: {audio.dBFS:.2f}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
