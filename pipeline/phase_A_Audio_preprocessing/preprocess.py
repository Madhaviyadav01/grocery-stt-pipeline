"""
Audio Preprocessing Pipeline - Phase A
Orchestrates the complete preprocessing workflow.
"""

import os

# Configure FFmpeg path BEFORE importing pydub modules
FFMPEG_DIR = r"C:\Users\madha\AppData\Local\Microsoft\WinGet\Links"
if os.path.exists(FFMPEG_DIR) and FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

from pydub import AudioSegment
from .audio_loader import load_audio
from .audio_normalizer import normalize_audio
from .silence_trimmer import trim_silence


def preprocess_audio(input_path: str, output_folder: str = "artifacts/audio_processed") -> str:
    """
    Preprocess an audio file: Load → Normalize → Trim Silence → Save.
    
    """
    print(f"\n{'='*60}")
    print(f"🎵 Starting audio preprocessing: {input_path}")
    print(f"{'='*60}\n")
    
    # 1. Load audio
    print("Step 1: Loading audio...")
    audio = load_audio(input_path)
    
    # 2. Normalize
    print("\nStep 2: Normalizing audio...")
    audio = normalize_audio(audio)
    
    # 3. Trim silence
    print("\nStep 3: Trimming silence...")
    audio = trim_silence(audio)
    
    # 4. Save
    print("\nStep 4: Saving processed audio...")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate output filename
    input_filename = os.path.basename(input_path)
    input_name, _ = os.path.splitext(input_filename)
    output_filename = f"cleaned_{input_name}.wav"
    output_path = os.path.join(output_folder, output_filename)
    
    # Export as WAV
    audio.export(output_path, format="wav")
    
    print(f"✓ Saved to: {output_path}")
    print(f"\n{'='*60}")
    print(f"✅ Preprocessing complete!")
    print(f"{'='*60}\n")
    
    return output_path


class PreprocessPipeline:
    """
    Legacy class for backward compatibility.
    """
    def __init__(self, output_folder: str = "artifacts/audio_processed"):
        self.output_folder = output_folder
        
    def process_file(self, input_path: str, output_path: str = None):
        """
        Process a single audio file.
        
        Args:
            input_path (str): Path to input audio file.
            output_path (str, optional): Specific output path (overrides default naming).
            
        Returns:
            str: Path to the processed audio file.
        """
        if output_path:
            # If specific output path is given, use it
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            audio = load_audio(input_path)
            audio = normalize_audio(audio)
            audio = trim_silence(audio)
            audio.export(output_path, format="wav")
            return output_path
        else:
            return preprocess_audio(input_path, self.output_folder)
    
    def process_batch(self, input_files, output_dir):
        """
        Process a batch of files.
        """
        results = []
        for file_path in input_files:
            try:
                output_path = preprocess_audio(file_path, output_dir)
                results.append((file_path, True))
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                results.append((file_path, False))
        return results
