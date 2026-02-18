"""
Output Writer Module - Phase B
Handles saving transcripts to text files.
"""

import os


def save_transcript(
    transcript_text: str,
    audio_path: str,
    output_folder: str = "artifacts/transcriptions"
) -> str:
    """
    Save a transcript to a text file.
    
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate output filename based on audio file
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.txt")
    
    # Save transcript
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    print(f"Transcript saved: {output_path}")
    
    return output_path
