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
    
    Args:
        transcript_text (str): The transcribed text to save.
        audio_path (str): Path to the original audio file (used for naming).
        output_folder (str): Folder to save the transcript.
                            Default: "artifacts/transcriptions".
    
    Returns:
        str: Path to the saved transcript file.
    
    Example:
        >>> save_transcript("Hello world", "cleaned_rec_1.wav")
        'artifacts/transcriptions/cleaned_rec_1.txt'
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
