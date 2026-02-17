"""
Phase B Runner - Speech-to-Text Transcription
Main orchestration for the transcription pipeline.
"""

from .transcription_engine import TranscriptionEngine
from .output_writer import save_transcript


def run_phase_B(
    audio_path: str,
    model_name: str = "base",
    language: str = "en"
) -> str:
    """
    Run Phase B: Transcribe audio to text and save the transcript.
    
    Args:
        audio_path (str): Path to the cleaned audio file (WAV recommended).
        model_name (str): Whisper model size (tiny, base, small, medium, large).
                         Default: "base".
        language (str): Language code. Default: "en" for English.
    
    Returns:
        tuple: output_path (str) - Path to the saved transcript file
    
    Example:
        >>> from pipeline.phase_B_transcription.run_phase_B import run_phase_B
        >>> text, path = run_phase_B("artifacts/audio_processed/cleaned_rec_1.wav")
        >>> print(text)
    """
    print(f"\n{'='*60}")
    print("Starting Phase B: Speech-to-Text Transcription")
    print(f"{'='*60}\n")
    
    # Step 1: Initialize transcription engine
    engine = TranscriptionEngine(model_name=model_name)
    
    # Step 2: Transcribe audio
    transcript_text = engine.transcribe(audio_path, language=language)
    
    # Step 3: Save transcript
    output_path = save_transcript(transcript_text, audio_path)
    
    print(f"\n{'='*60}")
    print("Phase B Complete!")
    print(f"{'='*60}\n")
    
    return output_path


if __name__ == "__main__":
    import os
    import csv
    
    AUDIO_FOLDER = "artifacts/audio_processed"
    EVAL_FOLDER = "artifacts/evaluation"
    OUTPUT_CSV = os.path.join(EVAL_FOLDER, "predictions.csv")
    
    os.makedirs(EVAL_FOLDER, exist_ok=True)
    
    print("\n🚀 Starting Batch Transcription...\n")
    
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["record_id", "file_name", "predicted_text"])
        
        for idx, file_name in enumerate(os.listdir(AUDIO_FOLDER), start=1):
            if file_name.endswith(".wav"):
                audio_path = os.path.join(AUDIO_FOLDER, file_name)
                
                print(f"\nProcessing: {audio_path}")
                
                output_path = run_phase_B(
                    audio_path=audio_path,
                    model_name="base",
                    language="en"
                )
                
                # Read back for CSV writing if needed, or just skip writing text to CSV here for now to keep it simple
                # or read the file:
                with open(output_path, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
                
                writer.writerow([idx, file_name, transcript_text])
    
    print("\n✅ Batch Transcription Completed!")
    print(f"📁 Predictions saved at: {OUTPUT_CSV}\n")

