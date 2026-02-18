"""
Batch Pipeline Runner
Runs complete pipeline (Phases A-E) on multiple audio recordings.
"""

import os
import sys
from typing import List, Dict, Any

# Import pipeline phases
from pipeline.phase_A_Audio_preprocessing.run_phase_A import run_phase_A
from pipeline.phase_B_transcription.run_phase_B import run_phase_B
from pipeline.phase_C_structured_boundary_extraction.run_phase_C import run_phase_C
from pipeline.phase_D_fuzzy_canonical_mapping.run_phase_D import run_phase_D
from pipeline.phase_E_evaluation.run_phase_E import run_phase_E

def run_batch_pipeline(
    num_recordings: int = 50,
    audio_dir: str = "data/Audio_samples"
) -> None:
    """
    Run full pipeline on N recordings.
    
    """
    print(f"\n{'='*70}")
    print(f"BATCH PIPELINE EXECUTION - {num_recordings} Recordings")
    print(f"{'='*70}\n")
    
    # Collect audio files
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found: {audio_dir}")
        return

    all_files = [f for f in os.listdir(audio_dir) if f.endswith(('.m4a', '.wav', '.mp3'))]
    
    # Sort files by recording number (rec_1, rec_2, etc.)
    try:
        audio_files = sorted(all_files, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    except ValueError:
        audio_files = sorted(all_files)

    # Process first N recordings
    recordings_to_process = audio_files[:num_recordings]
    print(f"Found {len(audio_files)} files. Processing first {len(recordings_to_process)}.\n")

    # ---------------------------------------------------------
    # Phases A, B, C (Per File)
    # ---------------------------------------------------------
    for idx, filename in enumerate(recordings_to_process, 1):
        print(f"\n--- Processing {idx}/{len(recordings_to_process)}: {filename} ---")
        audio_path = os.path.join(audio_dir, filename)
        
        try:
            # Phase A: Preprocessing
            cleaned_audio_path = run_phase_A(audio_path)
            
            # Phase B: Transcription
            transcript_path = run_phase_B(cleaned_audio_path)
            
            # Phase C: Structure Extraction
            run_phase_C(transcript_path) # Saves JSON to default artifact folder
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------
    # Phase D: Mapping (Batch)
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Running Phase D (Batch Mapping)...")
    print(f"{'='*70}")
    
    # Output of Phase C is in artifacts/structured_boundary_extraction
    # run_phase_D defaults to reading from there and saving to artifacts/phase_D_output.csv
    predictions_csv_path = run_phase_D()
    
    # ---------------------------------------------------------
    # Phase E: Evaluation
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Running Phase E (Evaluation)...")
    print(f"{'='*70}")
    
    # run_phase_E defaults to reading artifacts/phase_D_output.csv
    run_phase_E()
    
    print(f"\n{'='*70}")
    print(f"BATCH PIPELINE COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch pipeline (A-E)")
    parser.add_argument('--num', type=int, default=50, help="Number of recordings to process")
    
    args = parser.parse_args()
    
    run_batch_pipeline(
        num_recordings=args.num
    )
