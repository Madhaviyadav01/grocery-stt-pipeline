"""
Test script for Phase B transcription.
Tests the modular Phase B implementation.
"""

import os
import sys
from pipeline.phase_B_transcription.run_phase_B import run_phase_B


def main():
    input_file = "artifacts/audio_processed/cleaned_rec_1.wav"
    
    if not os.path.exists(input_file):
        print(f"Test file not found: {input_file}")
        print("Run Phase A first to generate cleaned audio.")
        sys.exit(1)
    
    try:
        # Run Phase B
        transcript_text, output_path = run_phase_B(input_file)
        
        # Display results
        print("\n" + "=" * 60)
        print("TRANSCRIPT:")
        print("=" * 60)
        print(transcript_text)
        print("=" * 60)
        
        # Verify output
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n✅ Test PASSED!")
            print(f"   Transcript saved: {output_path}")
            print(f"   File size: {file_size} bytes")
            print(f"   Text length: {len(transcript_text)} characters")
        else:
            print("\n❌ Test FAILED: Transcript file was not created")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
