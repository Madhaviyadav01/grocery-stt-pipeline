"""
Test script for Phase A audio preprocessing.
Tests the preprocessing pipeline with rec_1.m4a.
"""

from pipeline.Audio_preprocessing.preprocess import preprocess_audio
import os

if __name__ == "__main__":
    # Test with rec_1.m4a
    input_file = "data/Audio_samples/rec_1.m4a"
    
    if not os.path.exists(input_file):
        print(f"❌ Test file not found: {input_file}")
        exit(1)
    
    try:
        output_path = preprocess_audio(input_file)
        
        # Verify output exists
        if os.path.exists(output_path):
            print(f"\n✅ Test PASSED: Output file created at {output_path}")
            
            # Check file size
            file_size = os.path.getsize(output_path)
            print(f"   File size: {file_size / 1024:.2f} KB")
        else:
            print(f"\n❌ Test FAILED: Output file not found at {output_path}")
            
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
