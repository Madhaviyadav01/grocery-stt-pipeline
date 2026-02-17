import os
from .preprocess import PreprocessPipeline

RAW_FOLDER = "data/Audio_samples"
OUTPUT_FOLDER = "artifacts/audio_processed"

def run_phase_A(audio_path: str = None) -> str:
    pipeline = PreprocessPipeline(output_folder=OUTPUT_FOLDER)
    
    # Single file mode
    if audio_path:
        print(f"[Phase A] Processing single file: {audio_path}")
        return pipeline.process_file(audio_path)

    # Batch mode
    print("[Phase A] Batch mode started")
    input_files = []
    for file in os.listdir(RAW_FOLDER):
        if file.endswith((".wav", ".mp3", ".m4a")):
            input_files.append(os.path.join(RAW_FOLDER, file))
    
    print(f"Found {len(input_files)} audio files.")
    
    # Process all files
    results = pipeline.process_batch(input_files, OUTPUT_FOLDER)
    
    print("\nBatch Processing Summary:")
    for file_path, status in results:
        print(f"{file_path} → {'Success' if status else 'Failed'}")
    
    return OUTPUT_FOLDER  # Return folder in batch mode

if __name__ == "__main__":
    run_phase_A()
