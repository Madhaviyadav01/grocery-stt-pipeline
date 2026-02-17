"""
Batch Pipeline Runner
Runs complete pipeline (Phases A-D) on multiple audio recordings.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import pipeline phases
from pipeline.phase_B_transcription.run_phase_B import run_phase_B
from pipeline.phase_C_structured_boundary_extraction.run_phase_C import run_phase_C
from pipeline.phase_D_fuzzy_canonical_mapping.order_resolver import resolve_order
from pipeline.phase_D_fuzzy_canonical_mapping.sku_loader import (
    load_sku_dataset, 
    get_unique_items, 
    get_product_mapping,
    get_product_skus
)


def run_full_pipeline_on_recording(
    audio_path: str,
    record_id: int,
    product_list: List[str],
    sku_mapping: Dict[str, str],
    product_skus: Dict[str, List[Dict]],
    threshold: int = 70
) -> List[Dict[str, Any]]:
    """
    Run complete pipeline on a single audio recording.
    
    Args:
        audio_path: Path to the .m4a audio file
        record_id: Recording ID number
        product_list: List of product names for matching
        sku_mapping: SKU mapping dict
        product_skus: All SKUs with size info
        threshold: Fuzzy matching threshold
        
    Returns:
        List of resolved orders for this recording
    """
    print(f"\n{'='*70}")
    print(f"Processing Recording {record_id}: {Path(audio_path).name}")
    print(f"{'='*70}")
    
    try:
        # Phase A: Audio Preprocessing (convert .m4a to .wav using ffmpeg)
        print(f"[Phase A] Preprocessing audio...")
        processed_path = f"artifacts/audio_processed/cleaned_rec_{record_id}.wav"
        os.makedirs("artifacts/audio_processed", exist_ok=True)
        
        # Use ffmpeg to convert m4a to wav
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1",
            processed_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARNING] FFmpeg failed, skipping recording {record_id}")
            return []
        
        # Phase B: Transcription
        print(f"[Phase B] Transcribing...")
        os.makedirs("artifacts/transcriptions", exist_ok=True)
        transcript_path = run_phase_B(processed_path)
        
        # Read transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        print(f"[Phase B] Transcript: {transcript[:100]}...")
        
        # Phase C: Structured Boundary Extraction
        print(f"[Phase C] Extracting structured items...")
        structured_items_path = run_phase_C(transcript_path)
        
        # Read structured items
        with open(structured_items_path, 'r', encoding='utf-8') as f:
            structured_items = json.load(f)
        print(f"[Phase C] Found {len(structured_items)} items")
        
        # Phase D: Order Resolution
        print(f"[Phase D] Resolving orders...")
        results = []
        for idx, item in enumerate(structured_items, 1):
            result = resolve_order(
                item, 
                product_list, 
                sku_mapping, 
                product_skus,
                threshold=threshold
            )
            results.append(result)
            print(f"  [{idx}] {item[:40]:40s} → SKU: {result.get('sku_code', 'None')}")
        
        print(f"[Complete] Processed {len(results)} items from recording {record_id}")
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to process recording {record_id}: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_batch_pipeline(
    num_recordings: int = 20,
    audio_dir: str = "data/Audio_samples",
    dataset_path: str = "data/GVoiceAI_data_with_units.xlsx",
    output_file: str = "artifacts/order_resolution/resolved_orders_20.json",
    threshold: int = 70
) -> List[Dict[str, Any]]:
    """
    Run pipeline on first N recordings.
    
    Args:
        num_recordings: Number of recordings to process
        audio_dir: Directory containing audio files
        dataset_path: SKU dataset path
        output_file: Output JSON file path
        threshold: Fuzzy matching threshold
        
    Returns:
        List of all resolved orders
    """
    print(f"\n{'='*70}")
    print(f"BATCH PIPELINE EXECUTION - {num_recordings} Recordings")
    print(f"{'='*70}\n")
    
    # Load SKU dataset (once for all recordings)
    print("[Setup] Loading SKU dataset...")
    df = load_sku_dataset(dataset_path)
    product_list = get_unique_items(df)
    sku_mapping = get_product_mapping(df)
    product_skus = get_product_skus(df)
    print(f"[Setup] Loaded {len(product_list)} products\n")
    
    # Collect all audio files
    audio_files = sorted([
        f for f in Path(audio_dir).glob("rec_*.m4a")
    ], key=lambda x: int(x.stem.split('_')[1]))
    
    # Process first N recordings
    all_results = []
    
    for i in range(min(num_recordings, len(audio_files))):
        audio_path = str(audio_files[i])
        record_id = i + 1
        
        results = run_full_pipeline_on_recording(
            audio_path,
            record_id,
            product_list,
            sku_mapping,
            product_skus,
            threshold
        )
        
        all_results.extend(results)
    
    # Save results
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total recordings processed: {min(num_recordings, len(audio_files))}")
    print(f"Total items resolved: {len(all_results)}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch pipeline")
    parser.add_argument('--num', type=int, default=20, help="Number of recordings to process")
    parser.add_argument('--threshold', type=int, default=70, help="Fuzzy matching threshold")
    parser.add_argument('--output', type=str, default="artifacts/order_resolution/resolved_orders_20.json")
    
    args = parser.parse_args()
    
    run_batch_pipeline(
        num_recordings=args.num,
        output_file=args.output,
        threshold=args.threshold
    )
