"""
Run Phase C
Orchestrates the structured boundary extraction process.
"""

import os
import json
from typing import List, Tuple
from .snapshot_loader import load_snapshot
from .extractor import extract_item_boundaries


def run_phase_C(
    transcript_path: str,
    output_folder: str = "artifacts/structured_boundary_extraction"
) -> str:
    """
    Run Phase C: Extract structured items from a transcript file.

    Returns:
        str: output file path
    """
    print(f"[Phase C] Loading transcript: {transcript_path}")
    transcript_text = load_snapshot(transcript_path)

    print("[Phase C] Extracting item boundaries...")
    items = extract_item_boundaries(transcript_text)
    print(f"[Phase C] Extracted {len(items)} items.")

    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(transcript_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_structured.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"[Phase C] Output saved: {output_path}")

    return output_path


def main():
    import sys
    if len(sys.argv) > 1:
        run_phase_C(sys.argv[1])
    else:
        print("Usage: python run_phase_C.py <path_to_transcript_file>")


def main():
    import sys

    transcripts_dir = "artifacts/transcriptions"
    output_folder = "artifacts/structured_boundary_extraction"

    # If specific file provided → process single file
    if len(sys.argv) > 1:
        run_phase_C(sys.argv[1], output_folder)
        return

    # Otherwise → batch mode
    if not os.path.exists(transcripts_dir):
        print(f"[Phase C] Transcripts folder not found: {transcripts_dir}")
        return

    transcript_files = [
        f for f in os.listdir(transcripts_dir)
        if f.endswith(".txt") or f.endswith(".csv")
    ]

    if not transcript_files:
        print("[Phase C] No transcript files found!")
        return

    print(f"[Phase C] Found {len(transcript_files)} transcript files")

    for file in sorted(transcript_files):
        transcript_path = os.path.join(transcripts_dir, file)
        print(f"\nProcessing: {file}")
        run_phase_C(transcript_path, output_folder)

    print("\n✅ Phase C Batch Processing Complete!")


if __name__ == "__main__":
    main()


