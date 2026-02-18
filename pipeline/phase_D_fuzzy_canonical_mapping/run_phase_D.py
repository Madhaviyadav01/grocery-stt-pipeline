"""
Run Phase D - Fuzzy Canonical Matching
"""

import os
import json
import time
import pandas as pd

# Importing your optimized Phase D modules
from pipeline.phase_D_fuzzy_canonical_mapping.master_loader import load_master_data
from pipeline.phase_D_fuzzy_canonical_mapping.mapper import map_transcript
from pipeline.phase_D_fuzzy_canonical_mapping.tfidf_retriever import TFIDFRetriever
from pipeline.phase_D_fuzzy_canonical_mapping.embedding_retriever import EmbeddingRetriever

def run_phase_D(
    structured_json_path: str = None,
    master_path: str = "data/cleaned_master.csv",
    transcripts_dir: str = "artifacts/structured_boundary_extraction",
    output_path: str = "artifacts/phase_D_output.csv"
) -> str:

    print("=" * 60)
    print("PHASE D: HYBRID FUZZY CANONICAL MATCHING")
    print("=" * 60)

    start_time = time.time()

    # 1️⃣ Load master data & Known Brands
    # Modified to unpack the brand set needed for penalties
    master_records, all_known_brands = load_master_data(master_path)

    if not master_records:
        print("[Phase D] Error: No master records loaded!")
        return output_path
    
    # 2️⃣ Initialize Dual-Stage Retrievers
    print("[Phase D] Building Hybrid Retrievers (TF-IDF + Embeddings)...")
    # TF-IDF handles exact keyword matching
    tfidf_retriever = TFIDFRetriever(master_records)
    # Embedding retriever handles STT phonetics/semantics
    emb_retriever = EmbeddingRetriever(master_records)
    print("[Phase D] Retrievers ready!")

    # 3️⃣ File handling logic
    transcript_files = []
    if structured_json_path:
        if not os.path.exists(structured_json_path):
             print(f"[Phase D] Error: File not found: {structured_json_path}")
             return output_path
        transcripts_dir = os.path.dirname(structured_json_path)
        transcript_files = [os.path.basename(structured_json_path)]
    else:
        if not os.path.exists(transcripts_dir):
            print(f"[Phase D] Error: Directory not found: {transcripts_dir}")
            return output_path
        transcript_files = [f for f in os.listdir(transcripts_dir) if f.endswith("_structured.json")]

    if not transcript_files:
        print("[Phase D] No structured transcript files found!")
        return output_path

    predictions = []

    # 4️⃣ Process transcripts with Hybrid Logic
    for filename in sorted(transcript_files):
        filepath = os.path.join(transcripts_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                segments = json.load(f)

            print(f"\nProcessing {filename} ({len(segments)} segments)")

            for idx, transcript in enumerate(segments):
                if not isinstance(transcript, str) or not transcript.strip():
                    continue

                # PASSING ALL COMPONENTS: Retrievers + Brand Set
                prediction = map_transcript(
                    transcript,
                    tfidf_retriever,
                    emb_retriever,
                    all_known_brands=all_known_brands
                )


                prediction["source_file"] = filename
                prediction["item_index"] = idx + 1
                predictions.append(prediction)

                if idx == 0:
                    print(f"  Sample: '{transcript[:50]}...'")
                    print(f"  → {prediction['predicted_canonical_name']} "
                          f"(score: {prediction['confidence_score']:.1f})")

        except Exception as e:
            print(f"[Phase D] Error processing {filename}: {e}")

    # 5️⃣ Save and Summarize
    if not predictions:
        print("[Phase D] No predictions generated!")
        return output_path

    df = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    end_time = time.time()
    print("\n" + "=" * 60)
    print("PHASE D SUMMARY")
    print(f"Total Predictions: {len(df)}")
    print(f"Mean Confidence Score: {df['confidence_score'].mean():.2f}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print("=" * 60)
    
    return output_path

if __name__ == "__main__":
    run_phase_D()