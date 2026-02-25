"""
Run Phase E - Evaluation
"""

import json
import os
from datetime import datetime, timezone

from pipeline.phase_E_evaluation.load_inputs import load_inputs
from pipeline.phase_E_evaluation.metrics import calculate_all_metrics
from pipeline.phase_D_fuzzy_canonical_mapping.master_loader import load_master_data


def print_report(metrics: dict):
    print("\n" + "=" * 60)
    print("PHASE E — EVALUATION METRICS")
    print("=" * 60)

    # --- Confusion Matrix --------------------------------------------------
    print("\n[ Confusion Matrix ]")
    print(f"  TP (correct SKU predicted) : {metrics.get('tp', 'N/A')}")
    print(f"  FP (wrong/phantom predicted): {metrics.get('fp', 'N/A')}")
    print(f"    └─ FP Mismatch            : {metrics.get('fp_mismatch', 'N/A')}")
    print(f"    └─ FP Phantom             : {metrics.get('fp_phantom', 'N/A')}")
    print(f"  FN (missed GT items)        : {metrics.get('fn', 'N/A')}")
    print(f"  TN (correctly not predicted): {metrics.get('tn', 'N/A')}")

    # --- Classification Metrics -------------------------------------------
    print("\n[ Classification Metrics ]")
    print(f"  Accuracy  : {metrics.get('accuracy',  'N/A'):.4f}"
          f"  ({metrics.get('accuracy', 0) * 100:.2f}%)")
    print(f"  Precision : {metrics.get('precision', 'N/A'):.4f}")
    print(f"  Recall    : {metrics.get('recall',    'N/A'):.4f}")
    print(f"  F1-Score  : {metrics.get('f1_score',  'N/A'):.4f}")

    # --- Transcription Metrics --------------------------------------------
    print("\n[ Transcription Metrics ]")
    wer = metrics.get("mean_wer")
    cer = metrics.get("mean_cer")
    print(f"  Mean WER  : {wer:.4f}" if wer is not None else "  Mean WER  : N/A")
    print(f"  Mean CER  : {cer:.4f}" if cer is not None else "  Mean CER  : N/A")

    # --- Quantity Error ---------------------------------------------------
    print("\n[ Quantity Prediction ]")
    mae = metrics.get("quantity_mae")
    print(f"  MAE       : {mae:.4f}" if mae is not None else "  MAE       : N/A")
    qty_acc = metrics.get("quantity_match_accuracy")
    print(f"  Exact-Match Accuracy : {qty_acc:.2f}%"
          if qty_acc is not None else "  Exact-Match Accuracy : N/A")

    # --- Additional Metrics -----------------------------------------------
    print("\n[ Additional Metrics ]")
    print(f"  Exact SKU Match Accuracy : {metrics.get('exact_match_accuracy', 'N/A'):.2f}%")
    print(f"  Family Match Accuracy    : {metrics.get('family_match_accuracy', 'N/A'):.2f}%")
    print(f"  Top-3 Accuracy           : {metrics.get('top_3_accuracy', 'N/A'):.2f}%")
    print(f"  Mean Confidence Score    : {metrics.get('mean_confidence_score', 'N/A'):.4f}")
    print(f"  Confidence-Weighted Acc  : {metrics.get('confidence_weighted_accuracy', 'N/A'):.4f}")

    # --- Dataset Info ------------------------------------------------------
    print("\n[ Dataset Info ]")
    print(f"  Ground Truth Items  : {metrics.get('num_ground_truth', 'N/A')}")
    print(f"  Predicted Items     : {metrics.get('num_predictions', 'N/A')}")
    print(f"  Evaluated (matched) : {metrics.get('num_evaluated_samples', 'N/A')}")
    print("=" * 60)


def run_phase_E(
    predictions_path: str = "artifacts/phase_D_output.csv",
    ground_truth_path: str = "data/ground_truth (2).xlsx",
    master_path: str = "data/cleaned_master.csv",
    transcriptions_dir: str = "artifacts/transcriptions",
    output_path: str = "artifacts/phase_E_metrics.json"
) -> str:
    """
    Run Phase E evaluation and generate a comprehensive report.
    """

    print("=" * 60)
    print("PHASE E: EVALUATION REPORT")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load Master Data
    # -----------------------------------------------------------------------
    master_records, _ = load_master_data(master_path)

    # -----------------------------------------------------------------------
    # Load Inputs
    # -----------------------------------------------------------------------
    predictions_df, ground_truth_df, merged_df = load_inputs(
        predictions_path,
        ground_truth_path
    )

    # -----------------------------------------------------------------------
    # Calculate All Metrics
    # -----------------------------------------------------------------------
    metrics = calculate_all_metrics(
        merged_df=merged_df,
        predictions_df=predictions_df,
        ground_truth_df=ground_truth_df,
        master_records=master_records,
        transcriptions_dir=transcriptions_dir,
    )

    # -----------------------------------------------------------------------
    # Add Metadata
    # -----------------------------------------------------------------------
    metrics["evaluation_timestamp"] = datetime.now(timezone.utc).isoformat()

    # -----------------------------------------------------------------------
    # Print Report
    # -----------------------------------------------------------------------
    print_report(metrics)

    # -----------------------------------------------------------------------
    # Generate Error Analysis CSV
    # -----------------------------------------------------------------------
    print("\n[Phase E] Generating detailed error analysis...")
    import pandas as pd

    analysis_records = []
    matched_pred_indices = set()
    matched_gt_indices = set()

    for _, row in merged_df.iterrows():
        rid = row["record_id"]
        idx = row["item_index"]
        matched_pred_indices.add((rid, idx))
        matched_gt_indices.add((rid, idx))

        pred_sku = str(row.get("matched_sku", "")).upper()
        gt_sku   = str(row.get("correct_sku_code", "")).upper()
        status   = "TRUE_POSITIVE" if pred_sku == gt_sku else "FALSE_POSITIVE_MISMATCH"

        analysis_records.append({
            "record_id":         rid,
            "item_index":        idx,
            "status":            status,
            "transcript_segment":      row.get("transcript", ""),
            "predicted_canonical":     row.get("predicted_canonical_name", ""),
            "predicted_sku":           pred_sku,
            "ground_truth_sku":        gt_sku,
            "predicted_qty":           row.get("item_quantity", ""),
            "ground_truth_qty":        row.get("correct_quantity", ""),
            "confidence":              row.get("confidence_score", 0),
        })

    # Phantom FPs
    pred_keys = set(zip(predictions_df["record_id"], predictions_df["item_index"]))
    for _, row in predictions_df.iterrows():
        if (row["record_id"], row["item_index"]) not in matched_pred_indices:
            analysis_records.append({
                "record_id":         row["record_id"],
                "item_index":        row["item_index"],
                "status":            "FALSE_POSITIVE_PHANTOM",
                "transcript_segment":      row.get("transcript", ""),
                "predicted_canonical":     row.get("predicted_canonical_name", ""),
                "predicted_sku":           str(row.get("matched_sku", "")).upper(),
                "ground_truth_sku":        "N/A",
                "predicted_qty":           row.get("item_quantity", ""),
                "ground_truth_qty":        "N/A",
                "confidence":              row.get("confidence_score", 0),
            })

    # FNs
    for _, row in ground_truth_df.iterrows():
        if (row["record_id"], row["item_index"]) not in matched_gt_indices:
            analysis_records.append({
                "record_id":         row["record_id"],
                "item_index":        row["item_index"],
                "status":            "FALSE_NEGATIVE_MISSED",
                "transcript_segment":      "N/A",
                "predicted_canonical":     "N/A",
                "predicted_sku":           "N/A",
                "ground_truth_sku":        str(row.get("correct_sku_code", "")).upper(),
                "predicted_qty":           "N/A",
                "ground_truth_qty":        row.get("correct_quantity", ""),
                "confidence":              0,
            })

    if analysis_records:
        error_df = pd.DataFrame(analysis_records)
        error_df = error_df.sort_values(by=["record_id", "item_index"])
        error_path = output_path.replace("phase_E_metrics.json", "phase_E_error_analysis.csv")
        error_df.to_csv(error_path, index=False)
        print(f"[Phase E] Error analysis saved to: {error_path}")

    # -----------------------------------------------------------------------
    # Save JSON Report  (only the requested metrics)
    # -----------------------------------------------------------------------
    _JSON_KEYS = [
        "accuracy", "precision", "recall", "f1_score",
        "quantity_mae", "mean_wer", "mean_cer",
        "exact_match_accuracy", "family_match_accuracy",
        "quantity_match_accuracy", "top_3_accuracy",
        "mean_confidence_score",
        "evaluation_timestamp",
    ]
    json_metrics = {k: metrics[k] for k in _JSON_KEYS if k in metrics}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(json_metrics, f, indent=2)

    print(f"\n[Phase E] Metrics saved to: {output_path}")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    run_phase_E()
