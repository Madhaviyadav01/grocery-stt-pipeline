"""
Run Phase E - Evaluation (Production Version)
Generate comprehensive evaluation report.
"""

import json
import os
from datetime import datetime

from pipeline.phase_E_evaluation.load_inputs import load_inputs
from pipeline.phase_E_evaluation.metrics import calculate_all_metrics
from pipeline.phase_D_fuzzy_canonical_mapping.master_loader import load_master_data


def print_report(metrics: dict):
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    print(f"Exact Match Accuracy:        {metrics['exact_match_accuracy']:.2f}%")
    print(f"Top-3 Accuracy:              {metrics['top_3_accuracy']:.2f}%")
    print(f"Precision (Macro):           {metrics['precision']:.4f}")
    print(f"Recall (Macro):              {metrics['recall']:.4f}")
    print(f"F1 Score (Macro):            {metrics['f1_score']:.4f}")
    print(f"Not-Found Accuracy:          {metrics['not_found_accuracy']:.4f}")
    print(f"Confidence-Weighted Acc:     {metrics['confidence_weighted_accuracy']:.4f}")
    print(f"Mean Confidence Score:       {metrics['mean_confidence_score']:.4f}")

    print("=" * 60)


def run_phase_E(
    predictions_path: str = "artifacts/phase_D_output.csv",
    ground_truth_path: str = "data/ground_truth (2).xlsx",
    master_path: str = "data/cleaned_master.csv",
    output_path: str = "artifacts/phase_E_metrics.json"
) -> str:
    """
    Run Phase E evaluation and generate report.
    """

    print("=" * 60)
    print("PHASE E: EVALUATION REPORT")
    print("=" * 60)

    # ----------------------------
    # Load Master Data (FIXED)
    # ----------------------------
    master_records, _ = load_master_data(master_path)

    # ----------------------------
    # Load Inputs
    # ----------------------------
    predictions_df, ground_truth_df, merged_df = load_inputs(
        predictions_path,
        ground_truth_path
    )

    # ----------------------------
    # Calculate Metrics
    # ----------------------------
    metrics = calculate_all_metrics(
        merged_df,
        master_records
    )

    # ----------------------------
    # Add Metadata
    # ----------------------------
    metrics["evaluation_timestamp"] = datetime.utcnow().isoformat()
    metrics["num_predictions"] = len(predictions_df)
    metrics["num_ground_truth"] = len(ground_truth_df)
    metrics["num_evaluated_samples"] = len(merged_df)

    # ----------------------------
    # Print Report
    # ----------------------------
    print_report(metrics)

    # ----------------------------
    # Save JSON Report
    # ----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Phase E] Metrics saved to: {output_path}")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    run_phase_E()
