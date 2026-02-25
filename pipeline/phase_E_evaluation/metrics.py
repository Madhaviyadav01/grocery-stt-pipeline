"""
Metrics - Phase E

Computes comprehensive evaluation metrics:
  - Confusion Matrix: TP, FP, FN, TN
  - Classification: Accuracy, Precision, Recall, F1-score
  - Quantity: MAE
  - Transcription: Mean WER, Mean CER (per recording)
  - Plus original accuracy / confidence / top-k metrics
"""

import os
import re
import ast
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_upper(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


# ---------------------------------------------------------------------------
# WER / CER helpers  (no extra dependencies – pure Python edit-distance)
# ---------------------------------------------------------------------------

def _edit_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    """Levenshtein edit distance between two token lists."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m]


def _wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    ref = list(reference.lower())
    hyp = list(hypothesis.lower())
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _build_reference_text(gt_rows: pd.DataFrame) -> str:
    """
    Build a reference string for a single recording from ground-truth rows.
    Concatenates Brand + Item Name + Item Quantity + correct_unit per item.
    """
    parts = []
    for _, row in gt_rows.iterrows():
        brand = str(row.get("brand", row.get("Brand", ""))).strip()
        item_name = str(row.get("item_name", row.get("Item Name", ""))).strip()
        item_qty = str(row.get("item_quantity", row.get("Item Quantity", ""))).strip()
        unit = str(row.get("correct_unit", "")).strip()
        token = " ".join(filter(None, [brand, item_name, item_qty, unit]))
        parts.append(token)
    return " ".join(parts).lower()


def compute_wer_cer(
    ground_truth_df: pd.DataFrame,
    transcriptions_dir: str = "artifacts/transcriptions"
) -> Tuple[float, float]:
    """
    Per-recording WER and CER.

    Hypothesis : cleaned STT transcript file (cleaned_rec_{N}.txt)
    Reference  : ground-truth items for that recording, joined into a string.

    Returns (mean_wer, mean_cer).
    """
    # Normalise column names for flexible access
    gt = ground_truth_df.copy()
    gt.columns = gt.columns.str.lower().str.strip().str.replace(" ", "_")

    wer_scores, cer_scores = [], []

    for rec_id, group in gt.groupby("record_id"):
        # Derive the numeric recording number from record_id
        # e.g. "cleaned_rec_1" or 1 → "cleaned_rec_1.txt"
        rid = str(rec_id)
        num_match = re.search(r"(\d+)$", rid)
        if not num_match:
            continue
        num = num_match.group(1)
        txt_path = os.path.join(transcriptions_dir, f"cleaned_rec_{num}.txt")

        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
            hypothesis = fh.read().strip()

        reference = _build_reference_text(group)

        if not reference:
            continue

        wer_scores.append(_wer(reference, hypothesis))
        cer_scores.append(_cer(reference, hypothesis))

    mean_wer = float(np.mean(wer_scores)) if wer_scores else float("nan")
    mean_cer = float(np.mean(cer_scores)) if cer_scores else float("nan")
    return mean_wer, mean_cer


# ---------------------------------------------------------------------------
# Confusion Matrix  (binary, system-level)
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> Dict[str, int]:
    """
    Binary confusion matrix over all GT items.

    TP : SKU predicted and matches GT  (from merged_df where pred == gt)
    FP : SKU predicted but wrong (merged mismatch) + phantom preds (pred not in GT)
    FN : GT item has no matched prediction
    TN : GT item correctly has no prediction (both GT SKU empty and pred empty)
    """
    y_true = safe_upper(merged_df["correct_sku_code"])
    y_pred = safe_upper(merged_df["matched_sku"])

    # Matched pairs
    tp = int((y_true == y_pred).sum())
    fp_mismatch = int((y_true != y_pred).sum())

    # Matched GT keys
    matched_gt_keys = set(zip(merged_df["record_id"], merged_df["item_index"]))
    matched_pred_keys = set(zip(merged_df["record_id"], merged_df["item_index"]))

    # Phantom FPs: predictions that never joined with any GT item
    pred_keys = set(zip(predictions_df["record_id"], predictions_df["item_index"]))
    fp_phantom = len(pred_keys - matched_pred_keys)

    # FN: GT items that had no prediction matched
    gt_keys = set(zip(ground_truth_df["record_id"], ground_truth_df["item_index"]))
    fn = len(gt_keys - matched_gt_keys)

    # TN: GT items with empty SKU that were also not matched (correctly absent)
    gt_norm = ground_truth_df.copy()
    gt_norm["correct_sku_code"] = safe_upper(gt_norm["correct_sku_code"])
    tn_gt_empty = gt_norm[gt_norm["correct_sku_code"] == ""]
    gt_tn_keys = set(zip(tn_gt_empty["record_id"], tn_gt_empty["item_index"]))
    # TN = GT-empty items that were NOT matched (i.e., we correctly didn't predict)
    tn = len(gt_tn_keys - matched_gt_keys)

    fp = fp_mismatch + fp_phantom

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_mismatch": fp_mismatch,
        "fp_phantom": fp_phantom,
    }


# ---------------------------------------------------------------------------
# Classification Metrics (derived from CM counts)
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    tp: int, fp: int, fn: int, tn: int
) -> Dict[str, float]:
    total = tp + fp + fn + tn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
    }


# ---------------------------------------------------------------------------
# Quantity MAE
# ---------------------------------------------------------------------------

def compute_quantity_mae(merged_df: pd.DataFrame) -> float:
    """Mean Absolute Error of item quantity prediction vs ground truth."""
    pred_qty = pd.to_numeric(merged_df.get("item_quantity",  pd.Series(dtype=float)), errors="coerce")
    true_qty = pd.to_numeric(merged_df.get("correct_quantity", pd.Series(dtype=float)), errors="coerce")

    valid_mask = pred_qty.notna() & true_qty.notna()
    if valid_mask.sum() == 0:
        return float("nan")

    mae = float((pred_qty[valid_mask] - true_qty[valid_mask]).abs().mean())
    return round(mae, 4)


# ---------------------------------------------------------------------------
# Existing / legacy metrics (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def _compute_legacy_metrics(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """Exact match accuracy, family match, top-3, and confidence metrics."""
    metrics: Dict[str, Any] = {}
    total = len(merged_df)
    if total == 0:
        return metrics

    y_true = safe_upper(merged_df["correct_sku_code"])
    y_pred = safe_upper(merged_df["matched_sku"])
    correct_mask = y_true == y_pred

    # Exact match
    metrics["exact_match_accuracy"] = round(correct_mask.mean() * 100, 4)

    # Family match (brand + item name in predicted canonical)
    def is_family_match(row):
        pred  = str(row.get("predicted_canonical_name", "")).lower()
        brand = str(row.get("brand", "")).lower()
        item  = str(row.get("item_name", "")).lower()
        return bool(brand and brand in pred) and bool(item and item in pred)

    try:
        family_matches = merged_df.apply(is_family_match, axis=1)
        metrics["family_match_accuracy"] = round(family_matches.mean() * 100, 4)
    except Exception:
        metrics["family_match_accuracy"] = None

    # Quantity (exact match %)
    quantity_match = (
        merged_df["item_quantity"].fillna(-1) ==
        merged_df["correct_quantity"].fillna(-2)
    )
    metrics["quantity_match_accuracy"] = round(quantity_match.mean() * 100, 4)

    # Top-3 accuracy
    correct_top3 = 0
    for _, row in merged_df.iterrows():
        gt_sku = str(row.get("correct_sku_code", "")).upper()
        try:
            candidates = row.get("top_5_candidates", "[]")
            if isinstance(candidates, str):
                candidates = ast.literal_eval(candidates)
            top3 = [str(c.get("sku", "")).upper() for c in candidates[:3]]
            if gt_sku in top3:
                correct_top3 += 1
        except Exception:
            continue
    metrics["top_3_accuracy"] = round((correct_top3 / total) * 100, 4)

    # Confidence metrics
    conf = merged_df["confidence_score"].fillna(0)
    metrics["mean_confidence_score"] = round(float(conf.mean()), 4)
    metrics["confidence_weighted_accuracy"] = round(
        float((conf * correct_mask.astype(int)).mean()), 4
    )

    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calculate_all_metrics(
    merged_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    master_records,
    transcriptions_dir: str = "artifacts/transcriptions",
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics and return as a flat dict.

    Parameters
    ----------
    merged_df        : Inner-joined predictions × ground truth (matched items only)
    predictions_df   : Full predictions DataFrame (all predicted items)
    ground_truth_df  : Full ground truth DataFrame (all GT items)
    master_records   : Master product records (kept for API compatibility)
    transcriptions_dir : Directory containing cleaned STT .txt files
    """
    metrics: Dict[str, Any] = {}

    if len(merged_df) == 0:
        return {"error": "No data to evaluate"}

    # Ensure required columns exist in merged_df
    for col in ["matched_sku", "correct_sku_code", "predicted_canonical_name",
                "brand", "item_name", "item_quantity", "correct_quantity",
                "confidence_score", "top_5_candidates"]:
        if col not in merged_df.columns:
            merged_df[col] = None

    # -----------------------------------------------------------------------
    # 1. Confusion Matrix
    # -----------------------------------------------------------------------
    print("[Phase E] Computing confusion matrix...")
    cm = compute_confusion_matrix(predictions_df, ground_truth_df, merged_df)
    metrics.update(cm)

    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    # -----------------------------------------------------------------------
    # 2. Classification Metrics
    # -----------------------------------------------------------------------
    print("[Phase E] Computing classification metrics...")
    cls = compute_classification_metrics(tp, fp, fn, tn)
    metrics.update(cls)

    # -----------------------------------------------------------------------
    # 3. Quantity MAE
    # -----------------------------------------------------------------------
    print("[Phase E] Computing quantity MAE...")
    metrics["quantity_mae"] = compute_quantity_mae(merged_df)

    # -----------------------------------------------------------------------
    # 4. WER / CER
    # -----------------------------------------------------------------------
    print("[Phase E] Computing WER / CER per recording...")
    mean_wer, mean_cer = compute_wer_cer(ground_truth_df, transcriptions_dir)
    metrics["mean_wer"] = round(mean_wer, 4) if not np.isnan(mean_wer) else None
    metrics["mean_cer"] = round(mean_cer, 4) if not np.isnan(mean_cer) else None

    # -----------------------------------------------------------------------
    # 5. Legacy / additional metrics
    # -----------------------------------------------------------------------
    print("[Phase E] Computing additional metrics...")
    metrics.update(_compute_legacy_metrics(merged_df))

    # -----------------------------------------------------------------------
    # 6. Dataset size info
    # -----------------------------------------------------------------------
    metrics["num_predictions"]        = len(predictions_df)
    metrics["num_ground_truth"]       = len(ground_truth_df)
    metrics["num_evaluated_samples"]  = len(merged_df)

    return metrics
