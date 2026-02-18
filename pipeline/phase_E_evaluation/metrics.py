"""
Metrics - Phase E 

"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score


def safe_upper(series):
    return series.fillna("").astype(str).str.upper()


def calculate_all_metrics(
    merged_df: pd.DataFrame,
    master_records
) -> Dict[str, Any]:

    metrics: Dict[str, Any] = {}
    total = len(merged_df)

    if total == 0:
        return {"error": "No data to evaluate"}

    # ------------------------------------------------------------------
    # SAFE COLUMN ACCESS
    # ------------------------------------------------------------------
    required_columns = [
        "matched_sku",
        "correct_sku_code",
        "predicted_canonical_name",
        "brand",
        "item_name",
        "item_quantity",
        "correct_quantity",
        "confidence_score",
        "top_5_candidates"
    ]

    for col in required_columns:
        if col not in merged_df.columns:
            merged_df[col] = None

    # ------------------------------------------------------------------
    # 1️⃣ EXACT SKU MATCH
    # ------------------------------------------------------------------
    correct_mask = (
        safe_upper(merged_df["matched_sku"]) ==
        safe_upper(merged_df["correct_sku_code"])
    )

    metrics["exact_match_accuracy"] = correct_mask.mean() * 100

    # ------------------------------------------------------------------
    # 2️⃣ FAMILY MATCH (Brand + Item Level)
    # ------------------------------------------------------------------
    def is_family_match(row):
        pred = str(row["predicted_canonical_name"]).lower()
        brand = str(row["brand"]).lower()
        item = str(row["item_name"]).lower()

        brand_match = brand and brand in pred
        item_match = item and item in pred
        return brand_match and item_match

    family_matches = merged_df.apply(is_family_match, axis=1)
    metrics["family_match_accuracy"] = family_matches.mean() * 100

    # ------------------------------------------------------------------
    # 3️⃣ QUANTITY ACCURACY
    # ------------------------------------------------------------------
    quantity_match = (
        merged_df["item_quantity"].fillna(-1) ==
        merged_df["correct_quantity"].fillna(-2)
    )

    metrics["quantity_accuracy"] = quantity_match.mean() * 100

    # ------------------------------------------------------------------
    # 4️⃣ TOP-3 ACCURACY
    # ------------------------------------------------------------------
    correct_top3 = 0

    for _, row in merged_df.iterrows():
        gt_sku = str(row["correct_sku_code"]).upper()

        try:
            candidates = row["top_5_candidates"]

            if isinstance(candidates, str):
                import ast
                candidates = ast.literal_eval(candidates)

            top3 = [str(c.get("sku", "")).upper() for c in candidates[:3]]

            if gt_sku in top3:
                correct_top3 += 1

        except Exception:
            continue

    metrics["top_3_accuracy"] = (correct_top3 / total) * 100

    # ------------------------------------------------------------------
    # 5️⃣ CONFIDENCE METRICS
    # ------------------------------------------------------------------
    metrics["mean_confidence_score"] = (
        merged_df["confidence_score"].fillna(0).mean()
    )

    is_correct_numeric = correct_mask.astype(int)

    metrics["confidence_weighted_accuracy"] = (
        (merged_df["confidence_score"].fillna(0) * is_correct_numeric).mean()
    )

    # ------------------------------------------------------------------
    # 6️⃣ PRECISION / RECALL / F1 (Macro)
    # ------------------------------------------------------------------
    y_true = safe_upper(merged_df["correct_sku_code"])
    y_pred = safe_upper(merged_df["matched_sku"])

    metrics["precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics["recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics["f1_score"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # ------------------------------------------------------------------
    # 7️⃣ NOT FOUND ACCURACY
    # ------------------------------------------------------------------
    not_found_mask = y_pred == ""
    correct_not_found = (not_found_mask & (y_true == "")).mean()

    metrics["not_found_accuracy"] = correct_not_found * 100

    return metrics
