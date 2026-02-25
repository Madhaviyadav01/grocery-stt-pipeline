"""
Load Inputs - Phase E 
"""

import pandas as pd
from typing import Tuple

# Required for merging/indexing
REQUIRED_KEYS = ["record_id", "item_index"]

# Required for actual accuracy calculation
EVAL_COLUMNS = ["correct_sku_code", "correct_quantity"]


def validate_columns(df: pd.DataFrame, required_cols: list, df_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def safe_extract_numeric(series: pd.Series):
    return (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )


def load_inputs(
    predictions_path: str = "artifacts/phase_D_output.csv",
    ground_truth_path: str = "data/ground_truth.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    print("=" * 60)
    print("[Phase E] Loading Data for Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1️⃣ LOAD PREDICTIONS
    # ------------------------------------------------------------
    print("[Phase E] Loading predictions...")
    preds_df = pd.read_csv(predictions_path)

    # Extract record_id from source_file if needed
    if "record_id" not in preds_df.columns:
        if "source_file" in preds_df.columns:
            preds_df["record_id"] = safe_extract_numeric(preds_df["source_file"])
        else:
            raise ValueError("Predictions missing both 'record_id' and 'source_file' columns.")

    validate_columns(preds_df, REQUIRED_KEYS, "Predictions")

    # Clean numeric keys
    preds_df[REQUIRED_KEYS] = preds_df[REQUIRED_KEYS].apply(
        pd.to_numeric, errors="coerce"
    )

    preds_df = preds_df.dropna(subset=REQUIRED_KEYS)
    preds_df[REQUIRED_KEYS] = preds_df[REQUIRED_KEYS].astype(int)

    # ------------------------------------------------------------
    # 2️⃣ LOAD GROUND TRUTH
    # ------------------------------------------------------------
    print("[Phase E] Loading ground truth...")

    if ground_truth_path.endswith(".csv"):
        gt_df = pd.read_csv(ground_truth_path)
    else:
        gt_df = pd.read_excel(ground_truth_path)

    # Standardize column names
    gt_df.columns = gt_df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Ensure record_id exists
    if "record_id" not in gt_df.columns:
        raise ValueError("Ground Truth missing 'record_id' column.")

    # Extract numeric record_id safely
    gt_df["record_id"] = safe_extract_numeric(gt_df["record_id"])

    validate_columns(gt_df, REQUIRED_KEYS + EVAL_COLUMNS, "Ground Truth")

    # Clean numeric keys
    gt_df[REQUIRED_KEYS] = gt_df[REQUIRED_KEYS].apply(
        pd.to_numeric, errors="coerce"
    )

    gt_df = gt_df.dropna(subset=REQUIRED_KEYS)
    gt_df[REQUIRED_KEYS] = gt_df[REQUIRED_KEYS].astype(int)

    # Remove duplicate GT rows
    gt_df = gt_df.drop_duplicates(subset=REQUIRED_KEYS)

    # ------------------------------------------------------------
    # 3️⃣ MERGE (Accuracy Engine)
    # ------------------------------------------------------------
    merged_df = pd.merge(
        preds_df,
        gt_df,
        on=["record_id", "item_index"],
        how="inner",
        suffixes=("_pred", "_gt")
    )

    print("\n[Evaluation Stats]")
    print(f"Total Items in Ground Truth:  {len(gt_df)}")
    print(f"Total Items Predicted:        {len(preds_df)}")
    print(f"Directly Evaluable Items:     {len(merged_df)}")
    
    # Rename Columns for Metrics
    # 1. Restore Pred columns that got suffixed
    if "item_quantity_pred" in merged_df.columns:
        merged_df["item_quantity"] = merged_df["item_quantity_pred"]
        
    # 2. Restore GT columns for Family Match (brand_gt -> brand)
    rename_map = {
        "brand_gt": "brand",
        "item_name_gt": "item_name"
    }
    
    for old_col, new_col in rename_map.items():
        if old_col in merged_df.columns:
            # print(f"  -> Renaming {old_col} to {new_col}")
            merged_df[new_col] = merged_df[old_col]

    if len(merged_df) < len(gt_df):
        print(
            f"WARNING: {len(gt_df) - len(merged_df)} "
            f"items from Ground Truth were not matched to any prediction."
        )

    return preds_df, gt_df, merged_df
