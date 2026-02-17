import pandas as pd
from typing import List, Dict, Any, Set
import re
import jellyfish

# ==========================================================
# TEXT NORMALIZATION (Standardized with Phase D Normalizer)
# ==========================================================

def normalize_basic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_unit(unit: str) -> str:
    # Synchronized with your TextNormalizer.py
    unit_map = {
        'gm': 'g', 'g': 'g', 'gram': 'g', 'grams': 'g',
        'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
        'ltr': 'ltr', 'l': 'ltr', 'liter': 'ltr', 'litre': 'ltr',
        'ml': 'ml', 'milliliter': 'ml', 'pcs': 'pcs', 'pc': 'pcs'
    }
    unit = str(unit).strip().lower()
    return unit_map.get(unit, unit)

# ==========================================================
# SIZE STANDARDIZATION (Internal Metric System)
# ==========================================================

def standardize_size_unit(size, unit):
    if size is None or unit is None:
        return None, None
    try:
        size = float(size)
    except:
        return None, None

    unit = normalize_unit(unit)
    # Convert all to base metric (g or ml) for mathematical comparison in Phase D
    if unit == "ltr": return size * 1000, "ml"
    if unit == "kg": return size * 1000, "g"
    return size, unit

# ==========================================================
# LOAD MASTER DATA (Optimized for Scoring)
# ==========================================================

def load_master_data(file_path: str) -> List[Dict[str, Any]]:
    print(f"[Master Loader] Processing catalog: {file_path}")

    if str(file_path).lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, encoding="cp1252")

    df.columns = df.columns.str.strip()
    df = df.drop_duplicates(subset=["SKU_Codes"], keep="first")

    master_records = []
    # Collect all unique brands for the Scorer's 'all_known_brands' set
    all_brands = set()

    for _, row in df.iterrows():
        brand = str(row.get('Brand_Name', '')).strip()
        item = str(row.get('Item_name', '')).strip()
        if not brand or not item: continue

        # 1. Standardize Quantity
        quantity_text = str(row.get("Quantities", "")).lower()
        match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|gm|g|ltr|l|ml|pcs|pc)', quantity_text)
        
        size_std, unit_std = (None, None)
        if match:
            size_std, unit_std = standardize_size_unit(float(match.group(1)), match.group(2))

        # 2. Generate Search & Scoring Fields
        normalized_brand = normalize_basic(brand)
        normalized_item = normalize_basic(item)
        all_brands.add(normalized_brand)

        # Build a robust canonical name for the user interface
        size_label = f"{int(size_std) if size_std == int(size_std) else size_std} {unit_std}" if size_std else ""
        canonical = f"{brand} {item} {size_label}".strip()
        normalized_name = normalize_basic(canonical)

        # 3. Build Record with Phonetic & Token support
        record = {
            "sku_code": str(row.get("SKU_Codes", "")).strip().upper(),
            "canonical_name": canonical,
            "normalized_name": normalized_name,
            "normalized_brand": normalized_brand,
            "normalized_item": normalized_item,
            "size": size_std,
            "unit": unit_std,
            # PHONETIC: Critical for STT mishearings
            "phonetic_item": jellyfish.metaphone(normalized_item),
            "phonetic_brand": jellyfish.metaphone(normalized_brand),
            # TOKENS: Used for Jaccard similarity and early filtering
            "tokens": set(normalized_name.split())
        }

        master_records.append(record)

    print(f"[Master Loader] Loaded {len(master_records)} SKUs and {len(all_brands)} unique brands.")
    return master_records, all_brands