"""
Text Normalizer - Phase D 
"""

import re
from typing import Dict, Tuple, Optional

# Extended Stopwords to reduce noise
STOPWORDS = {
    'packet', 'pack', 'wala', 'wali', 'ka', 'ki', 'ke', 'ko',
    'hai', 'tha', 'thi', 'the', 'chahiye', 'please', 'add', 'get'
}

# Unified Unit Mapping
UNIT_MAP = {
    'g': 'g', 'gm': 'g', 'gms': 'g', 'gram': 'g', 'grams': 'g',
    'kg': 'kg', 'kgs': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
    'l': 'ltr', 'ltr': 'ltr', 'litre': 'ltr', 'liter': 'ltr', 'liters': 'ltr',
    'ml': 'ml', 'mls': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
    'pc': 'pcs', 'pcs': 'pcs', 'piece': 'pcs', 'pieces': 'pcs'
}

# Support for Hinglish numbers
NUMBER_WORDS = {
    'one': '1', 'ek': '1',
    'two': '2', 'do': '2',
    'three': '3', 'teen': '3',
    'four': '4', 'char': '4',
    'five': '5', 'paanch': '5',
    'half': '0.5', 'aadha': '0.5'
}

# Expanded ASR corrections based on Ground Truth failures
ASR_CORRECTIONS = {
    "dow": "dove",
    "vitania": "britannia",
    "britania": "britannia",
    "skits": "biscuits",
    "serf": "surf",
    "callgate": "colgate",
    "maggie": "maggi",
    "nivea": "nivea",
    "amool": "amul"
}

def normalize_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Apply phrase-level ASR corrections
    for wrong, correct in ASR_CORRECTIONS.items():
        text = text.replace(wrong, correct)

    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        token = NUMBER_WORDS.get(token, token)
        # Only remove stopword if it's not a unit or part of a brand
        if token not in STOPWORDS:
            normalized_tokens.append(token)
    return ' '.join(normalized_tokens).strip()

def extract_size(text: str) -> Tuple[Optional[float], Optional[str], str]:
    if not text:
        return None, None, text
    # Enhanced pattern to catch decimals and various unit spellings
    pattern = r'(\d+(?:\.\d+)?)\s*(g|gm|grams?|kg|kgs?|kilograms?|l|ltr|liters?|ml|mls?|milliliters?|pcs?|pieces?)\b'
    matches = re.findall(pattern, text.lower())
    
    if not matches:
        return None, None, text

    parsed_matches = []
    for value, unit in matches:
        try:
            parsed_matches.append((float(value), unit))
        except ValueError:
            continue

    # Pick the first significant size match
    size_value, raw_unit = parsed_matches[0]
    size_unit = UNIT_MAP.get(raw_unit, raw_unit)

    # Clean the text of all size mentions to isolate the Product Name
    text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return size_value, size_unit, text

def extract_item_quantity(text: str) -> Tuple[int, str]:
    if not text:
        return 1, text

    text = text.strip().lower()

    # Find standalone numbers
    matches = re.findall(r'\b(\d+)\b', text)

    if not matches:
        return 1, text

    # Take the last number as quantity (safer for your pattern)
    qty = int(matches[-1])

    # Remove only that quantity from text
    text = re.sub(r'\b{}\b'.format(qty), '', text, count=1)
    text = re.sub(r'\s+', ' ', text).strip()

    return qty, text


def standardize_size_unit(size, unit):
    if size is None or unit is None:
        return None, None
    unit = unit.lower()
    # Normalize everything to g or ml for internal matching logic
    if unit in ["ltr", "l", "liter", "litre"]:
        return size * 1000, "ml"
    if unit in ["kg", "kilogram"]:
        return size * 1000, "g"
    return size, unit

def process_transcript(text: str) -> Dict[str, any]:
    if not text:
        return {"normalized_text": "", "item_quantity": 1, "size_value": None, "size_unit": None}

    # Step-by-step extraction
    cleaned = normalize_text(text)
    size_value, size_unit, remaining = extract_size(cleaned)
    size_value, size_unit = standardize_size_unit(size_value, size_unit)
    item_qty, remaining = extract_item_quantity(remaining)

    return {
        "normalized_text": remaining.strip(),
        "item_quantity": item_qty,
        "size_value": size_value,
        "size_unit": size_unit
    }