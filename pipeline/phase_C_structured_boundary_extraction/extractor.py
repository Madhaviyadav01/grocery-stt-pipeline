"""
Extractor - Phase C (Original Simple Version)
Smart boundary detection for grocery items.
"""

import re


def extract_item_boundaries(text: str) -> list:
    """
    Extract grocery items by splitting before quantity numbers.
    
    Strategy:
    - Split before small numbers (1-20) that indicate item quantity
    - Keep measurement numbers (100, 500, etc.) with their units
    - Preserve product names and descriptions together
    """
    if not text:
        return []

    # 1. Lowercase and remove commas
    text = text.lower()
    text = text.replace(",", " ")
    
    # 2. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Split before quantity numbers (1-20 range)
    # BUT NOT before numbers followed by unit words (gram, liter, ml, kg, packet)
    delimiter = "|||"
    
    # Unit words that indicate measurement, not quantity
    unit_pattern = r'(?:gram|g|kg|ml|liter|litre|packet|packets)'
    
    # Replace spaces before quantity numbers with delimiter
    # Pattern: space + number (1-20) + space + NOT a unit word + alphabetic
    # Negative lookahead ensures we don't split before measurements like "1 liter"
    text = re.sub(
        r'\s+(?=(?:[1-9]|1[0-9]|20)\s+(?!' + unit_pattern + r'\b)[a-z])',
        delimiter,
        text
    )
    
    # 4. Split by delimiter
    parts = text.split(delimiter)
    
    # 5. Clean and return
    items = []
    for part in parts:
        cleaned = part.strip()
        
        # Remove trailing isolated numbers (1-20) that are artifacts
        # Pattern: ends with space + number (1-20) at the end
        # BUT keep measurements like "100 gram", "1 liter"
        cleaned = re.sub(r'\s+(?:[1-9]|1[0-9]|20)$', '', cleaned)
        
        if cleaned:
            items.append(cleaned)
    
    return items
