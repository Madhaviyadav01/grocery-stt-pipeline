"""
Test Phase D SKU-Level Matching with Phase C output
Tests SKU-level fuzzy matching on extracted grocery items from rec_1
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.phase_D_fuzzy_canonical_mapping import (
    load_sku_dataset,
    get_product_mapping,
    get_unique_items,
    map_to_canonical
)


def test_phase_d_sku_level():
    """Test Phase D SKU-level matching with structured items from Phase C rec_1 output."""
    
    # Load Phase C output
    phase_c_output = "artifacts/structured_boundary_extraction/cleaned_rec_1_structured.json"
    
    with open(phase_c_output, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    print(f"Testing Phase D SKU-Level Matching with {len(items)} items from Phase C\n")
    print("="*70)
    
    # Load SKU dataset
    print("\n[Loading SKU Dataset...]")
    df = load_sku_dataset()
    product_list = get_unique_items(df)
    sku_mapping = get_product_mapping(df)
    print(f"Loaded {len(product_list)} SKU-level products from dataset\n")
    
    results = []
    
    for idx, item in enumerate(items, 1):
        print(f"\n[{idx}] Testing: '{item}'")
        
        result = map_to_canonical(item, product_list, sku_mapping, threshold=70)
        
        print(f"    Normalized: '{result['normalized_input']}'")
        
        if result['matched_item_name']:
            print(f"    ✓ Matched: '{result['matched_item_name']}'")
            print(f"      SKU: {result['sku_code']}")
            print(f"      Confidence: {result['confidence_score']}%")
        else:
            print(f"    ✗ No match (best score: {result['confidence_score']}%)")
        
        results.append(result)
    
    print("\n" + "="*70)
    print("\nSUMMARY:")
    print(f"Total items: {len(items)}")
    print(f"Matched: {sum(1 for r in results if r['matched_item_name'])}")
    print(f"Unmatched: {sum(1 for r in results if not r['matched_item_name'])}")
    
    # Save results
    output_path = "artifacts/phase_d_sku_test_rec1.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Show matched items
    print("\n" + "="*70)
    print("MATCHED ITEMS:")
    for r in results:
        if r['matched_item_name']:
            print(f"  '{r['original_input']}'")
            print(f"    → {r['matched_item_name']} (SKU: {r['sku_code']}, {r['confidence_score']}%)")


if __name__ == "__main__":
    test_phase_d_sku_level()
