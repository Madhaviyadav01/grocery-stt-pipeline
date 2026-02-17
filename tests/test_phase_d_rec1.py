"""
Test Phase D with Phase C output
Tests fuzzy matching on extracted grocery items from rec_1
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.phase_D_fuzzy_canonical_mapping import run_phase_D, load_master_products


def test_phase_d_with_rec1():
    """Test Phase D with structured items from Phase C rec_1 output."""
    
    # Load Phase C output
    phase_c_output = "artifacts/structured_boundary_extraction/cleaned_rec_1_structured.json"
    
    with open(phase_c_output, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    print(f"Testing Phase D with {len(items)} items from Phase C\n")
    print("="*70)
    
    # Load master products once
    master_products = load_master_products()
    
    results = []
    
    for idx, item in enumerate(items, 1):
        print(f"\n[{idx}] Testing: '{item}'")
        
        from pipeline.phase_D_fuzzy_canonical_mapping import map_to_canonical
        result = map_to_canonical(item, master_products, threshold=80)
        
        print(f"    Normalized: '{result['normalized_input']}'")
        
        if result['matched_product']:
            print(f"    ✓ Matched: '{result['matched_product']}' (confidence: {result['confidence_score']}%)")
        else:
            print(f"    ✗ No match (best score: {result['confidence_score']}%)")
        
        results.append(result)
    
    print("\n" + "="*70)
    print("\nSUMMARY:")
    print(f"Total items: {len(items)}")
    print(f"Matched: {sum(1 for r in results if r['matched_product'])}")
    print(f"Unmatched: {sum(1 for r in results if not r['matched_product'])}")
    
    # Save results
    output_path = "artifacts/phase_d_test_rec1.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    test_phase_d_with_rec1()
