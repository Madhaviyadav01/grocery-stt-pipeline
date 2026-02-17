from typing import Dict, List, Any, Tuple
from .text_normalizer import process_transcript
from .fuzzy_matcher import get_confidence_level, HybridScorerV3
from rapidfuzz import fuzz
import jellyfish


# ==========================================================
# STAGE 1: PRODUCT FAMILY MATCHING
# ==========================================================

def match_product_family(
    query_text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:

    family_scores = {}

    for candidate in candidates:
        family_key = candidate.get("product_family_key", "")
        if not family_key:
            continue

        if family_key in family_scores:
            continue

        family_similarity = fuzz.token_set_ratio(query_text, family_key)

        brand = candidate.get("normalized_brand", "")
        if brand and brand in query_text:
            family_similarity += 15

        family_scores[family_key] = (candidate, min(family_similarity, 100))

    sorted_families = sorted(
        family_scores.items(),
        key=lambda x: x[1][1],
        reverse=True
    )

    return [item[1] for item in sorted_families[:top_k]]


# ==========================================================
# VARIANT RESOLUTION
# ==========================================================

def resolve_variant_within_family(
    query_size: float,
    query_unit: str,
    family_variants: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not family_variants:
        return family_variants[0] if family_variants else None

    # Exact size match
    if query_size is not None:
        for variant in family_variants:
            var_size = variant.get("size")
            var_unit = variant.get("unit")

            if var_size is not None and abs(query_size - var_size) <= 1:
                if not query_unit or str(var_unit).lower() == str(query_unit).lower():
                    return variant

        # Closest size fallback
        variants_with_size = [v for v in family_variants if v.get("size") is not None]
        if variants_with_size:
            return min(
                variants_with_size,
                key=lambda v: abs(v.get("size", 0) - query_size)
            )

    # No size mentioned → choose largest
    variants_with_size = [v for v in family_variants if v.get("size") is not None]
    if variants_with_size:
        return max(variants_with_size, key=lambda v: v.get("size", 0))

    return family_variants[0]


# ==========================================================
# MAIN TRANSCRIPT MAPPING
# ==========================================================

def map_transcript(
    transcript: str,
    tfidf_retriever,
    emb_retriever,
    all_known_brands: set = None
) -> Dict[str, Any]:


    # 1. Pre-process the transcript
    processed = process_transcript(transcript)
    query_text = processed["normalized_text"]
    query_size = processed["size_value"]
    query_unit = processed["size_unit"]
    item_quantity = processed["item_quantity"]

    if not query_text:
        return {
            "transcript": transcript,
            "predicted_canonical_name": None,
            "confidence_score": 0.0,
            "confidence_level": "low",
            "matched_sku": None,
            "top_5_candidates": []
        }

    # 2. Stage 1: Broad Retrieval (TF-IDF or Semantic)
    # Get a wide pool (50-100) to ensure the correct item is at least present
    # 2. Stage 1: Broad Retrieval (Hybrid)

    tfidf_candidates = tfidf_retriever.get_top_candidates(
    query_text=query_text,
    top_k=50
)

    embedding_candidates = emb_retriever.get_top_candidates(
    query_text=query_text,
    top_k=50
)

    # Combine both pools (avoid duplicates by SKU)
    combined = {c["sku_code"]: c for c in tfidf_candidates}

    for c in embedding_candidates:
        combined.setdefault(c["sku_code"], c)

    candidate_pool = list(combined.values())


    if not candidate_pool:
        return {
            "transcript": transcript,
            "predicted_canonical_name": None,
            "confidence_score": 0.0,
            "confidence_level": "low",
            "matched_sku": None,
            "top_5_candidates": []
        }

    # 3. Stage 2: Precision Re-Scoring (HybridScorerV3)
    # We now use the specialized scorer we built earlier
    scorer = HybridScorerV3(all_known_brands=all_known_brands)
    scored_candidates = []

    for candidate in candidate_pool:
        # Use the logic from our improved fuzzy_matcher
        final_score = scorer.compute_final_score(
            query_text=query_text,
            query_size=query_size,
            query_unit=query_unit,
            candidate=candidate
        )

        scored_candidates.append({
            "candidate": candidate,
            "score": final_score
        })

    # 4. Final Ranking
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    best_match = scored_candidates[0]
    best_candidate = best_match["candidate"]
    final_confidence = best_match["score"]

    # Prepare Top 5 for transparency/debugging
    top_5 = []
    for item in scored_candidates[:5]:
        cand = item["candidate"]
        top_5.append({
            "canonical_name": cand["canonical_name"],
            "sku": cand["sku_code"],
            "score": round(item["score"], 2)
        })

    return {
        "transcript": transcript,
        "predicted_canonical_name": best_candidate["canonical_name"],
        "confidence_score": round(final_confidence, 2),
        "confidence_level": get_confidence_level(final_confidence),
        "matched_sku": best_candidate["sku_code"],
        "normalized_query": query_text,
        "item_quantity": item_quantity,
        "extracted_size": query_size,
        "extracted_unit": query_unit,
        "top_5_candidates": top_5
    }
