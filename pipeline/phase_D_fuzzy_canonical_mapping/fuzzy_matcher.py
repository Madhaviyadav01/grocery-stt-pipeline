"""
Improved Fuzzy Matcher - Phase D (Production V3)
Optimized for 50%+ Accuracy in B2B Grocery Voice AI.
"""

from rapidfuzz import fuzz
from typing import Dict, Any, List, Tuple, Set
import jellyfish

# =====================================================
# UTILITIES
# =====================================================

def jaccard_similarity_tokens(tokens1: set, tokens2: set) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return (intersection / union) * 100 if union > 0 else 0.0

# =====================================================
# HYBRID STRUCTURED SCORER V3
# =====================================================

class HybridScorerV3:
    def __init__(self, all_known_brands: Set[str] = None):
        # Providing a set of brands allows for hard penalties on brand-mismatch
        self.all_known_brands = all_known_brands or set()

    def score_name(self, query: str, candidate: Dict[str, Any]) -> float:
        candidate_name = candidate.get("normalized_name", "")
        if not candidate_name:
            return 0.0

        # Using weighted combination of different fuzzy ratios
        token_set = fuzz.token_set_ratio(query, candidate_name)
        partial = fuzz.partial_ratio(query, candidate_name)
        token_sort = fuzz.token_sort_ratio(query, candidate_name)
        jaccard = jaccard_similarity_tokens(set(query.split()), candidate.get("tokens", set()))

        return (0.40 * token_set + 0.25 * partial + 0.20 * token_sort + 0.15 * jaccard)

    def score_brand(self, query: str, candidate: Dict[str, Any]) -> float:
        brand = candidate.get("normalized_brand", "").lower()
        if not brand:
            return 0.0

        query_tokens = [t.lower() for t in query.split()]
        
        # IMPROVEMENT: Brand Penalty Logic
        # If the user says "Amul" but the candidate brand is "Britannia", 
        # we return a negative score to push this candidate down.
        if any(b in query_tokens for b in self.all_known_brands if b != brand):
            return -30.0 

        if brand in query_tokens:
            return 100.0

        return fuzz.partial_ratio(brand, query)

    def score_size(self, query_size: float, candidate: Dict[str, Any]) -> float:
        candidate_size = candidate.get("size")
        if query_size is None or candidate_size is None:
            return None 

        diff = abs(query_size - candidate_size)
        # B2B specific: 100g and 500g are totally different SKUs
        if diff == 0:
            return 100.0
        elif diff <= 5:
            return 80.0
        elif diff <= 20:
            return 40.0
        else:
            return 0.0

    def score_unit(self, query_unit: str, candidate: Dict[str, Any]) -> float:
        candidate_unit = candidate.get("unit")
        if not query_unit or not candidate_unit:
            return None 
        return 100.0 if query_unit.lower() == candidate_unit.lower() else 0.0

    def score_phonetic(self, query: str, candidate: Dict[str, Any]) -> float:
        candidate_item = candidate.get("normalized_item", "")
        if not candidate_item:
            return 0.0

        query_phonetic = jellyfish.metaphone(query)
        candidate_phonetic = candidate.get("phonetic_item", "")
        
        # If the metaphones are identical, give high boost
        return fuzz.ratio(query_phonetic, candidate_phonetic)

    def compute_final_score(
        self,
        query_text: str,
        query_size: float,
        query_unit: str,
        candidate: Dict[str, Any]
    ) -> float:

        name_s = self.score_name(query_text, candidate)
        brand_s = self.score_brand(query_text, candidate)
        size_s = self.score_size(query_size, candidate)
        unit_s = self.score_unit(query_unit, candidate)
        phonetic_s = self.score_phonetic(query_text, candidate)

        # ADJUSTED WEIGHTS for 50%+ Accuracy
        weights = {
            "name": 0.35,     # Reduced slightly to allow Brand/Phonetic to shine
            "brand": 0.25,    # Increased for SKU precision
            "size": 0.10,
            "unit": 0.10,
            "phonetic": 0.20  # Increased to solve STT "mishearing" issues
        }

        # Dynamic adjustments if fields are missing
        if size_s is None:
            weights["name"] += weights["size"]
            weights["size"] = 0
        if unit_s is None:
            weights["name"] += weights["unit"]
            weights["unit"] = 0

        final_score = (
            weights["name"] * name_s +
            weights["brand"] * brand_s +
            weights["size"] * (size_s or 0) +
            weights["unit"] * (unit_s or 0) +
            weights["phonetic"] * phonetic_s
        )

        return round(max(0, final_score), 2)

# =====================================================
# MATCH FUNCTION WITH FUZZY PRE-FILTERING
# =====================================================

def fuzzy_match_structured_v3(
    query_text: str,
    query_size: float,
    query_unit: str,
    candidates: List[Dict[str, Any]],
    all_known_brands: Set[str] = None,
    top_k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:

    scorer = HybridScorerV3(all_known_brands=all_known_brands)
    scored = []

    for candidate in candidates:
        candidate_name = candidate.get("normalized_name", "")
        
        # IMPROVEMENT: Forgiving Early Filter
        # Instead of strict intersection, we use a quick partial ratio check.
        # This catches "Magi" vs "Maggi" which strict intersection misses.
        if fuzz.partial_ratio(query_text, candidate_name) < 45:
            continue

        score = scorer.compute_final_score(
            query_text, query_size, query_unit, candidate
        )
        scored.append((candidate, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def get_confidence_level(score: float) -> str:
    if score >= 85: return "high"
    elif score >= 70: return "medium"
    else: return "low"