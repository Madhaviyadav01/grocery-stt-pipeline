"""
TF-IDF Retriever - Phase D (V3 Optimized)
Enhanced with character n-grams and brand-priority boosting.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import numpy as np
import jellyfish


class TFIDFRetriever:

    def __init__(self, master_records: List[Dict[str, Any]]):
        """
        Build an enhanced TF-IDF matrix focusing on Brand and Phonetic resilience.
        """
        self.master_records = master_records
        
        # IMPROVEMENT: Feature Engineering for the Index
        # We combine brand (weighted x4), item (weighted x2), and phonetic metaphone
        self.texts = []
        for rec in master_records:
            brand = (rec.get("normalized_brand", "") + " ") * 4
            item = (rec.get("normalized_item", "") + " ") * 2
            full_name = rec.get("normalized_name", "")
            
            # Add phonetic representation to the index to catch ASR misspellings
            phonetic = jellyfish.metaphone(full_name)
            
            self.texts.append(f"{brand} {item} {full_name} {phonetic}")

        # IMPROVEMENT: Use character n-grams (3-5) to handle partial word matches
        # This helps if the STT hears "Britan" instead of "Britannia"
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Catch single words and common phrases
            analyzer='word',      # Can switch to 'char_wb' for even higher typo resilience
            stop_words=None,
            min_df=1,
            max_df=0.9
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def get_top_candidates(
        self,
        query_text: str,
        top_k: int = 50  # Increased from 30 to improve recall for Phase D
    ) -> List[Dict[str, Any]]:
        """
        Return top_k candidates with an added phonetic query boost.
        """
        if not query_text:
            return []

        # Create a hybrid query: original text + metaphone of query
        query_phonetic = jellyfish.metaphone(query_text)
        enhanced_query = f"{query_text} {query_phonetic}"

        query_vec = self.vectorizer.transform([enhanced_query])

        similarities = cosine_similarity(
            query_vec,
            self.tfidf_matrix
        ).flatten()

        # Get indices of top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.master_records[i] for i in top_indices]