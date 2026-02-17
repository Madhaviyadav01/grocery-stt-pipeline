"""
Embedding Retriever - Phase D (V3 Optimized)
Semantic retrieval with Phonetic-Awareness for B2B Grocery.
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import jellyfish
from typing import List, Dict, Any


class EmbeddingRetriever:
    """
    Handles STT errors by capturing semantic and phonetic similarity.
    Optimized for grocery-specific SKU resolution.
    """

    def __init__(self, master_records: List[Dict[str, Any]]):
        # Using a model that is robust to short, keyword-heavy text
        print("[Embedding Retriever] Loading model 'all-MiniLM-L6-v2'...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.master_records = master_records
        
        # IMPROVEMENT: Enrich the embedding string
        # We combine brand (3x boost), the full name, and its phonetic sound.
        self.texts = []
        for rec in master_records:
            brand = (rec.get("normalized_brand", "") + " ") * 3
            name = rec.get("normalized_name", "")
            # Adding phonetic context helps the embedding model 'group' similar sounds
            phonetic = jellyfish.metaphone(name)
            
            self.texts.append(f"{brand} {name} {phonetic}".strip())
        
        print(f"[Embedding Retriever] Encoding {len(self.texts)} products...")
        self.embeddings = self.model.encode(
            self.texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print("[Embedding Retriever] Ready!")

    def get_top_candidates(
        self,
        query_text: str,
        top_k: int = 50  # Increased for better recall
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k candidates using semantic + phonetic similarity.
        """
        if not query_text:
            return []
        
        # IMPROVEMENT: Apply same phonetic transformation to the query
        query_phonetic = jellyfish.metaphone(query_text)
        enhanced_query = f"{query_text} {query_phonetic}"
        
        # Encode query
        query_embedding = self.model.encode(
            enhanced_query,
            convert_to_tensor=True
        )
        
        # Compute similarities
        similarities = cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        k_actual = min(top_k, len(self.master_records))
        top_indices = torch.topk(similarities, k=k_actual).indices
        
        return [self.master_records[idx] for idx in top_indices.cpu().numpy()]