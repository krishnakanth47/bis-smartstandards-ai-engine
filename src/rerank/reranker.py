"""
BIS Standards RAG Engine - Reranking Module
Improve retrieval relevance with cross-encoder scoring
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved chunks for better relevance"""
    
    def __init__(self, embedding_generator=None):
        self.embedding_generator = embedding_generator
        # Keyword boost weights for different categories
        self.category_keywords = {
            "cement": ["cement", "opc", "portland", "ppc", "psc", "pozzolana", "slag", "grade", "33", "43", "53", "calcined", "clay", "supersulphated", "white", "masonry", "asbestos", "corrugated", "sheet"],
            "steel": ["steel", "tmt", "rebar", "reinforcement", "fe415", "fe500", "fe550", "yield", "elongation"],
            "concrete": ["concrete", "mix", "m20", "m25", "m30", "grade", "strength", "workability", "slump", "precast", "pipe", "hollow", "solid", "lightweight", "block", "water", "main"],
            "aggregate": ["aggregate", "sand", "fine", "coarse", "gravel", "crushed", "grading", "silt", "natural", "source"],
            "pozzolana": ["pozzolana", "fly ash", "ppc", "pozzolanic", "durability"]
        }
        
    def _compute_keyword_score(self, query: str, chunk: Dict) -> float:
        """Compute keyword matching score between query and chunk"""
        query_lower = query.lower()
        text_lower = chunk.get("text", "").lower()
        standard_id = chunk.get("standard_id", "").lower()
        
        score = 0.0
        
        # Determine query category
        query_category = self._classify_query_category(query_lower)
        
        # Boost for category match
        chunk_category = chunk.get("category", "general").lower()
        if query_category and query_category == chunk_category:
            score += 2.0
        
        # Check standard ID mentions in query
        if standard_id and standard_id in query_lower:
            score += 3.0
        
        # Count keyword matches
        if query_category and query_category in self.category_keywords:
            keywords = self.category_keywords[query_category]
            for kw in keywords:
                if kw in text_lower:
                    score += 0.5
                if kw in query_lower and kw in text_lower:
                    score += 1.0
        
        # Direct text overlap
        query_words = set(re.findall(r'\w+', query_lower))
        text_words = set(re.findall(r'\w+', text_lower))
        overlap = query_words.intersection(text_words)
        if overlap:
            score += len(overlap) * 0.1
        
        return score
    
    def _classify_query_category(self, query: str) -> str:
        """Classify query into category for keyword matching"""
        for category, keywords in self.category_keywords.items():
            for kw in keywords:
                if kw in query:
                    return category
        return None
        
    def rerank_by_similarity(
        self, 
        query: str, 
        chunks: List[Dict], 
        k: int = 5
    ) -> List[Dict]:
        """Rerank chunks using cosine similarity"""
        if not chunks:
            return []
            
        if self.embedding_generator is None:
            # Return original order if no embedding generator
            return chunks[:k]
        
        # Get query embedding
        query_emb = self.embedding_generator.encode_single(query)
        
        # Get chunk embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embs = self.embedding_generator.encode(chunk_texts)
        
        # Compute similarities
        similarities = self.embedding_generator.compute_similarities(query_emb, chunk_embs)
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Reorder chunks
        reranked = [chunks[i] for i in sorted_indices[:k]]
        
        logger.info(f"Reranked {len(chunks)} chunks to top {k}")
        return reranked
    
    def rerank_by_metadata(
        self, 
        chunks: List[Dict], 
        k: int = 5,
        prefer_standard_id: bool = True
    ) -> List[Dict]:
        """Rerank by metadata preferences"""
        scored_chunks = []
        
        for chunk in chunks:
            score = 0.0
            
            # Prefer chunks with standard IDs
            if prefer_standard_id and chunk.get("standard_id") != "Unknown":
                score += 1.0
                
            # Prefer specific categories over general
            if chunk.get("category") != "general":
                score += 0.5
                
            scored_chunks.append((chunk, score))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in scored_chunks[:k]]
    
    def hybrid_rerank(
        self,
        query: str,
        chunks: List[Dict],
        k: int = 5,
        similarity_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """Combine similarity and keyword-based reranking for better MRR"""
        if not chunks:
            return []
        
        # Get similarity scores
        if self.embedding_generator:
            query_emb = self.embedding_generator.encode_single(query)
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_embs = self.embedding_generator.encode(chunk_texts)
            similarities = self.embedding_generator.compute_similarities(query_emb, chunk_embs)
        else:
            similarities = np.ones(len(chunks))
            
        # Fallback safety
        original_max_sim = similarities.max() if len(similarities) > 0 else 0
        if original_max_sim < 0.4:
            logger.info("Similarity below threshold. Triggering fallback safety.")
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Inject 0.0 confidence so pipeline knows it's garbage
            fallback_chunks = []
            for i in sorted_indices[:k]:
                chunk_copy = chunks[i].copy()
                chunk_copy["confidence"] = 0.0
                fallback_chunks.append(chunk_copy)
            return fallback_chunks
        
        # Normalize similarities to 0-1
        sim_min, sim_max = similarities.min(), similarities.max()
        if sim_max > sim_min:
            similarities = (similarities - sim_min) / (sim_max - sim_min)
        
        # Calculate combined scores
        raw_combined = []
        for i, chunk in enumerate(chunks):
            sim_score = similarities[i]
            
            # Keyword score - use as tiebreaker (lower weight)
            keyword_score = self._compute_keyword_score(query, chunk)
            keyword_score = min(keyword_score / 5.0, 1.0)
            
            combined = similarity_weight * sim_score + keyword_weight * keyword_score
            raw_combined.append((chunk, combined))
            
        # Normalize combined scores to make the top result 1.0 (better separation)
        combined_scores = []
        if raw_combined:
            max_combined = max([score for _, score in raw_combined])
            if max_combined > 0:
                for i in range(len(raw_combined)):
                    chunk, score = raw_combined[i]
                    normalized_score = score / max_combined
                    
                    # Apply an exponent to increase separation significantly (e.g., cube it)
                    normalized_score = normalized_score ** 3.0
                    
                    # Inject confidence score into chunk
                    chunk_copy = chunk.copy()
                    chunk_copy["confidence"] = round(float(normalized_score), 2)
                    combined_scores.append((chunk_copy, normalized_score))
            else:
                # If all scores are 0, just append them
                for chunk, score in raw_combined:
                    chunk_copy = chunk.copy()
                    chunk_copy["confidence"] = 0.0
                    combined_scores.append((chunk_copy, score))
                    
        # Sort by combined score (descending)
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Reranked {len(chunks)} chunks using hybrid scoring")
        
        return [chunk for chunk, _ in combined_scores[:k]]


class LLMReranker:
    """Use LLM for intelligent reranking (for future enhancement)"""
    
    def __init__(self, llm=None):
        self.llm = llm
        
    def rerank_with_llm(
        self,
        query: str,
        chunks: List[Dict],
        k: int = 5
    ) -> List[Dict]:
        """Use LLM to rate relevance and rerank"""
        if self.llm is None:
            logger.warning("No LLM provided, falling back to similarity reranking")
            return chunks[:k]
        
        # This would use the LLM to score each chunk
        # For now, return original order
        return chunks[:k]
    
    def rate_relevance(self, query: str, chunk: Dict) -> float:
        """Rate relevance of a chunk to query (1-10)"""
        # Simple heuristic-based scoring
        query_lower = query.lower()
        chunk_lower = chunk["text"].lower()
        
        # Count query terms in chunk
        query_terms = query_lower.split()
        matches = sum(1 for term in query_terms if term in chunk_lower)
        
        score = matches / len(query_terms) if query_terms else 0
        return min(score * 10, 10)


if __name__ == "__main__":
    # Test reranking
    from embeddings import EmbeddingGenerator
    
    generator = EmbeddingGenerator()
    reranker = Reranker(embedding_generator=generator)
    
    test_chunks = [
        {"text": "Cement conforming to IS 269", "category": "cement", "standard_id": "IS 269"},
        {"text": "Steel bars for reinforcement", "category": "steel", "standard_id": "IS 1786"},
        {"text": "Concrete mix design as per IS 456", "category": "concrete", "standard_id": "IS 456"},
        {"text": "General building materials", "category": "general", "standard_id": "Unknown"},
        {"text": "Portland cement specifications", "category": "cement", "standard_id": "IS 269"},
    ]
    
    query = "cement for construction"
    
    # Test similarity reranking
    reranked = reranker.rerank_by_similarity(query, test_chunks, k=3)
    print("Similarity reranked:")
    for r in reranked:
        print(f"  - {r['text'][:50]}...")
    
    # Test hybrid reranking
    hybrid_reranked = reranker.hybrid_rerank(query, test_chunks, k=3)
    print("\nHybrid reranked:")
    for r in hybrid_reranked:
        print(f"  - {r['text'][:50]}...")