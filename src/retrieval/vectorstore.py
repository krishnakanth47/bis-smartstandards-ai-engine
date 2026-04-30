"""
BIS Standards RAG Engine - Vector Store Module
FAISS-based vector database for efficient similarity search
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for chunk retrieval"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """Build FAISS index from embeddings"""
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embs = embeddings / norms
        
        # Use Inner Product for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(normalized_embs.astype(np.float32))
        
        self.chunks = chunks
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10,
        category_filter: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k similar chunks"""
        if self.index is None:
            logger.warning("Index not built yet!")
            return np.array([]), np.array([])
        
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)
        
        # Soft filtering is applied in Reranker; just retrieve top-k directly
        distances, indices = self.index.search(query_norm, k)
        
        return distances, indices
    
    def get_chunks(self, indices: np.ndarray) -> List[Dict]:
        """Get chunk metadata for given indices"""
        results = []
        for i in indices[0]:
            if i >= 0 and i < len(self.chunks):
                results.append(self.chunks[i])
        return results
    
    def save_index(self, path: str):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, f"{path}.index")
        
        metadata = {
            "chunks": self.chunks,
            "dimension": self.dimension
        }
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Saved index to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index and metadata"""
        self.index = faiss.read_index(f"{path}.index")
        
        with open(f"{path}.meta", "rb") as f:
            metadata = pickle.load(f)
            
        self.chunks = metadata["chunks"]
        self.dimension = metadata["dimension"]
        logger.info(f"Loaded index from {path}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "total_chunks": len(self.chunks)
        }


class HybridVectorStore(VectorStore):
    """Enhanced vector store with keyword search capability"""
    
    def __init__(self, dimension: int = 384):
        super().__init__(dimension)
        self.keyword_index = {}  # Simple keyword inverted index
        
    def build_keyword_index(self):
        """Build keyword inverted index for hybrid search"""
        for i, chunk in enumerate(self.chunks):
            words = chunk["text"].lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(i)
                    
        logger.info(f"Built keyword index with {len(self.keyword_index)} unique terms")
    
    def keyword_search(self, query: str, k: int = 10) -> List[int]:
        """Simple keyword-based search"""
        query_words = query.lower().split()
        scores = {}
        
        for word in query_words:
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    scores[idx] = scores.get(idx, 0) + 1
        
        # Sort by score
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_indices[:k]]
    
    def hybrid_search(
        self, 
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """Combine vector and keyword search"""
        # Vector search
        vec_distances, vec_indices = self.search(query_embedding, k=k*2)
        
        # Keyword search
        kw_indices = self.keyword_search(query_text, k=k*2)
        
        # Combine scores
        combined_scores = {}
        
        # Add vector scores
        for d, i in zip(vec_distances[0], vec_indices[0]):
            if i >= 0:
                combined_scores[i] = combined_scores.get(i, 0) + d * vector_weight
                
        # Add keyword scores
        for rank, i in enumerate(kw_indices):
            score = (k - rank) / k * keyword_weight
            combined_scores[i] = combined_scores.get(i, 0) + score
        
        # Sort and return top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, _ in sorted_results[:k]:
            if i >= 0 and i < len(self.chunks):
                results.append(self.chunks[i])
                
        return results


if __name__ == "__main__":
    # Test vector store
    store = VectorStore(dimension=384)
    
    # Create dummy embeddings
    embeddings = np.random.randn(100, 384).astype(np.float32)
    chunks = [{"text": f"Chunk {i}", "category": "test"} for i in range(100)]
    
    store.build_index(embeddings, chunks)
    
    # Search
    query = np.random.randn(384).astype(np.float32)
    distances, indices = store.search(query, k=5)
    
    print(f"Search results: indices = {indices}, distances = {distances}")