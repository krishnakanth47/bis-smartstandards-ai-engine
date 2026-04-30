"""
BIS Standards RAG Engine - Embeddings Module
Generate embeddings using sentence transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0]
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Normalize
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1_norm, emb2_norm))
    
    def compute_similarities(self, query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query and corpus"""
        # Normalize
        query_norm = query_emb / np.linalg.norm(query_emb)
        corpus_norm = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)
        return np.dot(corpus_norm, query_norm)
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Embed all chunks with their metadata"""
        texts = [chunk["text"] for chunk in chunks]
        
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.encode(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
            
        return chunks
    
    def save_embeddings(self, chunks: List[Dict], path: str):
        """Save embeddings to file"""
        data = {
            "chunks": chunks,
            "model_name": self.model_name
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved embeddings to {path}")
    
    def load_embeddings(self, path: str) -> List[Dict]:
        """Load embeddings from file"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded embeddings from {path}")
        return data["chunks"]


if __name__ == "__main__":
    # Test embeddings
    generator = EmbeddingGenerator()
    
    test_texts = [
        "Cement conforming to IS 269",
        "Steel reinforcement bars",
        "Concrete mix design"
    ]
    
    embeddings = generator.encode(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = generator.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 0 and 1: {sim:.4f}")