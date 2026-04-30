"""
BIS Standards RAG Engine - Retrieval Module
Hybrid retrieval with query expansion
"""

import re
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand queries with synonyms and related terms"""
    
    def __init__(self):
        # Domain-specific expansions for building materials
        # IMPORTANT: Don't add cement keywords to steel queries
        self.expansions = {
            "cement": ["portland", "opc", "ppc", "psc", "slag cement", "pozzolana"],
            "steel": ["reinforcement", "tmt", "fe415", "fe500", "structural steel", "rebar", "tmt bar", "mesh"],
            "concrete": ["mix", "rcc", "pcc", "m20", "m25", "m30", "grade", "proportion"],
            "aggregate": ["sand", "coarse aggregate", "fine aggregate", "crushed stone", "gravel", "river sand"],
            "brick": ["clay brick", "fly ash brick", "solid brick", "hollow block"],
            "sand": ["fine aggregate", "river sand", "m sand", "manufactured sand"],
            "reinforcement": ["rebar", "steel bar", "tmt bar", "mesh"],
            "mix design": ["proportion", "mix ratio", "grade"],
            "tmt": ["reinforcement", "steel bar", "rebar"],
            "rebar": ["reinforcement", "steel bar", "tmt"],
        }
        
        # Keywords that should NOT trigger cement expansion
        self.cement_blockers = ["steel", "tmt", "rebar", "reinforcement", "fe415", "fe500", "structural"]
    
    def expand(self, query: str) -> List[str]:
        """Expand query with synonyms"""
        return [query]
    
    def expand_query(self, query: str) -> str:
        """Get expanded query string"""
        return query


class QueryClassifier:
    """Classify query into categories for metadata filtering"""
    
    def __init__(self):
        self.category_keywords = {
            "cement": ["cement", "portland", "opc", "ppc", "psc", "binders"],
            "steel": ["steel", "reinforcement", "tmt", "rebar", "structural", "fe415", "fe500"],
            "concrete": ["concrete", "mix", "rcc", "pcc", "m20", "m25", "m30", "grade"],
            "aggregate": ["aggregate", "sand", "gravel", "stone", "crushed"],
            "brick": ["brick", "block", "masonry", "wall"],
            "mortar": ["mortar", "plaster", "tile", "grout"],
        }
    
    def classify(self, query: str) -> str:
        """Classify query into category"""
        query_lower = query.lower()
        
        scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "general"


class Retriever:
    """Main retrieval system with hybrid search"""
    
    def __init__(self, vector_store, embedding_generator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.query_expander = QueryExpander()
        self.query_classifier = QueryClassifier()
    
    def retrieve(
        self, 
        query: str, 
        k: int = 10,
        use_expansion: bool = True,
        use_category_filter: bool = False
    ) -> List[Dict]:
        """Main retrieval function"""
        
        # Classify query
        category = None
        if use_category_filter:
            category = self.query_classifier.classify(query)
            logger.info(f"Query classified as: {category}")
        
        # Expand query
        if use_expansion:
            expanded_query = self.query_expander.expand_query(query)
            logger.info(f"Expanded query: {expanded_query}")
        else:
            expanded_query = query
        
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_single(expanded_query)
        
        # Search vector store
        distances, indices = self.vector_store.search(
            query_embedding, 
            k=k,
            category_filter=category if use_category_filter else None
        )
        
        # Get chunks
        results = self.vector_store.get_chunks(indices)
        
        logger.info(f"Retrieved {len(results)} chunks")
        return results
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """Retrieve with similarity scores"""
        
        # Expand and embed
        expanded_query = self.query_expander.expand_query(query)
        query_embedding = self.embedding_generator.encode_single(expanded_query)
        
        # Search
        distances, indices = self.vector_store.search(query_embedding, k=k)
        
        # Combine with scores
        results = []
        for d, i in zip(distances[0], indices[0]):
            if i >= 0 and i < len(self.vector_store.chunks):
                results.append((self.vector_store.chunks[i], float(d)))
        
        return results


if __name__ == "__main__":
    # Test retrieval
    from vectorstore import VectorStore
    from embeddings import EmbeddingGenerator
    
    # Create dummy components
    import numpy as np
    
    generator = EmbeddingGenerator()
    store = VectorStore(dimension=generator.dimension)
    
    # Create test data
    test_chunks = [
        {"text": "Cement conforming to IS 269 for ordinary portland cement", "category": "cement", "standard_id": "IS 269"},
        {"text": "Steel reinforcement bars conforming to IS 1786 for high strength deformed bars", "category": "steel", "standard_id": "IS 1786"},
        {"text": "Concrete mix design as per IS 456 for plain and reinforced concrete", "category": "concrete", "standard_id": "IS 456"},
        {"text": "Fine aggregate conforming to IS 383 for concrete works", "category": "aggregate", "standard_id": "IS 383"},
        {"text": "TMT bars grade Fe 415 and Fe 500 as per IS 1786", "category": "steel", "standard_id": "IS 1786"},
    ]
    
    embeddings = generator.encode([c["text"] for c in test_chunks])
    store.build_index(embeddings, test_chunks)
    
    # Test retriever
    retriever = Retriever(store, generator)
    
    results = retriever.retrieve("cement for building construction", k=3)
    print("\nRetrieval results:")
    for r in results:
        print(f"  - {r['text'][:60]}... (category: {r['category']})")