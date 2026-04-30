"""
BIS Standards RAG Engine - Text Chunking Module
Intelligent chunking with metadata preservation
"""

import re
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Intelligent text chunking with metadata"""
    
    def __init__(
        self, 
        chunk_size: int = 500,
        overlap: int = 50,
            min_chunk_size: int = 30
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
    def extract_standard_id(self, text: str) -> str:
        """Extract BIS standard ID from text (e.g., IS 456:2000)"""
        # More comprehensive patterns - order matters (more specific first)
        patterns = [
            (r'IS\s*(\d{4,5})[^\d]*(\d{4})', lambda m: f"IS {m.group(1)}: {m.group(2)}"),  # IS 10262: 2009
            (r'IS\s*(\d+):(\d{4})', lambda m: f"IS {m.group(1)}: {m.group(2)}"),  # IS 456: 2000
            (r'IS\s*(\d+)', lambda m: f"IS {m.group(1)}"),  # IS 269
            (r'BIS\s*(\d+)', lambda m: f"IS {m.group(1)}"),  # BIS 269 -> IS 269
        ]
        
        for pattern, formatter in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)
        return "Unknown"
    
    def extract_category(self, text: str) -> str:
        """Extract category from text based on keywords"""
        text_lower = text.lower()
        
        categories = {
            "cement": ["cement", "portland", "pozzolana", "slag"],
            "steel": ["steel", "reinforcement", "tmt", "structural"],
            "concrete": ["concrete", "mix", "aggregate", "mortar"],
            "aggregate": ["aggregate", "sand", "gravel", "stone"],
            "brick": ["brick", "block", "masonry"],
            "mortar": ["mortar", "plaster", "tile"],
        }
        
        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "general"
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs"""
        # Split by multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_by_tokens(self, text: str) -> List[str]:
        """Split text by approximate token count"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
                
        return chunks
    
    def chunk_text(self, text: str, source: str = "unknown") -> List[Dict]:
        """Main chunking function with metadata"""
        chunks = []
        
        # First try paragraph-based chunking
        paragraphs = self.chunk_by_paragraphs(text)
        
        for para in paragraphs:
            # If paragraph is too long, split by tokens
            if len(para.split()) > self.chunk_size:
                sub_chunks = self.chunk_by_tokens(para)
                for chunk_text in sub_chunks:
                    chunks.append({
                        "text": chunk_text,
                        "standard_id": self.extract_standard_id(chunk_text),
                        "category": self.extract_category(chunk_text),
                        "source": source
                    })
            else:
                chunks.append({
                    "text": para,
                    "standard_id": self.extract_standard_id(para),
                    "category": self.extract_category(para),
                    "source": source
                })
        
        # Filter out small chunks
        chunks = [c for c in chunks if len(c["text"]) >= self.min_chunk_size]
        
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks
    
    def chunk_documents(self, documents: Dict[str, str]) -> List[Dict]:
        """Chunk multiple documents"""
        all_chunks = []
        
        for source, text in documents.items():
            chunks = self.chunk_text(text, source)
            all_chunks.extend(chunks)
            
        logger.info(f"Total chunks: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    # Test chunking
    chunker = TextChunker(chunk_size=500)
    
    sample_text = """
    IS 456:2000 - Plain and Reinforced Concrete Code of Practice
    
    Cement: Cement shall conform to IS 269 or IS 1489 or IS 8112 or IS 12269.
    
    Fine Aggregate: Fine aggregate shall conform to IS 383.
    
    Coarse Aggregate: Coarse aggregate shall conform to IS 383.
    """
    
    chunks = chunker.chunk_text(sample_text, "test")
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Category: {chunk['category']}")
        print(f"  Standard: {chunk['standard_id']}")
        print(f"  Text: {chunk['text'][:100]}...")