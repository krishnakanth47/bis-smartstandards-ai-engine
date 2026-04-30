"""
BIS Standards RAG Engine - Main Pipeline
Complete RAG pipeline orchestration
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

from src.ingestion.ingestion import PDFIngestion
from src.preprocessing.chunking import TextChunker
from src.preprocessing.embeddings import EmbeddingGenerator
from src.retrieval.vectorstore import VectorStore, HybridVectorStore
from src.retrieval.retriever import Retriever
from src.rerank.reranker import Reranker
from src.rag.generator import SimpleGenerator, LLMGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BISRAGEngine:
    """Complete BIS Standards RAG Engine"""
    
    def __init__(
        self,
        data_dir: str = "data",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        use_llm: bool = False,
        llm_api=None
    ):
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        
        # Check if 'dataset' directory exists with PDFs (official hackathon data)
        if Path("dataset").exists() and list(Path("dataset").glob("*.pdf")):
            logger.info("Found official 'dataset' directory with PDFs. Using for ingestion.")
            self.data_dir = "dataset"
            
        # Initialize components
        logger.info("Initializing RAG engine components...")
        
        # Ingestion
        self.ingestion = PDFIngestion(self.data_dir)
        
        # Chunking
        self.chunker = TextChunker(chunk_size=chunk_size)
        
        # Embeddings
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        
        # Vector store
        self.vector_store = VectorStore(
            dimension=self.embedding_generator.dimension
        )
        
        # Retriever
        self.retriever = Retriever(self.vector_store, self.embedding_generator)
        
        # Reranker
        self.reranker = Reranker(embedding_generator=self.embedding_generator)
        
        # Generator
        if use_llm:
            self.generator = LLMGenerator(llm_api=llm_api)
        else:
            self.generator = SimpleGenerator()
        
        # State
        self.is_indexed = False
    
    def build_index(self, force_rebuild: bool = False):
        """Build or load the vector index"""
        index_path = Path("data/index")
        
        # Check if index exists
        if not force_rebuild and index_path.with_suffix(".meta").exists():
            logger.info("Loading existing index...")
            self.vector_store.load_index(str(index_path))
            self.is_indexed = True
            return
        
        logger.info("Building new index from PDFs...")
        
        # Extract PDFs
        documents = self.ingestion.extract_all_pdfs()
        
        if not documents:
            logger.warning("No PDF documents found in data directory")
            # Create sample data for testing
            documents = self._create_sample_documents()
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        if not chunks:
            logger.error("No chunks created")
            return
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_generator.encode(texts)
        
        # Build index
        logger.info("Building FAISS index...")
        self.vector_store.build_index(embeddings, chunks)
        
        # Save index
        self.vector_store.save_index(str(index_path))
        
        self.is_indexed = True
        logger.info("Index built successfully!")
    
    def _create_sample_documents(self) -> Dict[str, str]:
        """Create sample documents for testing"""
        logger.info("Creating sample BIS documents...")
        
        sample_docs = {
            "IS_456_2000": """
IS 456:2000 - Plain and Reinforced Concrete - Code of Practice

Cement: Cement shall conform to IS 269 or IS 1489 or IS 8112 or IS 12269.

Fine Aggregate: Fine aggregate shall conform to IS 383. Sand shall be clean and free from organic impurities.

Coarse Aggregate: Coarse aggregate shall conform to IS 383. Maximum size not exceeding 20mm for reinforced concrete.

Water: Water shall be clean and free from deleterious materials. Water cement ratio not exceeding 0.50 for durability.

Mix Proportion: Concrete mix shall be designed as per IS 10262. Grade of concrete shall be M20 or higher for reinforced concrete.

Workability: Concrete shall have adequate workability for placement and compaction. Slump typically 25-50mm for reinforced members.
            """,
            
            "IS_269_2015": """
IS 269:2015 - Ordinary Portland Cement - Specification

This standard covers ordinary Portland cement (OPC) used in general concrete construction.

Types: OPC 33 Grade, OPC 43 Grade, OPC 53 Grade

Composition: Cement shall consist mainly of calcium silicates. Tricalcium aluminate content not exceeding 10%.

Fineness: Specific surface area not less than 225 m²/kg for OPC 43 grade.

Strength: Compressive strength at 7 days shall be not less than 22 MPa for OPC 33, 33 MPa for OPC 43, and 43 MPa for OPC 53.

Setting Time: Initial setting time not less than 30 minutes, final setting time not exceeding 600 minutes.
            """,
            
            "IS_1786_2008": """
IS 1786:2008 - High Strength Deformed Steel Bars for Reinforcement

This standard covers hot rolled deformed steel bars for concrete reinforcement.

Grades: Fe 415, Fe 415D, Fe 500, Fe 500D, Fe 550, Fe 550D, Fe 600

Yield Strength: Minimum yield stress 415 N/mm² for Fe 415, 500 N/mm² for Fe 500.

Elongation: Minimum elongation 14.5% for Fe 415, 12% for Fe 500.

Bend Test: Bars shall withstand bending without fracture through 90° around a mandrel.

Chemical Composition: Carbon content not exceeding 0.30%, Sulphur and Phosphorus not exceeding 0.060% each.
            """,
            
            "IS_383_2016": """
IS 383:2016 - Coarse and Fine Aggregate from Natural Sources

This standard covers aggregate for concrete from natural sources.

Fine Aggregate: Sand from natural sources, pit sand, river sand, sea sand.

Coarse Aggregate: Crushed stone, gravel, broken stone, shingle.

Grading: Aggregate shall be well graded. Fine aggregate conforming to Zone I, II, III, or IV of IS 383.

Silt Content: Silt content in fine aggregate not exceeding 3% by weight.

Deleterious Materials: Aggregate shall be free from deleterious materials like clay, silt, organic matter.
            """,
            
            "IS_1489_2015": """
IS 1489:2015 - Portland Pozzolana Cement - Specification

This standard covers Portland Pozzolana Cement (PPC) using pozzolanic materials.

Types: PPC with fly ash as pozzolanic material.

Pozzolanic Material: Fly ash conforming to IS 3812. Minimum 15% by mass.

Strength: Compressive strength at 7 days not less than 22 MPa, at 28 days not less than 33 MPa.

Fineness: Specific surface area not less than 300 m²/kg.

Setting Time: Initial setting time not less than 30 minutes, final setting time not exceeding 600 minutes.
            """,
            
            "IS_8112_2013": """
IS 8112:2013 - 43 Grade Ordinary Portland Cement - Specification

This standard covers 43 grade OPC for general concrete construction.

Composition: Tricalcium silicate content not less than 45%, dicalcium silicate not less than 20%.

Fineness: Specific surface area not less than 225 m²/kg.

Strength: Compressive strength at 3 days not less than 23 MPa, at 7 days not less than 33 MPa, at 28 days not less than 43 MPa.

Setting Time: Initial setting time not less than 30 minutes, final setting time not exceeding 600 minutes.
            """,
            
            "IS_12269_2013": """
IS 12269:2013 - 53 Grade Ordinary Portland Cement - Specification

This standard covers 53 grade OPC for high strength concrete.

Composition: Tricalcium silicate content not less than 50%.

Fineness: Specific surface area not less than 225 m²/kg.

Strength: Compressive strength at 3 days not less than 27 MPa, at 7 days not less than 37 MPa, at 28 days not less than 53 MPa.

Setting Time: Initial setting time not less than 30 minutes, final setting time not exceeding 600 minutes.
            """,
            
            "IS_1077_1992": """
IS 1077:1992 - Common Burnt Clay Building Bricks - Specification

This standard covers the requirements, dimensions and methods of test for common burnt clay building bricks.
Classes: Classified on the basis of average compressive strength.
Dimensions: Standard modular size is 190 x 90 x 90 mm.
Water Absorption: Not more than 20 percent by weight up to class 12.5.
            """,
            
            "IS_2062_2011": """
IS 2062:2011 - Hot Rolled Medium and High Tensile Structural Steel

This standard covers the requirements of steel including micro-alloyed steel supplied in plates, sections, and bars.
Grades: E250, E275, E300, E350, E410, E450.
Chemical Composition: Max Carbon equivalent for weldability.
Mechanical Properties: Tensile strength, yield stress, and percentage elongation specified.
            """,
            
            "IS_875_1987": """
IS 875:1987 - Code of Practice for Design Loads (Other Than Earthquake)

Part 1: Dead Loads - Unit weights of building materials and stored materials.
Part 2: Imposed Loads - Live loads for different occupancies.
Part 3: Wind Loads - Wind pressure coefficients for various shapes and regions.
Part 4: Snow Loads - Design for snow accumulation.
            """,
            
            "IS_3495_1992": """
IS 3495:1992 - Methods of Tests of Burnt Clay Solid Bricks
Part 1: Determination of Compressive Strength.
Part 2: Determination of Water Absorption.
Part 3: Determination of Efflorescence.
            """,
            
            "IS_2185_2005": """
IS 2185:2005 - Concrete Masonry Units - Specification
Part 1: Hollow and Solid Concrete Blocks.
Used for load bearing and non-load bearing walls.
            """,
            
            "IS_12894_2002": """
IS 12894:2002 - Pulverized Fuel Ash-Lime Bricks (Fly Ash Bricks)
Specification for unburnt bricks made of fly ash, lime, and sand.
            """
        }
        
        return sample_docs
    
    def query(self, query: str, k: int = 10, rerank: bool = True) -> Dict:
        """Process a single query"""
        
        if not self.is_indexed:
            self.build_index()
        
        # 1. Retrieve (Get more candidates to counter expansion dilution)
        retrieved = self.retriever.retrieve(query, k=k * 3)
        
        # Detect Category
        category = self.retriever.query_classifier.classify(query)
        
        # Rerank
        if rerank:
            retrieved = self.reranker.hybrid_rerank(query, retrieved, k=5)
            
        # Graceful fallback for unknown queries
        if not retrieved or retrieved[0].get("confidence", 1.0) < 0.25:
            standards = [{
                "standard": "No Match",
                "reason": "No relevant BIS standard found. Try specifying material type.",
                "confidence": 0.0
            }]
        else:
            # Generate
            standards = self.generator.generate_response(query, retrieved)
        
        return {
            "standards": standards,
            "retrieved_chunks": len(retrieved),
            "category": category
        }
    
    def process_batch(self, queries: List[Dict]) -> List[Dict]:
        """Process multiple queries"""
        
        results = []
        
        for item in queries:
            query_id = item.get("id", "unknown")
            query_text = item.get("query", "")
            
            start_time = time.time()
            
            # Process query
            response = self.query(query_text)
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Format output
            result = {
                "id": query_id,
                "retrieved_standards": response["standards"],
                "latency_seconds": round(latency, 2)
            }
            if "expected_standards" in item:
                result["expected_standards"] = item["expected_standards"]
                
            results.append(result)
            
            logger.info(f"Processed query {query_id} in {latency:.2f}s")
        
        return results


def main(input_path: str, output_path: str):
    """Main entry point for inference"""
    
    logger.info(f"Starting BIS RAG Engine...")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    # Load input
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # Initialize engine
    engine = BISRAGEngine()
    
    # Build index
    engine.build_index()
    
    # Process queries
    results = engine.process_batch(data)
    
    # Save output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Processed {len(results)} queries")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BIS Standards RAG Engine")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    main(args.input, args.output)