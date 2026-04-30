"""
BIS Standards RAG Engine - PDF Ingestion Module
Extracts text from BIS PDF documents
"""

from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFIngestion:
    """Extract text from PDF files"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_all_pdfs(self) -> Dict[str, str]:
        """Extract text from all PDFs in data directory"""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        documents = {}
        
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(str(pdf_file))
            if text.strip():
                documents[pdf_file.stem] = text
                
        logger.info(f"Extracted text from {len(documents)} PDF files")
        return documents
    
    def extract_with_metadata(self, pdf_path: str) -> Dict:
        """Extract text with metadata from a PDF"""
        try:
            reader = PdfReader(pdf_path)
            pages = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                pages.append({
                    "page_num": i + 1,
                    "text": text
                })
                
            return {
                "file": pdf_path,
                "num_pages": len(reader.pages),
                "pages": pages
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {"file": pdf_path, "num_pages": 0, "pages": []}


if __name__ == "__main__":
    # Test the ingestion
    ingestion = PDFIngestion("data")
    docs = ingestion.extract_all_pdfs()
    print(f"Loaded {len(docs)} documents")