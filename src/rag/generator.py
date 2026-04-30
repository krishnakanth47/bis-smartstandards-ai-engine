"""
BIS Standards RAG Engine - Generator Module
LLM-based response generation with anti-hallucination
"""

import re
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGGenerator:
    """Generate responses using RAG with retrieved context"""
    
    def __init__(self, llm=None):
        self.llm = llm
        
    def extract_standards_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Extract BIS standards from retrieved chunks"""
        standards = []
        seen = set()
        
        for chunk in chunks:
            standard_id = chunk.get("standard_id", "Unknown")
            
            if standard_id != "Unknown" and standard_id not in seen:
                seen.add(standard_id)
                
                # Extract relevant text
                text = chunk.get("text", "")
                
                standards.append({
                    "standard": standard_id,
                    "reason": self._generate_reason(text, standard_id),
                    "category": chunk.get("category", "general"),
                    "source": chunk.get("source", "unknown")
                })
        
        return standards[:5]  # Top 5 standards
    
    def _generate_reason(self, text: str, standard_id: str) -> str:
        """Generate a reason for why this standard applies"""
        # Extract key information from text
        sentences = text.split(".")
        
        # Find most relevant sentence
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip()[:200]
        
        return f"Relevant to {standard_id} requirements"
    
    def generate_response(
        self, 
        query: str, 
        retrieved_chunks: List[Dict],
        max_standards: int = 5
    ) -> List[Dict]:
        """Generate final response with retrieved standards"""
        
        if not retrieved_chunks:
            return [{
                "standard": "No standard found",
                "reason": "No relevant BIS standard found for this product"
            }]
        
        # Extract standards from chunks
        standards = self.extract_standards_from_chunks(retrieved_chunks)
        
        # Limit to max_standards
        standards = standards[:max_standards]
        
        # If using LLM, generate more detailed response
        if self.llm and standards:
            standards = self._enhance_with_llm(query, standards, retrieved_chunks)
        
        return standards
    
    def _enhance_with_llm(
        self, 
        query: str, 
        standards: List[Dict],
        chunks: List[Dict]
    ) -> List[Dict]:
        """Enhance standard explanations with LLM"""
        # This would call the LLM for better explanations
        # For now, return extracted standards
        return standards
    
    def format_output(
        self, 
        query_id: str, 
        standards: List[Dict], 
        latency: float
    ) -> Dict:
        """Format output for evaluation"""
        return {
            "id": query_id,
            "retrieved_standards": standards,
            "latency_seconds": round(latency, 2)
        }


class SimpleGenerator(RAGGenerator):
    """Simple rule-based generator without LLM"""
    
    def __init__(self):
        super().__init__(llm=None)
        
        # Known standards mapping
        self.standards_db = {
            "IS 269": "Ordinary Portland Cement - Specification",
            "IS 269:2015": "Ordinary Portland Cement - Specification",
            "IS 456": "Plain and Reinforced Concrete - Code of Practice",
            "IS 456:2000": "Plain and Reinforced Concrete - Code of Practice",
            "IS 383": "Coarse and Fine Aggregate from Natural Sources",
            "IS 383:2016": "Coarse and Fine Aggregate from Natural Sources",
            "IS 1786": "High Strength Deformed Steel Bars",
            "IS 1786:2008": "High Strength Deformed Steel Bars for Reinforcement",
            "IS 1489": "Portland Pozzolana Cement",
            "IS 1489:2015": "Portland Pozzolana Cement - Specification",
            "IS 8112": "43 Grade Ordinary Portland Cement",
            "IS 8112:2013": "43 Grade Ordinary Portland Cement - Specification",
            "IS 12269": "53 Grade Ordinary Portland Cement",
            "IS 12269:2013": "53 Grade Ordinary Portland Cement - Specification",
            "IS 2386": "Methods of Test for Aggregates",
            "IS 516": "Methods of Test for Concrete",
            "IS 1199": "Methods of Sampling and Analysis of Concrete",
            "IS 10262": "Concrete Mix Proportioning - Guidelines",
            "IS 1077": "Common Burnt Clay Building Bricks",
            "IS 2062": "Hot Rolled Medium and High Tensile Structural Steel",
            "IS 875": "Code of Practice for Design Loads for Buildings",
            "IS 3495": "Methods of Tests of Burnt Clay Bricks",
            "IS 2185": "Concrete Masonry Units",
            "IS 12894": "Fly Ash Bricks",
            "IS 5454": "Clay Building Tiles",
        }
    
    def generate_response(
        self, 
        query: str, 
        retrieved_chunks: List[Dict],
        max_standards: int = 5
    ) -> List[Dict]:
        """Generate response using rule-based approach"""
        
        if not retrieved_chunks:
            # Return default standards based on query keywords
            return self._get_default_standards(query)
        
        # Extract unique standards
        standards = []
        seen_ids = set()
        
        for chunk in retrieved_chunks:
            standard_id = chunk.get("standard_id", "Unknown")
            
            # Normalize standard ID to prevent duplicates (e.g., IS_383_2016 or IS 383:2016 -> IS 383)
            import re
            match = re.search(r'IS[\s_]*(\d+)', standard_id.upper())
            if match:
                base_id = f"IS {match.group(1)}"
            else:
                base_id = standard_id.split(':')[0].strip()
            
            if base_id != "Unknown" and base_id not in seen_ids:
                seen_ids.add(base_id)
                
                # Get description from DB or use chunk text
                description = self.standards_db.get(standard_id, "")
                if not description:
                    description = self.standards_db.get(base_id, "")
                if not description:
                    description = f"Technical Specification for {base_id}"
                
                # Short, clean reasoning text specific to the standard
                reason = f"[{description}] Applicable for {description.lower().replace(' - specification', '').replace(' - code of practice', '')}."
                
                standards.append({
                    "standard": standard_id,
                    "reason": reason,
                    "confidence": chunk.get("confidence", 0.0)
                })
                
                if len(standards) >= max_standards:
                    break
        
        # Priority 1: Always Return Top-3 Results
        if len(standards) < 3:
            defaults = self._get_default_standards(query, 3)
            for d in defaults:
                base_id = d["standard"].split(':')[0].strip()
                if base_id not in seen_ids:
                    d["confidence"] = 0.50  # default fallback confidence
                    d["reason"] = f"[{d['reason']}] Applicable for {d['reason'].lower().replace(' - specification', '').replace(' - code of practice', '')}."
                    standards.append(d)
                    seen_ids.add(base_id)
                if len(standards) >= 3:
                    break
        
        return standards[:max_standards]
    
    def _get_default_standards(self, query: str, max_standards: int = 5) -> List[Dict]:
        """Get default standards based on query keywords"""
        query_lower = query.lower()
        defaults = []
        
        # Map keywords to standards
        keyword_map = {
            "cement": [
                {"standard": "IS 269", "reason": "Ordinary Portland Cement - Specification"},
                {"standard": "IS 8112", "reason": "43 Grade Ordinary Portland Cement"},
                {"standard": "IS 456", "reason": "Plain and Reinforced Concrete - Code of Practice"}
            ],
            "steel": [
                {"standard": "IS 1786", "reason": "High Strength Deformed Steel Bars for Reinforcement"},
                {"standard": "IS 456", "reason": "Plain and Reinforced Concrete - Code of Practice"}
            ],
            "concrete": [
                {"standard": "IS 456", "reason": "Plain and Reinforced Concrete - Code of Practice"},
                {"standard": "IS 10262", "reason": "Concrete Mix Proportioning - Guidelines"}
            ],
            "aggregate": [
                {"standard": "IS 383", "reason": "Coarse and Fine Aggregate from Natural Sources"}
            ],
            "sand": [
                {"standard": "IS 383", "reason": "Coarse and Fine Aggregate from Natural Sources"}
            ],
            "pozzolana": [
                {"standard": "IS 1489", "reason": "Portland Pozzolana Cement - Specification"}
            ],
            "tmt": [
                {"standard": "IS 1786", "reason": "High Strength Deformed Steel Bars for Reinforcement"}
            ],
            "reinforcement": [
                {"standard": "IS 1786", "reason": "High Strength Deformed Steel Bars for Reinforcement"}
            ]
        }
        
        for keyword, standards in keyword_map.items():
            if keyword in query_lower:
                for s in standards:
                    if s not in defaults:
                        defaults.append(s)
                if len(defaults) >= 3:
                    break
        
        # If no keywords matched, return general standards
        if not defaults:
            defaults = [
                {"standard": "IS 456", "reason": "Plain and Reinforced Concrete - Code of Practice"},
                {"standard": "IS 383", "reason": "Coarse and Fine Aggregate from Natural Sources"}
            ]
        
        return defaults[:max_standards]


class LLMGenerator(RAGGenerator):
    """LLM-based generator for more natural responses"""
    
    def __init__(self, llm_api=None):
        super().__init__(llm=llm_api)
        self.llm_api = llm_api
        
    def build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM"""
        prompt = f"""You are a BIS compliance expert. Your task is to recommend applicable Bureau of Indian Standards (BIS) for the given product.

IMPORTANT RULES:
1. ONLY recommend standards that are explicitly mentioned in the provided context
2. NEVER invent or hallucinate any standard numbers
3. If no relevant standard is found, state that clearly
4. Provide brief reasons for each recommendation

Product: {query}

Context (from BIS documents):
{context}

Return the top 3-5 most relevant BIS standards in the following JSON format:
[
  {{"standard": "IS XXXX:YYYY", "reason": "Why this applies"}},
  ...
]

If no relevant standard is found, return:
[{{"standard": "No applicable BIS standard found", "reason": "No matching standard in database"}}]
"""
        return prompt
    
    def generate_with_llm(
        self, 
        query: str, 
        retrieved_chunks: List[Dict]
    ) -> List[Dict]:
        """Generate response using LLM API"""
        
        if not self.llm_api:
            logger.warning("No LLM API configured, using simple generator")
            return self.generate_response(query, retrieved_chunks)
        
        # Build context from chunks
        context = "\n\n".join([
            chunk.get("text", "")[:500] for chunk in retrieved_chunks[:5]
        ])
        
        # Build prompt
        prompt = self.build_prompt(query, context)
        
        try:
            # Call LLM API
            response = self.llm_api(prompt)
            
            # Parse response
            standards = self._parse_llm_response(response)
            return standards
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to simple generation
            return self.generate_response(query, retrieved_chunks)
    
    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract standards"""
        try:
            # Try to find JSON in response
            import json
            
            # Find JSON array
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            
            if json_match:
                standards = json.loads(json_match.group())
                return standards
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            
        return []


if __name__ == "__main__":
    # Test generator
    generator = SimpleGenerator()
    
    test_chunks = [
        {"text": "Cement conforming to IS 269 for ordinary portland cement", "standard_id": "IS 269", "category": "cement"},
        {"text": "Steel reinforcement bars conforming to IS 1786", "standard_id": "IS 1786", "category": "steel"},
        {"text": "Concrete mix design as per IS 456", "standard_id": "IS 456", "category": "concrete"},
    ]
    
    response = generator.generate_response("cement for construction", test_chunks)
    
    print("Generated response:")
    import json
    print(json.dumps(response, indent=2))