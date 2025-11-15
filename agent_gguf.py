"""
PubMed Research Agent
Uses llama.cpp for fast quantized model inference
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from tools import PubMedSearcher
from utils import ExtractedEntities
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GGUFEntityExtractor:
    """Fast entity extractor using GGUF quantized models via llama.cpp"""

    def __init__(self, model_path: str, n_gpu_layers: int = 1):
        """
        Initialize GGUF entity extractor.

        Args:
            model_path: Path to .gguf model file
            n_gpu_layers: Number of layers to offload to GPU (Metal on Mac)
                         Set to 1 to use Metal, 0 for CPU only
        """
        try:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF model from: {model_path}")

            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # Increased context window (원래 모델의 절반)
                n_gpu_layers=n_gpu_layers,  # Use Metal GPU on Mac
                verbose=False,  # Metal 경고 숨김
                n_threads=4  # CPU 스레드 수 제한
            )
            logger.info("GGUF model loaded successfully")
            self.available = True

        except ImportError:
            logger.error("llama-cpp-python not installed. Install with:")
            logger.error("  CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            self.available = False

        # Load extraction prompt from file
        prompt_file = Path(__file__).parent / "entity_extraction.prompt"
        try:
            self.extraction_prompt = prompt_file.read_text(encoding='utf-8').strip()
            logger.debug(f"Loaded extraction prompt from {prompt_file}")
        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            # Fallback to default prompt
            self.extraction_prompt = """Extract medical entities from the abstract in JSON format:
{
  "drugs": [{"name": "drug name", "context": "how it is used"}],
  "adverse_events": [{"event": "adverse event name", "severity": "mild/moderate/severe/unknown", "context": "details about the event"}],
  "demographics": {"age": "age range or mean age (e.g., 65±10, 18-65)", "gender": "Male/Female/Both/Unknown", "ethnicity": "ethnicity or race if mentioned", "sample_size": 0},
  "diseases": ["disease1", "disease2", "disease3", "disease4", "..."]
}

Instructions:
- Extract ALL drugs mentioned in the text
- Extract ALL adverse events/side effects mentioned
- For demographics: carefully look for age (mean age, age range, median age), gender distribution, ethnicity/race, and sample size (n=X, X patients, X participants, X subjects)
- Extract ALL diseases/conditions mentioned (not limited to 2-3, can be many)
- Return ONLY the JSON object, no explanations or other text"""

    def extract(self, text: str) -> ExtractedEntities:
        """Extract entities from medical text using GGUF model"""
        if not self.available:
            logger.warning("GGUF model not available, using rule-based extraction")
            return self._extract_rule_based(text)

        try:
            logger.info("Extracting entities with GGUF model...")

            # Format prompt
            full_prompt = f"""{self.extraction_prompt}

Abstract:
{text.strip()}

JSON:"""

            # Generate response
            logger.debug(f"Calling GGUF model with prompt length: {len(full_prompt)}")

            response = self.llm(
                full_prompt,
                max_tokens=512,
                temperature=0.1,
                echo=False
            )

            raw_text = response['choices'][0]['text']
            generated_text = raw_text.strip()
            logger.info(f"Generated {len(generated_text)} chars")
            logger.debug(f"Generated text preview: {generated_text[:200]}")

            # Parse JSON
            entities_dict = self._parse_json_response(generated_text)

            if entities_dict and (entities_dict.get('drugs') or entities_dict.get('adverse_events') or entities_dict.get('diseases')):
                entities_result = self._dict_to_entities(entities_dict)
                logger.info(f"Successfully extracted: {len(entities_result.drugs)} drugs, {len(entities_result.adverse_events)} AEs, {len(entities_result.diseases)} diseases")
                return entities_result
            else:
                logger.warning("GGUF model returned empty entities, falling back to rule-based")
                return self._extract_rule_based(text)

        except Exception as e:
            logger.error(f"Error during GGUF extraction: {e}", exc_info=True)
            return self._extract_rule_based(text)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from model response"""
        try:
            # Find JSON object
            if '{' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
            else:
                json_str = response

            parsed = json.loads(json_str)
            logger.info(f"Successfully parsed JSON with {len(parsed.get('drugs', []))} drugs, {len(parsed.get('adverse_events', []))} AEs, {len(parsed.get('diseases', []))} diseases")
            logger.debug(f"Parsed content: {parsed}")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response: {response[:500]}")
            return self._empty_entities_dict()

    def _dict_to_entities(self, data: Dict[str, Any]) -> ExtractedEntities:
        """Convert dictionary to ExtractedEntities"""
        return ExtractedEntities(
            drugs=data.get('drugs', []),
            adverse_events=data.get('adverse_events', []),
            demographics=data.get('demographics', {
                "age": "Unknown",
                "gender": "Unknown",
                "ethnicity": "Unknown",
                "sample_size": 0
            }),
            diseases=data.get('diseases', [])
        )

    def _empty_entities_dict(self) -> Dict[str, Any]:
        """Return empty entities dictionary"""
        return {
            'drugs': [],
            'adverse_events': [],
            'demographics': {
                "age": "Unknown",
                "gender": "Unknown",
                "ethnicity": "Unknown",
                "sample_size": 0
            },
            'diseases': []
        }

    def _extract_rule_based(self, text: str) -> ExtractedEntities:
        """Fallback to rule-based extraction"""
        from utils import EntityExtractor
        extractor = EntityExtractor(model_pipeline=None)
        return extractor.extract(text, use_model=False)


class PubMedResearchAgentGGUF:
    """PubMed Research Agent using GGUF quantized models"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_llm: bool = True,
        n_gpu_layers: int = 1,
        pubmed_email: Optional[str] = None
    ):
        """
        Initialize the research agent with GGUF model.

        Args:
            model_path: Path to .gguf model file
            use_llm: Whether to use LLM for entity extraction
            n_gpu_layers: Number of layers to offload to GPU (1 for Metal on Mac)
            pubmed_email: Optional email for NCBI API
        """
        logger.info("Initializing PubMed Research Agent (GGUF)...")

        # Initialize PubMed searcher
        self.searcher = PubMedSearcher(email=pubmed_email)

        # Initialize entity extractor
        self.use_llm = use_llm
        if use_llm and model_path:
            self.extractor = GGUFEntityExtractor(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers
            )
        else:
            logger.info("Using rule-based extraction")
            from utils import EntityExtractor
            self.extractor = EntityExtractor(model_pipeline=None)

        logger.info("Agent initialization complete")

    def search_and_extract(
        self,
        query: str,
        max_results: int = 50,
        max_records: int = 10,
        extract_entities: bool = True,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """Search PubMed and extract entities"""
        logger.info(f"Starting research query: {query}")

        # Search PubMed
        articles = self.searcher.search(
            query=query,
            max_results=max_results,
            max_records=max_records,
            rerank="referenced_by",
            start_year=start_year,
            end_year=end_year
        )

        logger.info(f"Found {len(articles)} articles")

        result = {
            "query": query,
            "total_articles": len(articles),
            "articles": articles,
            "entities": []
        }

        # Extract entities if requested
        if extract_entities and articles:
            logger.info("Extracting entities from abstracts...")
            for i, article in enumerate(articles, 1):
                logger.info(f"Processing article {i}/{len(articles)}: {article.get('pmid', 'Unknown')}")

                abstract = article.get('abstract', '')
                if not abstract:
                    continue

                entities = self.extractor.extract(abstract)

                result["entities"].append({
                    "pmid": article.get('pmid'),
                    "title": article.get('title'),
                    "entities": entities.to_dict()
                })

        return result


# Example usage
if __name__ == "__main__":
    import sys

    print("="*80)
    print("PubMed Research Agent - GGUF Fast Inference Demo")
    print("="*80)

    # Check if model path provided
    if len(sys.argv) < 2:
        print("\nUsage: python agent_gguf.py <path-to-gguf-model>")
        print("\nExample:")
        print("  python agent_gguf.py ~/.cache/huggingface/hub/models--TheBloke--BioMistral-7B-GGUF/snapshots/.../biomistral-7b.Q4_K_M.gguf")
        print("\nOr use rule-based extraction:")
        print("  python agent_gguf.py --rule-based")
        sys.exit(1)

    if sys.argv[1] == "--rule-based":
        agent = PubMedResearchAgentGGUF(use_llm=False)
    else:
        model_path = sys.argv[1]
        agent = PubMedResearchAgentGGUF(
            model_path=model_path,
            use_llm=True,
            n_gpu_layers=1  # Use Metal GPU on Mac
        )

    # Test search
    query = "metformin adverse events type 2 diabetes"
    print(f"\nSearching for: {query}")

    results = agent.search_and_extract(
        query=query,
        max_results=10,
        max_records=3,
        extract_entities=True
    )

    print(f"\n✓ Found {results['total_articles']} articles")
    print(f"✓ Extracted entities from {len(results['entities'])} articles")

    # Show first result
    if results['entities']:
        first = results['entities'][0]
        print(f"\nExample extraction from: {first['title'][:60]}...")
        print(f"  Drugs: {len(first['entities']['drugs'])}")
        print(f"  Adverse Events: {len(first['entities']['adverse_events'])}")
        print(f"  Diseases: {len(first['entities']['diseases'])}")
