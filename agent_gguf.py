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
                n_ctx=2048,  # Reduced context window to prevent segfaults
                n_gpu_layers=n_gpu_layers,  # Use Metal GPU on Mac
                verbose=False,  # Metal 경고 숨김
                n_threads=2,  # Reduced CPU threads for stability
                n_batch=512  # Smaller batch size for better stability
            )
            logger.info("GGUF model loaded successfully")
            self.available = True

            # Detect model type from path
            model_path_lower = model_path.lower()
            if 'biomistral' in model_path_lower or 'mistral' in model_path_lower:
                self.model_type = 'mistral'
            elif 'llama' in model_path_lower or 'medllama' in model_path_lower:
                self.model_type = 'llama'
            else:
                self.model_type = 'generic'
            logger.info(f"Detected model type: {self.model_type}")

        except ImportError:
            logger.error("llama-cpp-python not installed. Install with:")
            logger.error("  CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            self.available = False

        # Load extraction prompt from file
        prompt_file = Path(__file__).parent / "config" / "entity_extraction.prompt"
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
  "demographics": {
    "age": "age range or mean age (e.g., 65±10, 18-65)",
    "gender": "Male/Female/Both/Unknown",
    "race": "race or ethnicity if mentioned",
    "pregnancy": "pregnancy status if mentioned (e.g., pregnant, not pregnant, trimester, Unknown)",
    "bmi": "BMI or body mass index if mentioned (e.g., 25.3, overweight, Unknown)",
    "sample_size": 0
  }
}

Instructions:
- Extract ALL drugs mentioned in the text
- Extract ALL adverse events, side effects, diseases, conditions, and medical outcomes mentioned
  - Include both traditional adverse events (e.g., nausea, headache, rash) AND diseases/conditions (e.g., diabetes, hypertension, cancer)
  - All medical conditions should be categorized as adverse events
  - Examples: "type 2 diabetes", "cardiovascular disease", "acute headache", "nausea", "fatigue"
- For demographics, carefully look for:
  - age: mean age, age range, median age (e.g., 65±10 years, 18-65 years)
  - gender: Male, Female, Both, or Unknown
  - race: ethnicity or race if mentioned (e.g., Caucasian, African American, Asian, Hispanic)
  - pregnancy: pregnancy status if mentioned (e.g., "pregnant women", "first trimester", "not pregnant")
  - bmi: body mass index or weight status (e.g., "BMI 28.5", "obese", "overweight")
  - sample_size: n=X, X patients, X participants, X subjects
- Return ONLY the JSON object, no explanations or other text"""

    def extract(self, text: str) -> ExtractedEntities:
        """Extract entities from medical text using GGUF model"""
        if not self.available:
            logger.warning("GGUF model not available, using rule-based extraction")
            return self._extract_rule_based(text)

        try:
            logger.info("Extracting entities with GGUF model...")

            # Construct user message
            user_message = f"""{self.extraction_prompt}

Abstract:
{text.strip()}

JSON:"""

            # Choose prompt format based on model type
            if self.model_type == 'mistral':
                # Mistral/BioMistral format: [INST] ... [/INST]
                # More detailed and structured prompt for better entity extraction
                full_prompt = f"""[INST] You are a medical entity extraction specialist. Your task is to carefully analyze the following medical abstract and extract ALL relevant entities.

INSTRUCTIONS:
1. Extract ALL drug names, medications, vaccines, and treatments mentioned
2. Extract ALL adverse events, side effects, diseases, conditions, and medical outcomes
   - Include BOTH traditional adverse events (nausea, headache) AND diseases/conditions (diabetes, cancer)
   - ALL medical conditions should be categorized as adverse events
   - Examples: "type 2 diabetes", "hypertension", "nausea", "fatigue", "cardiovascular disease"
3. Extract ALL patient demographics (age, gender, race/ethnicity, sample size)
4. Be thorough - extract every relevant entity, even if mentioned only once

Return your findings in this EXACT JSON structure:
{{
  "drugs": [
    {{"name": "exact drug name from text", "context": "how it was used"}}
  ],
  "adverse_events": [
    {{"event": "specific adverse event or condition", "severity": "mild/moderate/severe/unknown", "context": "relevant details"}}
  ],
  "demographics": {{
    "age": "age range or mean±SD",
    "gender": "Male/Female/Both/Unknown",
    "race": "ethnicity information",
    "pregnancy": "pregnancy status if mentioned",
    "bmi": "BMI information if available",
    "sample_size": number_of_participants
  }}
}}

ABSTRACT:
{text.strip()}

RESPOND WITH ONLY THE JSON OBJECT - NO ADDITIONAL TEXT:[/INST]

"""
                stop_tokens = ["</s>", "[INST]"]
            elif self.model_type == 'llama':
                # Llama 3 chat template format
                full_prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a medical entity extraction assistant specialized in extracting entities from medical research abstracts.

IMPORTANT INSTRUCTIONS:
1. Extract ALL drug names, medications, vaccines, treatments, and therapeutic interventions (e.g., "acute headache treatments", "ibuprofen", "chemotherapy")
2. Extract ALL adverse events, side effects, diseases, conditions, and medical outcomes
   - Include BOTH traditional adverse events (nausea, headache) AND diseases/conditions (diabetes, cancer, hypertension)
   - ALL medical conditions should be categorized as adverse events
   - Examples: "type 2 diabetes", "cardiovascular disease", "acute headache", "nausea", "fatigue"
3. Extract ALL patient demographics: age, gender, race/ethnicity, pregnancy status, BMI, sample size
4. Be comprehensive - extract EVERY relevant entity, even if mentioned only once
5. Return ONLY a valid JSON object with NO additional text or explanations

Your response must be in this exact JSON format:
{{
  "drugs": [{{"name": "drug/treatment name", "context": "how it was used"}}],
  "adverse_events": [{{"event": "adverse event or condition", "severity": "mild/moderate/severe/unknown", "context": "details"}}],
  "demographics": {{"age": "age info", "gender": "Male/Female/Both/Unknown", "race": "race/ethnicity", "pregnancy": "pregnancy status", "bmi": "BMI info", "sample_size": number}}
}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
                stop_tokens = ["<|eot_id|>"]
            else:
                # Generic format
                full_prompt = f"""You are a medical entity extraction assistant.

{user_message}

"""
                stop_tokens = ["\n\n\n"]

            # Generate response
            logger.debug(f"Calling GGUF model ({self.model_type}) with prompt length: {len(full_prompt)}")

            response = self.llm(
                full_prompt,
                max_tokens=2048,  # Increased for complete JSON response
                temperature=0.1,
                echo=False,
                stop=stop_tokens,
                repeat_penalty=1.1,  # Prevent repetitive output
                top_p=0.95  # Nucleus sampling for better quality
            )

            generated_text = response['choices'][0]['text'].strip()
            logger.debug(f"Raw response length: {len(generated_text)} chars")

            logger.info(f"Generated {len(generated_text)} chars")
            logger.debug(f"Full generated text:\n{generated_text}")  # Debug only

            # Parse JSON
            entities_dict = self._parse_json_response(generated_text)

            # Convert to entities even if some fields are empty
            # The model attempted extraction, so we should use its results
            if entities_dict is not None:
                entities_result = self._dict_to_entities(entities_dict)
                logger.info(f"Successfully extracted with GGUF: {len(entities_result.drugs)} drugs, {len(entities_result.adverse_events)} AEs, {len(entities_result.diseases)} diseases")

                # If GGUF extraction is completely empty, try rule-based as fallback
                if (len(entities_result.drugs) == 0 and
                    len(entities_result.adverse_events) == 0 and
                    len(entities_result.diseases) == 0):
                    logger.warning("GGUF model returned no entities, falling back to rule-based")
                    return self._extract_rule_based(text)

                return entities_result
            else:
                logger.warning("GGUF model failed to parse response, falling back to rule-based")
                return self._extract_rule_based(text)

        except Exception as e:
            logger.error(f"Error during GGUF extraction: {e}", exc_info=True)
            return self._extract_rule_based(text)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from model response"""
        try:
            # Check if response is empty
            if not response or len(response.strip()) == 0:
                logger.error("Empty response received for JSON parsing")
                return self._empty_entities_dict()

            # Log the raw response for debugging
            logger.debug(f"Parsing response (first 500 chars): {response[:500]}")

            # Try to find JSON in the response
            # Look for content between ```json and ``` or { and }
            json_str = None

            if '```json' in response:
                logger.debug("Found JSON code block with ```json")
                start = response.find('```json') + 7
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif '```' in response:
                logger.debug("Found code block with ```")
                start = response.find('```') + 3
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif '{' in response:
                logger.debug("Found JSON object with {")
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
            else:
                logger.warning("No JSON markers found in response")
                json_str = response

            if not json_str or len(json_str.strip()) == 0:
                logger.error("Extracted JSON string is empty")
                logger.error(f"Original response: {response[:1000]}")
                return self._empty_entities_dict()

            logger.debug(f"Attempting to parse JSON (length: {len(json_str)} chars)")
            parsed = json.loads(json_str)
            logger.info(f"Successfully parsed JSON with {len(parsed.get('drugs', []))} drugs, {len(parsed.get('adverse_events', []))} AEs, {len(parsed.get('diseases', []))} diseases")
            logger.debug(f"Parsed content: {parsed}")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.debug(f"JSON string attempted: {json_str[:1000] if json_str else 'None'}")

            # Try to fix incomplete JSON
            fixed_json = self._try_fix_json(json_str)
            if fixed_json:
                try:
                    parsed = json.loads(fixed_json)
                    logger.info(f"Successfully parsed fixed JSON with {len(parsed.get('drugs', []))} drugs, {len(parsed.get('adverse_events', []))} AEs")
                    return parsed
                except json.JSONDecodeError:
                    logger.error("Failed to parse fixed JSON as well")

            logger.error(f"Full response: {response[:1000]}")
            return self._empty_entities_dict()
        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing: {e}", exc_info=True)
            return self._empty_entities_dict()

    def _try_fix_json(self, json_str: str) -> Optional[str]:
        """Try to fix incomplete or malformed JSON"""
        if not json_str or '{' not in json_str:
            return None

        try:
            # Count braces and brackets
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')

            logger.info(f"JSON repair: {open_braces} {{ vs {close_braces} }}, {open_brackets} [ vs {close_brackets} ]")

            # If JSON starts but is incomplete
            fixed = json_str

            # Close unclosed arrays
            while open_brackets > close_brackets:
                fixed += ']'
                close_brackets += 1

            # Close unclosed objects
            while open_braces > close_braces:
                fixed += '}'
                close_braces += 1

            # If the JSON ends with a comma, might need to add empty fields
            if fixed.rstrip().endswith(','):
                # Add missing fields with defaults
                fixed = fixed.rstrip().rstrip(',')
                # Check what's missing
                if '"adverse_events"' not in fixed:
                    fixed += ',\n  "adverse_events": []'
                if '"demographics"' not in fixed:
                    fixed += ',\n  "demographics": {"age": "Unknown", "gender": "Unknown", "race": "Unknown", "pregnancy": "Unknown", "bmi": "Unknown", "sample_size": 0}'
                if '"diseases"' not in fixed:
                    fixed += ',\n  "diseases": []'

                # Close the object
                if not fixed.endswith('}'):
                    fixed += '\n}'

            logger.info(f"Attempted JSON fix, new length: {len(fixed)}")
            logger.debug(f"Fixed JSON:\n{fixed}")

            return fixed
        except Exception as e:
            logger.error(f"Error during JSON fix attempt: {e}")
            return None

    def _dict_to_entities(self, data: Dict[str, Any]) -> ExtractedEntities:
        """Convert dictionary to ExtractedEntities"""
        # Handle both old (ethnicity) and new (race) field names for backwards compatibility
        demographics_data = data.get('demographics', {})
        if 'ethnicity' in demographics_data and 'race' not in demographics_data:
            demographics_data['race'] = demographics_data.pop('ethnicity')

        # Normalize sample_size to integer
        if demographics_data and 'sample_size' in demographics_data:
            sample_size = demographics_data['sample_size']
            if isinstance(sample_size, str):
                # Try to extract number from string like "n=102", "n = 102", or "102"
                import re
                number_match = re.search(r'\d+', sample_size)
                if number_match:
                    try:
                        demographics_data['sample_size'] = int(number_match.group(0))
                    except ValueError:
                        demographics_data['sample_size'] = 0
                else:
                    demographics_data['sample_size'] = 0
            elif not isinstance(sample_size, int):
                demographics_data['sample_size'] = 0

        return ExtractedEntities(
            drugs=data.get('drugs', []),
            adverse_events=data.get('adverse_events', []),
            demographics=demographics_data if demographics_data else {
                "age": "Unknown",
                "gender": "Unknown",
                "race": "Unknown",
                "pregnancy": "Unknown",
                "bmi": "Unknown",
                "sample_size": 0
            },
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
                "race": "Unknown",
                "pregnancy": "Unknown",
                "bmi": "Unknown",
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
        end_year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search PubMed and extract entities.

        Args:
            query: Search query
            max_results: Maximum results to fetch
            max_records: Maximum records to return
            extract_entities: Whether to extract entities
            start_year: Start year (deprecated - use start_date)
            end_year: End year (deprecated - use end_date)
            start_date: Start date in YYYY/MM/DD format
            end_date: End date in YYYY/MM/DD format

        Returns:
            Dictionary with search results and extracted entities
        """
        logger.info(f"Starting research query: {query}")

        # Search PubMed
        articles = self.searcher.search(
            query=query,
            max_results=max_results,
            max_records=max_records,
            rerank="referenced_by",
            start_year=start_year,
            end_year=end_year,
            start_date=start_date,
            end_date=end_date
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

    def generate_report(self, research_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable research report.

        Args:
            research_data: Output from search_and_extract()

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 100)
        lines.append("PUBMED RESEARCH REPORT")
        lines.append("=" * 100)
        lines.append(f"\nQuery: {research_data['query']}")
        lines.append(f"Total Articles Analyzed: {research_data['total_articles']}\n")

        # Summary of articles
        lines.append("\n" + "=" * 100)
        lines.append("ARTICLES SUMMARY")
        lines.append("=" * 100)

        for i, article in enumerate(research_data['articles'], 1):
            lines.append(f"\n[{i}] {article.get('title', 'No title')}")
            lines.append(f"    PMID: {article.get('pmid', 'N/A')} | "
                        f"Journal: {article.get('journal', 'Unknown')} | "
                        f"Year: {article.get('year', 'N/A')}")

            if article.get('doi'):
                lines.append(f"    DOI: {article['doi']}")

            if 'referenced_by_count' in article:
                lines.append(f"    Citations in dataset: {article['referenced_by_count']}")

        # Entity extraction summary
        if research_data.get('entities'):
            lines.append("\n\n" + "=" * 100)
            lines.append("ENTITY EXTRACTION SUMMARY")
            lines.append("=" * 100)

            # Aggregate all entities
            all_drugs = set()
            all_adverse_events = set()
            all_diseases = set()

            for entity_data in research_data['entities']:
                entities = entity_data['entities']

                # Collect drugs
                for drug in entities.get('drugs', []):
                    all_drugs.add(drug.get('name', ''))

                # Collect adverse events
                for ae in entities.get('adverse_events', []):
                    all_adverse_events.add(ae.get('event', ''))

                # Collect diseases
                for disease in entities.get('diseases', []):
                    if disease:
                        all_diseases.add(disease)

            lines.append(f"\nUnique Drugs Mentioned: {len(all_drugs)}")
            if all_drugs:
                for drug in sorted(all_drugs):
                    if drug:
                        lines.append(f"  - {drug}")

            lines.append(f"\nUnique Adverse Events: {len(all_adverse_events)}")
            if all_adverse_events:
                for ae in sorted(all_adverse_events):
                    if ae:
                        lines.append(f"  - {ae}")

            lines.append(f"\nDiseases/Conditions: {len(all_diseases)}")
            if all_diseases:
                for disease in sorted(all_diseases):
                    lines.append(f"  - {disease}")

            # Detailed per-article extraction
            lines.append("\n\n" + "=" * 100)
            lines.append("DETAILED ENTITY EXTRACTION BY ARTICLE")
            lines.append("=" * 100)

            for i, entity_data in enumerate(research_data['entities'], 1):
                lines.append(f"\n[{i}] {entity_data['title']}")
                lines.append(f"PMID: {entity_data['pmid']}")

                entities = entity_data['entities']

                # Demographics
                demo = entities.get('demographics', {})
                if demo:
                    lines.append(f"\nDemographics:")
                    lines.append(f"  Sample Size: {demo.get('sample_size', 0)}")
                    lines.append(f"  Age: {demo.get('age', 'Unknown')}")
                    lines.append(f"  Gender: {demo.get('gender', 'Unknown')}")
                    if demo.get('ethnicity') and demo['ethnicity'] != 'Unknown':
                        lines.append(f"  Ethnicity: {demo['ethnicity']}")

                # Drugs
                drugs = entities.get('drugs', [])
                if drugs:
                    lines.append(f"\nDrugs ({len(drugs)}):")
                    for drug in drugs:
                        lines.append(f"  - {drug.get('name', 'Unknown')}: {drug.get('context', '')}")

                # Adverse Events
                aes = entities.get('adverse_events', [])
                if aes:
                    lines.append(f"\nAdverse Events ({len(aes)}):")
                    for ae in aes:
                        severity = ae.get('severity', 'unknown')
                        lines.append(f"  - {ae.get('event', 'Unknown')} "
                                   f"[{severity}]: {ae.get('context', '')}")

                lines.append("\n" + "-" * 100)

        lines.append("\n" + "=" * 100)
        lines.append("END OF REPORT")
        lines.append("=" * 100)

        return "\n".join(lines)

    def save_report(self, research_data: Dict[str, Any], filepath: str):
        """Save report to file"""
        report = self.generate_report(research_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Report saved to {filepath}")

    def save_json(self, research_data: Dict[str, Any], filepath: str):
        """Save research data as JSON"""
        # Convert ExtractedEntities to dict if needed
        data_copy = research_data.copy()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_copy, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON data saved to {filepath}")


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
