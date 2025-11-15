"""
Entity Extraction Module using Kimi-K2-Thinking model
Extracts drugs, adverse events, and patient demographics from medical abstracts
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Set logging level from environment variable, default to INFO
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Data class for extracted medical entities"""
    drugs: List[Dict[str, str]]  # [{name: str, context: str}]
    adverse_events: List[Dict[str, str]]  # [{event: str, severity: str, context: str}]
    demographics: Dict[str, Any]  # {age: str, gender: str, ethnicity: str, sample_size: int}
    diseases: List[str]  # List of diseases/conditions mentioned

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class EntityExtractor:
    """Extract medical entities from text using LLM"""

    def __init__(self, model_pipeline=None):
        """
        Initialize entity extractor.

        Args:
            model_pipeline: Transformers pipeline for text generation
                          (e.g., Kimi-K2-Thinking model)
        """
        self.model = model_pipeline
        self.extraction_prompt = self._build_extraction_prompt()

        # Detect model type for special handling
        self.model_name = None
        if model_pipeline is not None:
            try:
                self.model_name = model_pipeline.model.config._name_or_path
                logger.info(f"Detected model: {self.model_name}")
            except:
                logger.debug("Could not detect model name")

    def _build_extraction_prompt(self) -> str:
        """Build system prompt for entity extraction"""
        # Load extraction prompt from file
        prompt_file = Path(__file__).parent.parent / "entity_extraction.prompt"
        try:
            prompt = prompt_file.read_text(encoding='utf-8').strip()
            logger.debug(f"Loaded extraction prompt from {prompt_file}")
            return prompt
        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            # Fallback to default prompt
            return """Extract medical entities from the abstract in JSON format:
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

    def extract(self, text: str, use_model: bool = True) -> ExtractedEntities:
        """
        Extract entities from medical text.

        Args:
            text: Medical abstract or text to extract from
            use_model: Whether to use LLM (True) or fallback to rule-based (False)

        Returns:
            ExtractedEntities object containing all extracted information
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return self._empty_entities()

        if use_model and self.model is not None:
            return self._extract_with_llm(text)
        else:
            logger.info("Using rule-based extraction (LLM not available)")
            return self._extract_rule_based(text)

    def _extract_with_llm(self, text: str) -> ExtractedEntities:
        """Extract entities using LLM"""
        try:
            logger.info("Extracting entities with LLM...")

            # Format prompt based on model type
            # JSL-MedLlama models may need special formatting
            if self.model_name and 'JSL-MedLlama' in self.model_name:
                logger.info("Using JSL-MedLlama specific prompt format")
                # Concise format for JSL-MedLlama
                full_prompt = f"""### Instruction:
{self.extraction_prompt}

### Input:
{text}

### Response:
"""
            else:
                # Standard format for other models
                full_prompt = f"""{self.extraction_prompt}

Abstract:
{text}"""

            # Generate response using the model
            logger.debug(f"Calling model with prompt length: {len(full_prompt)}")

            # Call the pipeline with proper parameters for text-generation
            # Note: return_full_text=True because some models don't work well with False
            # Reduced max_new_tokens for faster generation (JSON is usually < 512 tokens)
            response = self.model(
                full_prompt,
                max_new_tokens=512,  # Reduced from 1024 for speed
                temperature=0.1,
                do_sample=True,
                return_full_text=True,  # Include the prompt, we'll strip it later
                pad_token_id=self.model.tokenizer.eos_token_id,  # Avoid warnings
                repetition_penalty=1.1  # Prevent repetitive output
            )

            logger.debug(f"Model response type: {type(response)}")

            # Extract generated text from response
            if isinstance(response, list) and len(response) > 0:
                # response is a list of dicts with 'generated_text' key
                full_text = response[0].get('generated_text', '')
                logger.debug(f"Full text length: {len(full_text)} chars")

                # Check if full_text is empty
                if not full_text or len(full_text.strip()) == 0:
                    logger.error("Model returned empty generated_text!")
                    logger.error(f"Raw response[0]: {response[0]}")
                    return self._extract_rule_based(text)

                # Remove the prompt from the response (since return_full_text=True)
                # The generated part comes after the prompt
                if full_text.startswith(full_prompt):
                    generated_text = full_text[len(full_prompt):].strip()
                    logger.debug("Stripped prompt from response")
                else:
                    # If prompt is not at the start, use the full text
                    generated_text = full_text
                    logger.debug("Could not find prompt in response, using full text")

                logger.info(f"Generated text length: {len(generated_text)} chars")

                # Check if generated_text is empty after stripping prompt
                if not generated_text or len(generated_text.strip()) == 0:
                    logger.error("Generated text is empty after stripping prompt!")
                    logger.error(f"Full text (first 500 chars): {full_text[:500]}")
                    return self._extract_rule_based(text)

                logger.info(f"Generated text preview:\n{generated_text[:500]}\n{'...' if len(generated_text) > 500 else ''}")

                # Parse JSON from response
                entities_dict = self._parse_llm_response(generated_text)

                # Check if parsing was successful
                if entities_dict and (entities_dict.get('drugs') or entities_dict.get('adverse_events') or entities_dict.get('diseases')):
                    logger.info(f"Successfully extracted entities with LLM: {len(entities_dict.get('drugs', []))} drugs, {len(entities_dict.get('adverse_events', []))} AEs")
                    return self._dict_to_entities(entities_dict)
                else:
                    logger.warning("LLM returned empty entities, falling back to rule-based")
                    return self._extract_rule_based(text)
            else:
                logger.error(f"Unexpected model response format: {type(response)}")
                logger.debug(f"Response: {response}")
                return self._extract_rule_based(text)

        except Exception as e:
            logger.error(f"Error during LLM extraction: {e}", exc_info=True)
            logger.info("Falling back to rule-based extraction")
            return self._extract_rule_based(text)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Check if response is empty
            if not response or len(response.strip()) == 0:
                logger.error("Empty response received for JSON parsing")
                return self._empty_entities_dict()

            # Try to find JSON in the response
            # Look for content between ```json and ``` or { and }
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
                logger.warning("No JSON markers found in response, using full response")
                logger.debug(f"Full response: {response}")
                json_str = response

            if not json_str or len(json_str.strip()) == 0:
                logger.error("Extracted JSON string is empty")
                logger.error(f"Original response: {response[:500]}")
                return self._empty_entities_dict()

            logger.debug(f"Attempting to parse JSON (length: {len(json_str)} chars, first 300 chars): {json_str[:300]}")
            parsed = json.loads(json_str)
            logger.info(f"Successfully parsed JSON with {len(parsed.get('drugs', []))} drugs, {len(parsed.get('adverse_events', []))} adverse events")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Full response (first 1000 chars): {response[:1000]}")
            return self._empty_entities_dict()

    def _extract_rule_based(self, text: str) -> ExtractedEntities:
        """
        Fallback rule-based extraction using keyword matching.
        This is a simple implementation - can be enhanced with spaCy/scispacy.
        """
        logger.info("Using rule-based entity extraction")

        text_lower = text.lower()
        import re

        # Simple keyword-based extraction
        drugs = []
        adverse_events = []
        diseases = []
        demographics = {
            "age": "Unknown",
            "gender": "Unknown",
            "ethnicity": "Unknown",
            "sample_size": 0
        }

        # Common drug names and patterns
        drug_patterns = [
            r'\b(metformin|aspirin|insulin|warfarin|lisinopril|atorvastatin|levothyroxine|amlodipine)\b',
            r'\b(omeprazole|simvastatin|losartan|gabapentin|hydrochlorothiazide|albuterol|furosemide)\b',
            r'\b(tramadol|prednisone|amoxicillin|pantoprazole|rosuvastatin|acetaminophen|ibuprofen)\b',
            r'\b([a-z]+mab|[a-z]+nib|[a-z]+pril|[a-z]+sartan|[a-z]+statin|[a-z]+olol)\b'  # Drug suffixes
        ]

        for pattern in drug_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                drug_name = match.group(0).capitalize()
                if not any(d['name'] == drug_name for d in drugs):
                    drugs.append({
                        "name": drug_name,
                        "context": self._extract_context(text, match.start(), match.end())
                    })

        # Common adverse events and symptoms
        ae_patterns = [
            (r'\b(headache|nausea|vomiting|dizziness|fatigue|pain|fever)\b', 'mild'),
            (r'\b(diarrhea|constipation|insomnia|rash|itching|dry mouth)\b', 'mild'),
            (r'\b(hypotension|hypertension|tachycardia|bradycardia|arrhythmia)\b', 'moderate'),
            (r'\b(myocardial infarction|stroke|seizure|anaphylaxis|death|mortality)\b', 'severe'),
            (r'\b(bleeding|hemorrhage|thrombosis|embolism|nephrotoxicity|hepatotoxicity)\b', 'severe'),
        ]

        for pattern, severity in ae_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                event_name = match.group(0).capitalize()
                if not any(ae['event'] == event_name for ae in adverse_events):
                    adverse_events.append({
                        "event": event_name,
                        "severity": severity,
                        "context": self._extract_context(text, match.start(), match.end())
                    })

        # Common diseases
        disease_patterns = [
            r'\b(diabetes|hypertension|cancer|asthma|copd|heart failure|depression)\b',
            r'\b(alzheimer|parkinson|multiple sclerosis|rheumatoid arthritis|osteoporosis)\b',
            r'\b(migraine|epilepsy|schizophrenia|bipolar disorder|obesity|anemia)\b',
            r'\b(pneumonia|tuberculosis|hiv|hepatitis|malaria|covid-19|coronavirus)\b'
        ]

        for pattern in disease_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                disease_name = match.group(0).capitalize()
                if disease_name not in diseases:
                    diseases.append(disease_name)

        # Find sample size
        size_patterns = [
            r'(\d+)\s+patients',
            r'(\d+)\s+participants',
            r'(\d+)\s+subjects',
            r'n\s*=\s*(\d+)',
            r'sample size[:\s]+(\d+)'
        ]

        for pattern in size_patterns:
            match = re.search(pattern, text_lower)
            if match:
                demographics['sample_size'] = int(match.group(1))
                break

        # Detect gender
        if 'male' in text_lower and 'female' in text_lower:
            demographics['gender'] = 'Both'
        elif 'male' in text_lower:
            demographics['gender'] = 'Male'
        elif 'female' in text_lower:
            demographics['gender'] = 'Female'

        # Age detection
        age_patterns = [
            r'age[d]?\s+(\d+\s*±\s*\d+)',
            r'mean age[:\s]+(\d+)',
            r'age[:\s]+(\d+[-–]\d+)',
            r'(\d+)\s+years old',
            r'age[d]?\s+(\d+)\s+years?'
        ]

        for pattern in age_patterns:
            age_match = re.search(pattern, text_lower)
            if age_match:
                demographics['age'] = age_match.group(1)
                break

        return ExtractedEntities(
            drugs=drugs,
            adverse_events=adverse_events,
            demographics=demographics,
            diseases=diseases
        )

    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract surrounding context from text."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].strip()
        return context

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

    def _empty_entities(self) -> ExtractedEntities:
        """Return empty entities structure"""
        return ExtractedEntities(
            drugs=[],
            adverse_events=[],
            demographics={
                "age": "Unknown",
                "gender": "Unknown",
                "ethnicity": "Unknown",
                "sample_size": 0
            },
            diseases=[]
        )

    def _empty_entities_dict(self) -> Dict[str, Any]:
        """Return empty entities as dictionary"""
        return self._empty_entities().to_dict()


# Example usage
if __name__ == "__main__":
    # Example without model (rule-based)
    extractor = EntityExtractor()

    sample_text = """
    A randomized controlled trial of 250 patients (mean age 65±10 years,
    both male and female) with type 2 diabetes treated with metformin 1000mg daily.
    The study observed adverse events including gastrointestinal disturbances (mild to moderate),
    lactic acidosis (severe, n=2), and headache (mild). Glycemic control improved significantly.
    """

    entities = extractor.extract(sample_text, use_model=False)
    print(entities.to_json())
