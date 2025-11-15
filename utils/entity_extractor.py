"""
Entity Extraction Module using Kimi-K2-Thinking model
Extracts drugs, adverse events, and patient demographics from medical abstracts
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
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

    def _build_extraction_prompt(self) -> str:
        """Build system prompt for entity extraction"""
        return """You are a medical information extraction expert. Your task is to extract specific entities from medical research abstracts.

Extract the following information in JSON format:

1. **Drugs/Medications**:
   - Name of drug or medication
   - Context: The full sentence where the drug is mentioned.

2. **Adverse Events** (including diseases):
   - Event/condition name
   - Severity (if mentioned): mild, moderate, severe, or unknown
   - Context: The full sentence where the event is mentioned.

3. **Patient Demographics**:
   - Age: Age range or mean age mentioned
   - Gender: Male, Female, Both, or Unknown
   - Ethnicity: If mentioned
   - Sample size: Number of patients/participants

4. **Diseases/Conditions**:
   - List all diseases or medical conditions mentioned

Return the result in this exact JSON structure:
{
  "drugs": [
    {"name": "drug name", "context": "how it's used/mentioned"}
  ],
  "adverse_events": [
    {"event": "event name", "severity": "mild/moderate/severe/unknown", "context": "description"}
  ],
  "demographics": {
    "age": "age range or mean",
    "gender": "Male/Female/Both/Unknown",
    "ethnicity": "ethnicity if mentioned",
    "sample_size": number_of_participants
  },
  "diseases": ["disease1", "disease2"]
}

If information is not available, use empty lists [] or "Unknown" for strings, and 0 for sample_size.
"""

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

            messages = [
                {"role": "system", "content": self.extraction_prompt},
                {"role": "user", "content": f"Extract medical entities from this abstract:\n\n{text}"}
            ]

            # Generate response using the model
            response = self.model(messages, max_new_tokens=1024, temperature=0.1)

            # Extract generated text
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')

                # Find the assistant's response
                if isinstance(generated_text, list):
                    for msg in generated_text:
                        if msg.get('role') == 'assistant':
                            assistant_response = msg.get('content', '')
                            break
                    else:
                        assistant_response = str(generated_text)
                else:
                    assistant_response = generated_text

                # Parse JSON from response
                entities_dict = self._parse_llm_response(assistant_response)
                return self._dict_to_entities(entities_dict)
            else:
                logger.error("Unexpected model response format")
                return self._extract_rule_based(text)

        except Exception as e:
            logger.error(f"Error during LLM extraction: {e}")
            logger.info("Falling back to rule-based extraction")
            return self._extract_rule_based(text)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Try to find JSON in the response
            # Look for content between ```json and ``` or { and }
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif '{' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
            else:
                json_str = response

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response was: {response}")
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
