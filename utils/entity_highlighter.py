"""Entity Highlighting Utility for Medical Abstracts"""

import re
import html
from typing import Dict, List, Tuple


def highlight_entities_in_text(text: str, entities: dict) -> str:
    """
    Highlight extracted entities in text with different colors.

    Args:
        text: The abstract text
        entities: Dictionary containing drugs, adverse_events, demographics

    Returns:
        HTML string with highlighted entities
    """
    if not text or not entities:
        return text

    # Define colors for different entity types (darker shades)
    colors = {
        'drug': '#90CAF9',  # Darker blue
        'adverse_event': '#EF9A9A',  # Darker red
        'demographics': '#CE93D8'  # Darker purple
    }

    # Collect all entities with their types
    entity_map = []

    # Add drugs
    for drug in entities.get('drugs', []):
        name = drug.get('name', '')
        if name:
            entity_map.append({'text': name, 'type': 'drug', 'label': '💊 Drug'})

    # Add adverse events (now includes diseases)
    for ae in entities.get('adverse_events', []):
        event = ae.get('event', '')
        if event:
            entity_map.append({'text': event, 'type': 'adverse_event', 'label': '⚠️ Adverse Event'})

    # Add demographics
    entity_map.extend(_extract_demographics_entities(entities.get('demographics', {}), text))

    # Sort by length (longest first) to avoid partial matches
    entity_map.sort(key=lambda x: len(x['text']), reverse=True)

    # Find all occurrences and apply highlighting
    return _apply_highlighting(text, entity_map, colors)


def _extract_demographics_entities(demo: Dict, text: str) -> List[Dict]:
    """Extract demographics entities from demographics dict"""
    entities = []
    
    if not demo:
        return entities

    # Age patterns
    age = demo.get('age', '')
    if age and age != 'Unknown':
        entities.extend(_extract_age_patterns(age, text))

    # Gender patterns
    gender = demo.get('gender', '')
    if gender and gender != 'Unknown':
        entities.extend(_extract_gender_patterns(gender, text, age))

    # Race/ethnicity patterns
    race = demo.get('race', demo.get('ethnicity', ''))
    if race and race != 'Unknown' and len(race) > 3:
        entities.extend(_extract_race_patterns(race, text))

    # Pregnancy
    pregnancy = demo.get('pregnancy', '')
    if pregnancy and pregnancy != 'Unknown' and len(pregnancy) < 100:
        entities.append({'text': pregnancy, 'type': 'demographics', 'label': '👤 Demographics'})

    # BMI
    bmi = demo.get('bmi', '')
    if bmi and bmi != 'Unknown' and len(bmi) < 100:
        entities.append({'text': bmi, 'type': 'demographics', 'label': '👤 Demographics'})

    # Sample size
    sample_size = demo.get('sample_size', 0)
    if isinstance(sample_size, str):
        number_match = re.search(r'\d+', sample_size)
        if number_match:
            try:
                sample_size = int(number_match.group(0))
            except ValueError:
                sample_size = 0
        else:
            sample_size = 0

    if sample_size and sample_size > 0:
        entities.extend(_extract_sample_size_patterns(sample_size, text))

    return entities


def _extract_age_patterns(age: str, text: str) -> List[Dict]:
    """Extract age-related patterns from text"""
    entities = []
    age_number = re.search(r'[\d.]+', age)

    # Age context patterns
    age_context_patterns = [
        rf'\b(?:women|men|males|females|adults|children|participants|patients|subjects)\s+aged\s+{re.escape(age)}\b',
        rf'\b(?:women|men|males|females|adults|children|participants|patients|subjects)\s+\(\s*{re.escape(age)}\s*\)',
        rf'\b{re.escape(age)}\s*(?:year|yr)(?:s)?[\s-]*old\s+(?:women|men|males|females|adults|children|participants|patients|subjects)\b',
    ]

    # Add mean age patterns if numeric age found
    if age_number:
        age_val = age_number.group(0)
        age_context_patterns.extend([
            rf'mean age[,:]?\s*{re.escape(age_val)}\s*(?:\[?SD[,:]?\s*[\d.]+\]?)?(?:\s*years?)?',
            rf'median age[,:]?\s*{re.escape(age_val)}\s*(?:\[?(?:IQR|range)[,:]?\s*[\d.\-]+\]?)?(?:\s*years?)?',
            rf'age[,:]?\s*{re.escape(age_val)}\s*±\s*[\d.]+(?:\s*years?)?',
            rf'aged?\s*{re.escape(age_val)}\s*±\s*[\d.]+(?:\s*years?)?',
        ])

    found_context = False
    for pattern in age_context_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            for match in matches:
                entities.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})
                found_context = True

    # If no context found, just highlight the age value
    if not found_context:
        entities.append({'text': age, 'type': 'demographics', 'label': '👤 Demographics'})

    return entities


def _extract_gender_patterns(gender: str, text: str, age: str = '') -> List[Dict]:
    """Extract gender-related patterns from text"""
    entities = []

    # Gender with age context patterns
    if age and age != 'Unknown':
        gender_age_patterns = [
            rf'\b{re.escape(age)}\s+(?:women|men|males|females)\b',
            rf'\b(?:women|men|males|females)\s+{re.escape(age)}\b',
        ]
        for pattern in gender_age_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                entities.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})

    # Gender patterns
    gender_keywords = ['male', 'female', 'men', 'women']
    for keyword in gender_keywords:
        if keyword in gender.lower():
            gender_patterns = [
                rf'\d+\.?\d*%?\s+(?:were|was)?\s*{keyword}',
                rf'{keyword}\s+(?:patients|participants|subjects)',
                rf'\b{keyword}\b'
            ]
            for pattern in gender_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    entities.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})

    return entities


def _extract_race_patterns(race: str, text: str) -> List[Dict]:
    """Extract race/ethnicity-related patterns from text"""
    entities = []
    race_terms = ['Asian', 'Black', 'African American', 'White', 'Hispanic', 'Latino', 
                  'Caucasian', 'Native American', 'Pacific Islander', 'Indigenous']

    found_race_patterns = False
    for race_term in race_terms:
        if race_term.lower() in race.lower():
            race_patterns = [
                rf'\d+\.?\d*%?\s+(?:were|was)?\s*{race_term}(?:\s+(?:or\s+)?[A-Za-z\s]+)?',
                rf'{race_term}\s+(?:patients|participants|subjects)'
            ]
            for pattern in race_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    entities.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})
                    found_race_patterns = True

    # If no specific patterns found, try to match the race value directly
    if not found_race_patterns:
        entities.append({'text': race, 'type': 'demographics', 'label': '👤 Demographics'})

    return entities


def _extract_sample_size_patterns(sample_size: int, text: str) -> List[Dict]:
    """Extract sample size patterns from text"""
    entities = []
    size_patterns = [
        rf'\b{sample_size}\s+patients\b',
        rf'\b{sample_size}\s+participants\b',
        rf'\b{sample_size}\s+subjects\b',
        rf'\bn\s*=\s*{sample_size}\b'
    ]
    
    for pattern in size_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            for match in matches:
                entities.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})
            break

    return entities


def _apply_highlighting(text: str, entity_map: List[Dict], colors: Dict[str, str]) -> str:
    """Apply highlighting to entities in text"""
    replacements = []
    occupied_ranges = []

    # Find all occurrences of entities
    for entity_info in entity_map:
        entity_text = entity_info['text']
        entity_type = entity_info['type']
        label = entity_info['label']
        color = colors.get(entity_type, '#E0E0E0')

        # Case-insensitive search
        pattern = re.compile(re.escape(entity_text), re.IGNORECASE)

        for match in pattern.finditer(text):
            start, end = match.span()

            # Check if this range overlaps with any already occupied range
            overlaps = any(not (end <= occ_start or start >= occ_end) 
                          for occ_start, occ_end in occupied_ranges)

            # Only add if no overlap
            if not overlaps:
                original = match.group()
                escaped_original = html.escape(original)
                escaped_entity_text = html.escape(entity_text)
                replacements.append({
                    'start': start,
                    'end': end,
                    'original': original,
                    'highlighted': f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 0 2px;" title="{label}: {escaped_entity_text}">{escaped_original}</span>'
                })
                occupied_ranges.append((start, end))

    # Sort replacements by start position (descending) to avoid offset issues
    replacements.sort(key=lambda x: x['start'], reverse=True)

    # Apply replacements
    highlighted_text = text
    for repl in replacements:
        highlighted_text = (
            highlighted_text[:repl['start']] +
            repl['highlighted'] +
            highlighted_text[repl['end']:]
        )

    return highlighted_text
