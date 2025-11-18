"""
Streamlit Web Interface for PubMed Research Agent
"""

import streamlit as st
import json
import os
import re
import html
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# GGUF 모델 지원 확인
try:
    from agent_gguf import PubMedResearchAgentGGUF
    GGUF_AVAILABLE = True
except:
    GGUF_AVAILABLE = False

# Load environment variables
load_dotenv()

# llama.cpp Metal 경고 숨김
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['GGML_METAL_LOG_LEVEL'] = '0'  # Metal 로그 최소화

# GGUF 모델 스캔 함수
def scan_installed_gguf_models():
    """설치된 GGUF 모델 스캔"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        return []

    gguf_models = []
    for model_dir in cache_dir.glob("models--*"):
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue

            for gguf_file in snapshot.glob("*.gguf"):
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                display_name = gguf_file.name

                gguf_models.append({
                    "display_name": display_name,
                    "model_name": model_name,
                    "filename": gguf_file.name,
                    "path": str(gguf_file),
                    "size_gb": gguf_file.stat().st_size / (1024**3),
                })

    return gguf_models


def highlight_entities_in_text(text: str, entities: dict) -> str:
    """
    Highlight extracted entities in text with different colors.

    Args:
        text: The abstract text
        entities: Dictionary containing drugs, adverse_events, diseases

    Returns:
        HTML string with highlighted entities
    """
    if not text or not entities:
        return text

    # Define colors for different entity types (darker shades)
    colors = {
        'drug': '#90CAF9',  # Darker blue
        'adverse_event': '#EF9A9A',  # Darker red
        'disease': '#FFCC80',  # Darker orange
        'demographics': '#CE93D8'  # Darker purple
    }

    # Collect all entities with their types
    entity_map = []

    # Add drugs
    for drug in entities.get('drugs', []):
        name = drug.get('name', '')
        if name:
            entity_map.append({'text': name, 'type': 'drug', 'label': '💊 Drug'})

    # Add adverse events
    for ae in entities.get('adverse_events', []):
        event = ae.get('event', '')
        if event:
            entity_map.append({'text': event, 'type': 'adverse_event', 'label': '⚠️ AE'})

    # Add diseases
    for disease in entities.get('diseases', []):
        if disease:
            entity_map.append({'text': disease, 'type': 'disease', 'label': '🏥 Disease'})

    # Add demographics (age, gender, race, pregnancy, bmi, sample size patterns)
    demo = entities.get('demographics', {})
    if demo:
        # Age with context patterns (e.g., "women aged 30-44", "mean age, 41.3", "mean age 65±10")
        age = demo.get('age', '')
        if age and age != 'Unknown':
            # Extract numeric age value for pattern matching
            age_number = re.search(r'[\d.]+', age)

            # Try to find age with gender/demographic context
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
                        entity_map.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})
                        found_context = True

            # If no context found, just highlight the age value
            if not found_context:
                entity_map.append({'text': age, 'type': 'demographics', 'label': '👤 Demographics'})

        # Gender with age context patterns (e.g., "30-44 women", "women 30-44")
        gender = demo.get('gender', '')
        if gender and gender != 'Unknown' and age and age != 'Unknown':
            gender_age_patterns = [
                rf'\b{re.escape(age)}\s+(?:women|men|males|females)\b',
                rf'\b(?:women|men|males|females)\s+{re.escape(age)}\b',
            ]
            for pattern in gender_age_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    for match in matches:
                        entity_map.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})

        # Gender patterns (e.g., "56.7% were female", "female patients")
        gender = demo.get('gender', '')
        if gender and gender != 'Unknown':
            gender_keywords = ['male', 'female', 'men', 'women']
            for keyword in gender_keywords:
                if keyword in gender.lower():
                    # Find patterns like "X% were female", "X% female", "female patients"
                    gender_patterns = [
                        rf'\d+\.?\d*%?\s+(?:were|was)?\s*{keyword}',
                        rf'{keyword}\s+(?:patients|participants|subjects)',
                        rf'\b{keyword}\b'
                    ]
                    for pattern in gender_patterns:
                        matches = list(re.finditer(pattern, text, re.IGNORECASE))
                        for match in matches:
                            entity_map.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})

        # Race/ethnicity patterns (e.g., "3.6% were Asian", "83.0% were White")
        race = demo.get('race', demo.get('ethnicity', ''))
        if race and race != 'Unknown' and len(race) > 3:
            # Extract individual race terms if multiple races are mentioned
            race_terms = ['Asian', 'Black', 'African American', 'White', 'Hispanic', 'Latino', 'Caucasian',
                          'Native American', 'Pacific Islander', 'Indigenous']

            found_race_patterns = False
            for race_term in race_terms:
                if race_term.lower() in race.lower():
                    # Find patterns like "X% were Asian", "X% Asian"
                    race_patterns = [
                        rf'\d+\.?\d*%?\s+(?:were|was)?\s*{race_term}(?:\s+(?:or\s+)?[A-Za-z\s]+)?',
                        rf'{race_term}\s+(?:patients|participants|subjects)'
                    ]
                    for pattern in race_patterns:
                        matches = list(re.finditer(pattern, text, re.IGNORECASE))
                        for match in matches:
                            entity_map.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})
                            found_race_patterns = True

            # If no specific patterns found, try to match the race value directly
            if not found_race_patterns:
                entity_map.append({'text': race, 'type': 'demographics', 'label': '👤 Demographics'})

        # Pregnancy (if it's a short match, show it)
        pregnancy = demo.get('pregnancy', '')
        if pregnancy and pregnancy != 'Unknown' and len(pregnancy) < 100:
            entity_map.append({'text': pregnancy, 'type': 'demographics', 'label': '👤 Demographics'})

        # BMI (if it's a short match, show it)
        bmi = demo.get('bmi', '')
        if bmi and bmi != 'Unknown' and len(bmi) < 100:
            entity_map.append({'text': bmi, 'type': 'demographics', 'label': '👤 Demographics'})

        # Sample size patterns
        sample_size = demo.get('sample_size', 0)
        # Convert to int if it's a string (e.g., "102" or "n=102")
        if isinstance(sample_size, str):
            # Try to extract number from string like "n=102" or "102"
            number_match = re.search(r'\d+', sample_size)
            if number_match:
                try:
                    sample_size = int(number_match.group(0))
                except ValueError:
                    sample_size = 0
            else:
                sample_size = 0

        if sample_size and sample_size > 0:
            # Try to find the exact phrase in text
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
                        entity_map.append({'text': match.group(0), 'type': 'demographics', 'label': '👤 Demographics'})
                    break

    # Sort by length (longest first) to avoid partial matches
    entity_map.sort(key=lambda x: len(x['text']), reverse=True)

    # Find all occurrences of entities and track occupied positions
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
            overlaps = False
            for occ_start, occ_end in occupied_ranges:
                if not (end <= occ_start or start >= occ_end):
                    overlaps = True
                    break

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


# Page configuration
st.set_page_config(
    page_title="PubMed Research Agent",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Rule-based (Fast)"
if 'agent' not in st.session_state:
    st.session_state.agent = None

# Title and description
st.title("🔬 PubMed Research Agent")
st.markdown("""
Search PubMed articles and extract medical entities including drugs, adverse events, and patient demographics.
""")

# Calculate default dates (current date and 3 months ago)
current_date = datetime.now()
three_months_ago = current_date - timedelta(days=90)

# Sidebar configuration
with st.sidebar:
    st.header("🔍 Search Query")

    # Query input at the top
    query = st.text_area(
        "PubMed Query",
        placeholder="e.g., metformin adverse events type 2 diabetes",
        help="Enter your PubMed search query",
        height=100
    )

    # Model selection
    st.subheader("💡 Entity Extraction Model")

    # 기본 모델 옵션
    model_options = {
        "Rule-based (Fast)": {"type": "rule", "name": None, "use_llm": False}
    }

    # GGUF 모델 스캔 및 추가
    if GGUF_AVAILABLE:
        gguf_models = scan_installed_gguf_models()
        if gguf_models:
            for gguf in gguf_models:
                # Skip TinyLlama - not suitable for medical entity extraction
                if 'tinyllama' in gguf["display_name"].lower():
                    continue

                model_options[gguf["display_name"]] = {
                    "type": "gguf",
                    "path": gguf["path"],
                    "size": f"{gguf['size_gb']:.2f} GB"
                }

    # 기본값 확인
    if st.session_state.selected_model not in model_options:
        st.session_state.selected_model = "Rule-based (Fast)"

    selected_model = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(st.session_state.selected_model),
        help="Rule-based: Fast keyword matching, GGUF: AI-powered extraction (fast & accurate)"
    )

    # Reinitialize agent if model changed or not initialized
    if selected_model != st.session_state.selected_model or st.session_state.agent is None:
        st.session_state.selected_model = selected_model
        pubmed_email = os.getenv("PUBMED_EMAIL")
        model_config = model_options[selected_model]

        with st.spinner(f"Loading {selected_model} model..."):
            try:
                if model_config["type"] == "gguf":
                    # Add safety checks for GGUF model loading
                    import warnings
                    warnings.filterwarnings('ignore')

                    st.session_state.agent = PubMedResearchAgentGGUF(
                        model_path=model_config["path"],
                        use_llm=True,
                        n_gpu_layers=1,
                        pubmed_email=pubmed_email
                    )
                    st.success(f"✅ {selected_model} loaded! ({model_config['size']})")
                else:
                    # Rule-based
                    st.session_state.agent = PubMedResearchAgentGGUF(
                        use_llm=False,
                        pubmed_email=pubmed_email
                    )
                    st.success("✅ Rule-based extraction ready!")
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                st.error("Try selecting 'Rule-based (Fast)' or check the error logs")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.agent = None

    # Publication date filter
    st.subheader("📅 Publication Date")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=three_months_ago,
            min_value=datetime(1900, 1, 1),
            max_value=current_date,
            help="Filter articles from this date onwards"
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=current_date,
            min_value=datetime(1900, 1, 1),
            max_value=current_date,
            help="Filter articles up to this date"
        )

    max_results = st.slider(
        "Max Search Results",
        min_value=1,
        max_value=50,
        value=5,
        step=5,
        help="Maximum number of articles to fetch"
    )

    # Search button
    search_button = st.button("🔍 Search PubMed", type="primary", use_container_width=True)

    # Entity legend below search button
    st.markdown(
        '<div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">'
        '<div style="font-weight: bold; margin-bottom: 8px; font-size: 0.9em;">Entity Color Legend:</div>'
        '<div style="font-size: 0.85em; line-height: 2.2;">'
        '<span style="background-color: #90CAF9; padding: 2px 6px; border-radius: 3px;">💊 Drug</span><br>'
        '<span style="background-color: #EF9A9A; padding: 2px 6px; border-radius: 3px;">⚠️ Adverse Event</span><br>'
        '<span style="background-color: #FFCC80; padding: 2px 6px; border-radius: 3px;">🏥 Disease</span><br>'
        '<span style="background-color: #CE93D8; padding: 2px 6px; border-radius: 3px;">👤 Demographics</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# Process search
if search_button:
    if not query or not query.strip():
        st.error("❌ Please enter a search query in the sidebar.")
    elif st.session_state.agent is None:
        st.error("❌ Agent not initialized. Please select a model in the sidebar.")
    else:
        st.session_state.current_page = 1  # Reset to first page

        try:
            # Convert dates to YYYY/MM/DD format for PubMed API
            start_date_str = start_date.strftime("%Y/%m/%d")
            end_date_str = end_date.strftime("%Y/%m/%d")

            # First, search PubMed for articles (sorted by Most Recent from API)
            with st.spinner(f"🔍 Searching PubMed for: '{query}'..."):
                articles = st.session_state.agent.searcher.search(
                    query=query,
                    max_results=max_results,
                    max_records=None,
                    rerank=None,  # No reranking - use API's Most Recent sort
                    start_date=start_date_str,
                    end_date=end_date_str
                )

            if not articles:
                st.warning("No articles found for this query.")
            else:
                st.success(f"✅ Found {len(articles)} articles! Now extracting entities...")

                # Initialize results
                results = {
                    "query": query,
                    "total_articles": len(articles),
                    "articles": articles,
                    "entities": []
                }

                # Create tabs at the beginning
                tab1, tab2, tab3 = st.tabs(["📄 Articles", "🧬 Entities", "💾 Export"])

                # Add placeholder messages to Entities and Export tabs
                with tab2:
                    entities_placeholder = st.empty()
                    entities_placeholder.info("⏳ Processing in progress... Entity statistics will be updated once all articles are processed.")

                with tab3:
                    export_placeholder = st.empty()
                    export_placeholder.info("⏳ Processing in progress... Export options will be available once all articles are processed.")

                # Articles tab content (will be populated during processing)
                with tab1:
                    # Progress bar with timer on same line
                    progress_col1, progress_col2 = st.columns([5, 1])
                    with progress_col1:
                        progress_bar = st.progress(0, text="Starting entity extraction...")
                    with progress_col2:
                        timer_placeholder = st.empty()

                    summary_container = st.empty()  # Placeholder for summary card
                    articles_container = st.container()

                import time
                total_start_time = time.time()
                article_start_time = time.time()

                for i, article in enumerate(articles, 1):
                    # Reset article timer for each new article
                    article_start_time = time.time()

                    # Update progress
                    progress = i / len(articles)
                    with progress_col1:
                        progress_bar.progress(
                            progress,
                            text=f"Processing article {i}/{len(articles)}: {article.get('title', 'Unknown')[:60]}..."
                        )

                    # Extract entities (with real-time timer update)
                    abstract = article.get('abstract', '')
                    if abstract:
                        try:
                            # Show processing timer (starts at 00:00 for each article)
                            with progress_col2:
                                timer_placeholder.markdown(f"<div style='text-align: right; font-size: 13px; color: #666; margin-top: 8px;'>⏱️ 00:00</div>", unsafe_allow_html=True)

                            entities = st.session_state.agent.extractor.extract(abstract)

                            # Show time taken for this article (updates after extraction)
                            article_elapsed = int(time.time() - article_start_time)
                            article_mins = article_elapsed // 60
                            article_secs = article_elapsed % 60
                            with progress_col2:
                                timer_placeholder.markdown(f"<div style='text-align: right; font-size: 13px; color: #666; margin-top: 8px;'>⏱️ {article_mins:02d}:{article_secs:02d}</div>", unsafe_allow_html=True)

                            results["entities"].append({
                                "pmid": article.get('pmid'),
                                "title": article.get('title'),
                                "entities": entities.to_dict()
                            })
                            article_entities = entities.to_dict()
                        except Exception as e:
                            st.error(f"Error extracting entities from article {i}: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                            article_entities = None
                    else:
                        article_entities = None

                    # Show completed article immediately in Articles tab
                    with articles_container:
                        with st.expander(f"✅ [{i}] {article.get('title', 'No title')} ({article.get('year', 'N/A')})", expanded=False):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.markdown(f"**PMID:** {article.get('pmid', 'N/A')}")
                                st.markdown(f"**Journal:** {article.get('journal', 'Unknown')} ({article.get('year', 'N/A')})")
                                if article.get('doi'):
                                    st.markdown(f"**DOI:** [{article['doi']}](https://doi.org/{article['doi']})")
                                if article.get('authors'):
                                    authors = article['authors'][:3]
                                    author_str = ", ".join(authors)
                                    if len(article['authors']) > 3:
                                        author_str += f" et al. ({len(article['authors'])} total)"
                                    st.markdown(f"**Authors:** {author_str}")

                            with col2:
                                if 'referenced_by_count' in article:
                                    st.metric("Citations", article['referenced_by_count'])

                            if abstract:
                                st.markdown("**Abstract:**")
                                if article_entities:
                                    highlighted_abstract = highlight_entities_in_text(abstract, article_entities)
                                    st.markdown(
                                        f'<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">{highlighted_abstract}</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(f'<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">{abstract}</div>', unsafe_allow_html=True)

                            # Entity Counts table
                            if article_entities:
                                st.markdown("---")
                                st.markdown("### 📊 Entity Counts")

                                # Collect all entities with their types
                                entity_rows = []
                                row_num = 1

                                # Count occurrences of each keyword in the abstract
                                abstract_lower = abstract.lower()

                                # Add drugs
                                for drug in article_entities.get('drugs', []):
                                    keyword = drug.get('name', 'Unknown')
                                    count = abstract_lower.count(keyword.lower()) if keyword != 'Unknown' else 1
                                    entity_rows.append({
                                        "No.": row_num,
                                        "Keyword": keyword,
                                        "Entity Type": "Drug",
                                        "Count": count
                                    })
                                    row_num += 1

                                # Add adverse events
                                for ae in article_entities.get('adverse_events', []):
                                    keyword = ae.get('event', 'Unknown')
                                    count = abstract_lower.count(keyword.lower()) if keyword != 'Unknown' else 1
                                    entity_rows.append({
                                        "No.": row_num,
                                        "Keyword": keyword,
                                        "Entity Type": "Adverse Event",
                                        "Count": count
                                    })
                                    row_num += 1

                                # Add diseases
                                for disease in article_entities.get('diseases', []):
                                    count = abstract_lower.count(disease.lower()) if disease else 1
                                    entity_rows.append({
                                        "No.": row_num,
                                        "Keyword": disease,
                                        "Entity Type": "Disease",
                                        "Count": count
                                    })
                                    row_num += 1

                                # Add demographics (only those that were highlighted in the text)
                                demo = article_entities.get('demographics', {})
                                if demo:
                                    # Age patterns (same as highlighting logic)
                                    age = demo.get('age', '')
                                    if age and age != 'Unknown':
                                        age_number = re.search(r'[\d.]+', age)
                                        age_patterns = [
                                            rf'\b(?:women|men|males|females|adults|children|participants|patients|subjects)\s+aged\s+{re.escape(age)}\b',
                                            rf'\b(?:women|men|males|females|adults|children|participants|patients|subjects)\s+\(\s*{re.escape(age)}\s*\)',
                                            rf'\b{re.escape(age)}\s*(?:year|yr)(?:s)?[\s-]*old\s+(?:women|men|males|females|adults|children|participants|patients|subjects)\b',
                                            rf'\b{re.escape(age)}\s+(?:women|men|males|females)\b',
                                            rf'\b(?:women|men|males|females)\s+{re.escape(age)}\b',
                                        ]

                                        if age_number:
                                            age_val = age_number.group(0)
                                            age_patterns.extend([
                                                rf'mean age[,:]?\s*{re.escape(age_val)}\s*(?:\[?SD[,:]?\s*[\d.]+\]?)?(?:\s*years?)?',
                                                rf'median age[,:]?\s*{re.escape(age_val)}\s*(?:\[?(?:IQR|range)[,:]?\s*[\d.\-]+\]?)?(?:\s*years?)?',
                                                rf'age[,:]?\s*{re.escape(age_val)}\s*±\s*[\d.]+(?:\s*years?)?',
                                            ])

                                        for pattern in age_patterns:
                                            matches = list(re.finditer(pattern, abstract, re.IGNORECASE))
                                            for match in matches:
                                                phrase = match.group(0)
                                                count = abstract_lower.count(phrase.lower())
                                                entity_rows.append({
                                                    "No.": row_num,
                                                    "Keyword": phrase,
                                                    "Entity Type": "Demographics",
                                                    "Count": count
                                                })
                                                row_num += 1

                                    # Gender patterns
                                    gender = demo.get('gender', '')
                                    if gender and gender != 'Unknown':
                                        gender_keywords = ['male', 'female', 'men', 'women']
                                        for keyword in gender_keywords:
                                            if keyword in gender.lower():
                                                gender_patterns = [
                                                    rf'\d+\.?\d*%?\s+(?:were|was)?\s*{keyword}',
                                                    rf'{keyword}\s+(?:patients|participants|subjects)',
                                                ]
                                                for pattern in gender_patterns:
                                                    matches = list(re.finditer(pattern, abstract, re.IGNORECASE))
                                                    for match in matches:
                                                        phrase = match.group(0)
                                                        count = abstract_lower.count(phrase.lower())
                                                        entity_rows.append({
                                                            "No.": row_num,
                                                            "Keyword": phrase,
                                                            "Entity Type": "Demographics",
                                                            "Count": count
                                                        })
                                                        row_num += 1

                                    # Race/ethnicity patterns
                                    race = demo.get('race', demo.get('ethnicity', ''))
                                    if race and race != 'Unknown' and len(race) > 3:
                                        race_terms = ['Asian', 'Black', 'African American', 'White', 'Hispanic', 'Latino', 'Caucasian']
                                        for race_term in race_terms:
                                            if race_term.lower() in race.lower():
                                                race_patterns = [
                                                    rf'\d+\.?\d*%?\s+(?:were|was)?\s*{race_term}(?:\s+(?:or\s+)?[A-Za-z\s]+)?',
                                                    rf'{race_term}\s+(?:patients|participants|subjects)'
                                                ]
                                                for pattern in race_patterns:
                                                    matches = list(re.finditer(pattern, abstract, re.IGNORECASE))
                                                    for match in matches:
                                                        phrase = match.group(0)
                                                        count = abstract_lower.count(phrase.lower())
                                                        entity_rows.append({
                                                            "No.": row_num,
                                                            "Keyword": phrase,
                                                            "Entity Type": "Demographics",
                                                            "Count": count
                                                        })
                                                        row_num += 1

                                    # Sample size patterns
                                    sample_size = demo.get('sample_size', 0)
                                    if sample_size and sample_size > 0:
                                        size_patterns = [
                                            rf'\b{sample_size}\s+(?:patients|participants|subjects)\b',
                                            rf'\bn\s*=\s*{sample_size}\b'
                                        ]
                                        for pattern in size_patterns:
                                            matches = list(re.finditer(pattern, abstract, re.IGNORECASE))
                                            for match in matches:
                                                phrase = match.group(0)
                                                count = abstract_lower.count(phrase.lower())
                                                entity_rows.append({
                                                    "No.": row_num,
                                                    "Keyword": phrase,
                                                    "Entity Type": "Demographics",
                                                    "Count": count
                                                })
                                                row_num += 1
                                            if matches:
                                                break

                                if entity_rows:
                                    # Create DataFrame and display as table
                                    entity_df = pd.DataFrame(entity_rows)
                                    st.dataframe(entity_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No entities extracted from this article.")

                # Finalize progress and show total time
                total_elapsed = int(time.time() - total_start_time)
                total_mins = total_elapsed // 60
                total_secs = total_elapsed % 60

                with progress_col1:
                    progress_bar.progress(1.0, text="✅ All articles processed!")
                with progress_col2:
                    timer_placeholder.markdown(f"<div style='text-align: right; font-size: 13px; color: #10b981; margin-top: 8px; font-weight: bold;'>✅ {total_mins:02d}:{total_secs:02d}</div>", unsafe_allow_html=True)

                # Store results in session state
                st.session_state.results = results
                st.session_state.search_history.append({
                    'query': query,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'num_articles': len(results['articles'])
                })

                # Calculate metrics
                unique_drugs = len(set(
                    drug.get('name', '')
                    for entity in results['entities']
                    for drug in entity['entities'].get('drugs', [])
                    if drug.get('name')
                )) if results.get('entities') else 0

                unique_aes = len(set(
                    ae.get('event', '')
                    for entity in results['entities']
                    for ae in entity['entities'].get('adverse_events', [])
                    if ae.get('event')
                )) if results.get('entities') else 0

                unique_diseases = len(set(
                    disease
                    for entity in results['entities']
                    for disease in entity['entities'].get('diseases', [])
                    if disease
                )) if results.get('entities') else 0

                # Count unique demographics
                unique_demographics = len(set(
                    f"{demo.get('age', '')}_{demo.get('gender', '')}_{demo.get('race', '')}"
                    for entity in results['entities']
                    for demo in [entity['entities'].get('demographics', {})]
                    if demo and (demo.get('age') or demo.get('gender') or demo.get('race'))
                )) if results.get('entities') else 0

                # Summary Card in Articles tab (light design)
                with summary_container:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; margin: 20px 0;">
                        <h3 style="margin: 0 0 15px 0; color: #495057;">📊 Analysis Summary: {results['query']}</h3>
                        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                            <div style="flex: 1; min-width: 140px; background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #60a5fa; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <div style="font-size: 13px; color: #6c757d; margin-bottom: 5px;">Total Articles</div>
                                <div style="font-size: 28px; font-weight: bold; color: #212529;">{results['total_articles']}</div>
                            </div>
                            <div style="flex: 1; min-width: 140px; background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #a78bfa; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <div style="font-size: 13px; color: #6c757d; margin-bottom: 5px;">Unique Drugs</div>
                                <div style="font-size: 28px; font-weight: bold; color: #212529;">{unique_drugs}</div>
                            </div>
                            <div style="flex: 1; min-width: 140px; background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #f87171; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <div style="font-size: 13px; color: #6c757d; margin-bottom: 5px;">Adverse Events</div>
                                <div style="font-size: 28px; font-weight: bold; color: #212529;">{unique_aes}</div>
                            </div>
                            <div style="flex: 1; min-width: 140px; background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #fbbf24; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <div style="font-size: 13px; color: #6c757d; margin-bottom: 5px;">Diseases</div>
                                <div style="font-size: 28px; font-weight: bold; color: #212529;">{unique_diseases}</div>
                            </div>
                            <div style="flex: 1; min-width: 140px; background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #c4b5fd; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <div style="font-size: 13px; color: #6c757d; margin-bottom: 5px;">Demographics</div>
                                <div style="font-size: 28px; font-weight: bold; color: #212529;">{unique_demographics}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Update Entities tab (clear placeholder first)
                entities_placeholder.empty()
                with tab2:
                    if results.get('entities'):
                        st.subheader("Entity Statistics")

                        # Aggregate entities
                        all_drugs = {}
                        all_aes = {}
                        all_diseases = {}
                        all_genders = {}
                        all_races = {}
                        all_ages = {}

                        for entity_data in results['entities']:
                            entities = entity_data['entities']

                            # Drugs
                            for drug in entities.get('drugs', []):
                                name = drug.get('name', '')
                                if name:
                                    all_drugs[name] = all_drugs.get(name, 0) + 1

                            # Adverse events
                            for ae in entities.get('adverse_events', []):
                                event = ae.get('event', '')
                                if event:
                                    all_aes[event] = all_aes.get(event, 0) + 1

                            # Diseases
                            for disease in entities.get('diseases', []):
                                if disease:
                                    all_diseases[disease] = all_diseases.get(disease, 0) + 1

                            # Demographics
                            demo = entities.get('demographics', {})
                            if demo:
                                gender = demo.get('gender', 'Unknown')
                                if gender and gender != 'Unknown':
                                    all_genders[gender] = all_genders.get(gender, 0) + 1

                                race = demo.get('race', demo.get('ethnicity', 'Unknown'))
                                if race and race != 'Unknown':
                                    all_races[race] = all_races.get(race, 0) + 1

                                age = demo.get('age', 'Unknown')
                                if age and age != 'Unknown':
                                    all_ages[age] = all_ages.get(age, 0) + 1

                        # Create visualizations
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 💊 Top Drugs")
                            if all_drugs:
                                drug_df = pd.DataFrame(list(all_drugs.items()), columns=['Drug', 'Count']).sort_values(by='Count', ascending=False).head(10)
                                fig_bar = px.bar(drug_df, x='Count', y='Drug', orientation='h', title='Top 10 Drugs')
                                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig_bar, use_container_width=True)
                            else:
                                st.info("No drugs extracted.")

                            st.markdown("### 🏥 Top Diseases")
                            if all_diseases:
                                disease_df = pd.DataFrame(list(all_diseases.items()), columns=['Disease', 'Count']).sort_values(by='Count', ascending=False).head(10)
                                fig_bar = px.bar(disease_df, x='Count', y='Disease', orientation='h', title='Top 10 Diseases')
                                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig_bar, use_container_width=True)
                            else:
                                st.info("No diseases extracted.")

                        with col2:
                            st.markdown("### ⚠️ Top Adverse Events")
                            if all_aes:
                                ae_df = pd.DataFrame(list(all_aes.items()), columns=['Adverse Event', 'Count']).sort_values(by='Count', ascending=False).head(10)
                                fig_bar = px.bar(ae_df, x='Count', y='Adverse Event', orientation='h', title='Top 10 Adverse Events')
                                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig_bar, use_container_width=True)
                            else:
                                st.info("No adverse events extracted.")

                            st.markdown("### 👤 Demographics")
                            if all_genders:
                                gender_df = pd.DataFrame(list(all_genders.items()), columns=['Gender', 'Count'])
                                fig_pie = px.pie(gender_df, values='Count', names='Gender', title='Gender Distribution')
                                st.plotly_chart(fig_pie, use_container_width=True)

                            if all_ages:
                                age_df = pd.DataFrame(list(all_ages.items()), columns=['Age', 'Count']).sort_values(by='Count', ascending=False).head(10)
                                fig_bar = px.bar(age_df, x='Count', y='Age', orientation='h', title='Age Distribution')
                                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig_bar, use_container_width=True)

                            if all_races:
                                race_df = pd.DataFrame(list(all_races.items()), columns=['Race/Ethnicity', 'Count']).sort_values(by='Count', ascending=False).head(10)
                                fig_bar = px.bar(race_df, x='Count', y='Race/Ethnicity', orientation='h', title='Race/Ethnicity Distribution')
                                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("No entities extracted.")

                # Update Export tab (clear placeholder first)
                export_placeholder.empty()
                with tab3:
                    st.subheader("Export Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📊 CSV Data**")
                        # Convert results to CSV format
                        csv_rows = []
                        for article in results['articles']:
                            # Find corresponding entities
                            article_entities = None
                            if results.get('entities'):
                                for entity_data in results['entities']:
                                    if entity_data.get('pmid') == article.get('pmid'):
                                        article_entities = entity_data.get('entities', {})
                                        break

                            # Extract entity information
                            drugs = ', '.join([d.get('name', '') for d in article_entities.get('drugs', [])]) if article_entities else ''
                            adverse_events = ', '.join([ae.get('event', '') for ae in article_entities.get('adverse_events', [])]) if article_entities else ''
                            diseases = ', '.join(article_entities.get('diseases', [])) if article_entities else ''

                            demo = article_entities.get('demographics', {}) if article_entities else {}
                            age = demo.get('age', 'Unknown')
                            gender = demo.get('gender', 'Unknown')
                            race = demo.get('race', 'Unknown')
                            sample_size = demo.get('sample_size', 0)

                            csv_rows.append({
                                'PMID': article.get('pmid', ''),
                                'Title': article.get('title', ''),
                                'Journal': article.get('journal', ''),
                                'Year': article.get('year', ''),
                                'DOI': article.get('doi', ''),
                                'Authors': ', '.join(article.get('authors', [])[:3]),
                                'Citations': article.get('referenced_by_count', 0),
                                'Drugs': drugs,
                                'Adverse Events': adverse_events,
                                'Diseases': diseases,
                                'Age': age,
                                'Gender': gender,
                                'Race': race,
                                'Sample Size': sample_size
                            })

                        import csv
                        import io
                        csv_buffer = io.StringIO()
                        if csv_rows:
                            writer = csv.DictWriter(csv_buffer, fieldnames=csv_rows[0].keys())
                            writer.writeheader()
                            writer.writerows(csv_rows)
                            csv_data = csv_buffer.getvalue()
                        else:
                            csv_data = "No data available"

                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"pubmed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_csv",
                            use_container_width=True
                        )

                    with col2:
                        st.markdown("**📄 Text Report**")
                        report = st.session_state.agent.generate_report(results)
                        st.download_button(
                            label="📥 Download Text Report",
                            data=report,
                            file_name=f"pubmed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key="download_txt",
                            use_container_width=True
                        )

                    # Text Report Preview
                    st.markdown("---")
                    st.markdown("### 📄 Text Report Preview")
                    with st.expander("Click to view full text report", expanded=False):
                        st.text(report)

        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Old results display section removed - now handled during processing

# Footer
st.divider()
st.caption("PubMed Research Agent - Powered by NCBI E-utilities API")
