"""
Streamlit Web Interface for PubMed Research Agent
"""

import streamlit as st
import json
import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from agent import PubMedResearchAgent

# Load environment variables
load_dotenv()


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

    # Define colors for different entity types
    colors = {
        'drug': '#E3F2FD',  # Light blue
        'adverse_event': '#FFEBEE',  # Light red
        'disease': '#FFF3E0',  # Light orange
        'demographics': '#F3E5F5'  # Light purple
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

    # Sort by length (longest first) to avoid partial matches
    entity_map.sort(key=lambda x: len(x['text']), reverse=True)

    # Create a copy of text for highlighting
    highlighted_text = text
    replacements = []

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
            original = match.group()
            # Store replacement info
            replacements.append({
                'start': start,
                'end': end,
                'original': original,
                'highlighted': f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 0 2px;" title="{label}: {entity_text}">{original}</span>'
            })

    # Sort replacements by start position (descending) to avoid offset issues
    replacements.sort(key=lambda x: x['start'], reverse=True)

    # Apply replacements
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

    st.divider()

    # Model selection
    st.subheader("🤖 Entity Extraction Model")
    model_options = {
        "Rule-based (Fast)": {"name": None, "use_llm": False},
        "Kimi-K2-Thinking": {"name": "moonshotai/Kimi-K2-Thinking", "use_llm": True},
        "JSL-MedLlama-3-8B-v2.0": {"name": "johnsnowlabs/JSL-MedLlama-3-8B-v2.0", "use_llm": True}
    }

    selected_model = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(st.session_state.selected_model),
        help="Choose entity extraction method: Rule-based is fast, LLM models are more accurate but slower"
    )

    # Reinitialize agent if model changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        pubmed_email = os.getenv("PUBMED_EMAIL")
        model_config = model_options[selected_model]

        with st.spinner(f"Loading {selected_model} model..."):
            try:
                st.session_state.agent = PubMedResearchAgent(
                    model_name=model_config["name"],
                    use_llm=model_config["use_llm"],
                    pubmed_email=pubmed_email
                )
                st.success(f"✅ {selected_model} loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                st.session_state.agent = None

    # Initialize agent if not already done
    if st.session_state.agent is None:
        pubmed_email = os.getenv("PUBMED_EMAIL")
        model_config = model_options[selected_model]
        try:
            st.session_state.agent = PubMedResearchAgent(
                model_name=model_config["name"],
                use_llm=model_config["use_llm"],
                pubmed_email=pubmed_email
            )
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.session_state.agent = None

    st.divider()

    # Publication date filter
    st.subheader("📅 Publication Date")
    start_date = st.date_input(
        "Start Date",
        value=three_months_ago,
        min_value=datetime(1900, 1, 1),
        max_value=current_date,
        help="Filter articles from this date onwards"
    )
    end_date = st.date_input(
        "End Date",
        value=current_date,
        min_value=datetime(1900, 1, 1),
        max_value=current_date,
        help="Filter articles up to this date"
    )

    st.divider()

    max_results = st.slider(
        "Max Search Results",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Maximum number of articles to fetch"
    )

    # Search button
    search_button = st.button("🔍 Search PubMed", type="primary", use_container_width=True)

# Process search
if search_button and query:
    st.session_state.current_page = 1  # Reset to first page
    with st.spinner(f"Searching PubMed for: '{query}'..."):
        try:
            # Convert dates to years for PubMed API
            start_year = start_date.year
            start_month = start_date.month
            start_day = start_date.day

            end_year = end_date.year
            end_month = end_date.month
            end_day = end_date.day

            results = st.session_state.agent.search_and_extract(
                query=query,
                max_results=max_results,
                max_records=None,  # Get all results, we'll paginate in the UI
                extract_entities=True,  # Always extract entities
                start_year=start_year,
                end_year=end_year
            )

            # Sort articles by publication date (newest first)
            if results['articles']:
                results['articles'].sort(
                    key=lambda x: int(x.get('year', '0')) if x.get('year', '').isdigit() else 0,
                    reverse=True
                )

                # Update entities to match sorted articles (if entities exist)
                if results.get('entities'):
                    pmid_to_entity = {e['pmid']: e for e in results['entities']}
                    results['entities'] = [
                        pmid_to_entity[article['pmid']]
                        for article in results['articles']
                        if article.get('pmid') in pmid_to_entity
                    ]

            st.session_state.results = results
            st.session_state.search_history.append({
                'query': query,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'num_articles': len(results['articles'])
            })

            st.success(f"✅ Found {len(results['articles'])} articles!")

        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")

# Display results
if st.session_state.results:
    results = st.session_state.results

    # Pagination settings
    articles_per_page = 50
    total_articles = len(results['articles'])
    total_pages = (total_articles + articles_per_page - 1) // articles_per_page

    # Summary metrics
    st.header("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Query", results['query'][:20] + "..." if len(results['query']) > 20 else results['query'])
    with col2:
        st.metric("Total Articles", results['total_articles'])
    with col3:
        if results.get('entities'):
            unique_drugs = len(set(
                drug.get('name', '')
                for entity in results['entities']
                for drug in entity['entities'].get('drugs', [])
                if drug.get('name')
            ))
            st.metric("Unique Drugs", unique_drugs)
        else:
            st.metric("Unique Drugs", 0)
    with col4:
        if results.get('entities'):
            unique_aes = len(set(
                ae.get('event', '')
                for entity in results['entities']
                for ae in entity['entities'].get('adverse_events', [])
                if ae.get('event')
            ))
            st.metric("Adverse Events", unique_aes)
        else:
            st.metric("Adverse Events", 0)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Articles", "🧬 Entities", "📈 Statistics", "💾 Export"])

    with tab1:
        st.subheader("Articles")

        # Pagination controls
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", disabled=(st.session_state.current_page == 1)):
                    st.session_state.current_page -= 1
                    st.rerun()
            with col2:
                st.markdown(f"<h4 style='text-align: center'>Page {st.session_state.current_page} of {total_pages}</h4>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", disabled=(st.session_state.current_page == total_pages)):
                    st.session_state.current_page += 1
                    st.rerun()

        # Calculate pagination indices
        start_idx = (st.session_state.current_page - 1) * articles_per_page
        end_idx = min(start_idx + articles_per_page, total_articles)

        # Display articles for current page
        for i, article in enumerate(results['articles'][start_idx:end_idx], start=start_idx + 1):
            with st.expander(f"[{i}] {article.get('title', 'No title')} ({article.get('year', 'N/A')})"):
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

                    if article.get('abstract'):
                        st.markdown("**Abstract:**")

                        # Find corresponding entities for this article
                        article_entities = None
                        if results.get('entities'):
                            for entity_data in results['entities']:
                                if entity_data.get('pmid') == article.get('pmid'):
                                    article_entities = entity_data.get('entities', {})
                                    break

                        # Display abstract with highlighted entities
                        if article_entities:
                            highlighted_abstract = highlight_entities_in_text(
                                article['abstract'],
                                article_entities
                            )
                            st.markdown(
                                f'<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto;">{highlighted_abstract}</div>',
                                unsafe_allow_html=True
                            )

                            # Add legend
                            st.markdown(
                                '<div style="margin-top: 10px; font-size: 0.85em;">'
                                '<span style="background-color: #E3F2FD; padding: 2px 6px; border-radius: 3px; margin-right: 8px;">💊 Drug</span>'
                                '<span style="background-color: #FFEBEE; padding: 2px 6px; border-radius: 3px; margin-right: 8px;">⚠️ Adverse Event</span>'
                                '<span style="background-color: #FFF3E0; padding: 2px 6px; border-radius: 3px;">🏥 Disease</span>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            # No entities, display plain text
                            st.text_area("Abstract", article['abstract'], height=150, key=f"abstract_{i}", label_visibility="collapsed")

        # Pagination controls at bottom
        if total_pages > 1:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous Page", disabled=(st.session_state.current_page == 1), key="prev_bottom"):
                    st.session_state.current_page -= 1
                    st.rerun()
            with col2:
                st.markdown(f"<h4 style='text-align: center'>Page {st.session_state.current_page} of {total_pages}</h4>", unsafe_allow_html=True)
            with col3:
                if st.button("Next Page ➡️", disabled=(st.session_state.current_page == total_pages), key="next_bottom"):
                    st.session_state.current_page += 1
                    st.rerun()

        with tab2:
            if results.get('entities'):
                st.subheader("Extracted Entities")

                # Aggregate entities
                all_drugs = {}
                all_aes = {}
                all_diseases = set()

                for entity_data in results['entities']:
                    entities = entity_data['entities']

                    for drug in entities.get('drugs', []):
                        name = drug.get('name', '')
                        if name:
                            if name not in all_drugs:
                                all_drugs[name] = []
                            all_drugs[name].append(entity_data['pmid'])

                    for ae in entities.get('adverse_events', []):
                        event = ae.get('event', '')
                        if event:
                            if event not in all_aes:
                                all_aes[event] = {'count': 0, 'pmids': []}
                            all_aes[event]['count'] += 1
                            all_aes[event]['pmids'].append(entity_data['pmid'])

                    for disease in entities.get('diseases', []):
                        if disease:
                            all_diseases.add(disease)

                # Display drugs
                if all_drugs:
                    st.markdown("### 💊 Drugs/Medications")
                    for drug, pmids in sorted(all_drugs.items()):
                        st.markdown(f"- **{drug}** (mentioned in {len(pmids)} articles)")

                # Display adverse events
                if all_aes:
                    st.markdown("### ⚠️ Adverse Events")
                    for ae, data in sorted(all_aes.items(), key=lambda x: x[1]['count'], reverse=True):
                        st.markdown(f"- **{ae}** ({data['count']} occurrences)")

                # Display diseases
                if all_diseases:
                    st.markdown("### 🏥 Diseases/Conditions")
                    for disease in sorted(all_diseases):
                        st.markdown(f"- {disease}")

                # Detailed extraction by article
                st.markdown("---")
                st.markdown("### 📋 Detailed Extraction by Article")

                for i, entity_data in enumerate(results['entities'], 1):
                    with st.expander(f"[{i}] {entity_data['title'][:80]}..."):
                        st.markdown(f"**PMID:** {entity_data['pmid']}")

                        entities = entity_data['entities']

                        # Demographics
                        demo = entities.get('demographics', {})
                        if demo and demo.get('sample_size', 0) > 0:
                            st.markdown("**Demographics:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Sample Size", demo.get('sample_size', 0))
                            with col2:
                                st.metric("Age", demo.get('age', 'Unknown'))
                            with col3:
                                st.metric("Gender", demo.get('gender', 'Unknown'))
                            with col4:
                                ethnicity = demo.get('ethnicity', 'Unknown')
                                if ethnicity != 'Unknown':
                                    st.metric("Ethnicity", ethnicity)

                        # Drugs
                        drugs = entities.get('drugs', [])
                        if drugs:
                            st.markdown(f"**Drugs ({len(drugs)}):**")
                            for drug in drugs:
                                st.markdown(f"- {drug.get('name', 'Unknown')}: {drug.get('context', '')}")

                        # Adverse Events
                        aes = entities.get('adverse_events', [])
                        if aes:
                            st.markdown(f"**Adverse Events ({len(aes)}):**")
                            for ae in aes:
                                severity = ae.get('severity', 'unknown')
                                st.markdown(f"- {ae.get('event', 'Unknown')} [{severity}]: {ae.get('context', '')}")
            else:
                st.info("No entity extraction performed. Enable 'Extract Entities' in the sidebar.")

        with tab3:
            st.subheader("Statistics")

            if results.get('entities'):
                # Year distribution
                years = [int(entity['year']) for entity in results['entities'] if entity.get('year') and entity['year'].isdigit()]
                if years:
                    import pandas as pd
                    year_counts = pd.Series(years).value_counts().sort_index()
                    st.bar_chart(year_counts)

                st.markdown("---")

                # Sample size distribution
                sample_sizes = [
                    entity['entities']['demographics'].get('sample_size', 0)
                    for entity in results['entities']
                    if entity['entities']['demographics'].get('sample_size', 0) > 0
                ]

                if sample_sizes:
                    st.markdown("### Sample Size Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean", f"{sum(sample_sizes)/len(sample_sizes):.0f}")
                    with col2:
                        st.metric("Min", min(sample_sizes))
                    with col3:
                        st.metric("Max", max(sample_sizes))
            else:
                st.info("No statistics available without entity extraction.")

        with tab4:
            st.subheader("Export Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📄 Text Report**")
                if st.button("Generate Text Report"):
                    report = st.session_state.agent.generate_report(results)
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"pubmed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

            with col2:
                st.markdown("**📊 JSON Data**")
                json_data = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"pubmed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Footer
st.divider()
st.caption("PubMed Research Agent - Powered by NCBI E-utilities API")
