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
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Maximum number of articles to fetch"
    )

    # Search button
    search_button = st.button("🔍 Search PubMed", type="primary", use_container_width=True)

    # Entity legend below search button
    st.markdown(
        '<div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">'
        '<div style="font-weight: bold; margin-bottom: 8px; font-size: 0.9em;">Entity Color Legend:</div>'
        '<div style="font-size: 0.85em;">'
        '<span style="background-color: #E3F2FD; padding: 2px 6px; border-radius: 3px; margin-right: 8px;">💊 Drug</span>'
        '<span style="background-color: #FFEBEE; padding: 2px 6px; border-radius: 3px; margin-right: 8px;">⚠️ Adverse Event</span>'
        '<span style="background-color: #FFF3E0; padding: 2px 6px; border-radius: 3px;">🏥 Disease</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

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
    tab1, tab2, tab3 = st.tabs(["📄 Articles", "🧬 Entities", "💾 Export"])

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

                # Find corresponding entities for this article
                article_entities = None
                if results.get('entities'):
                    for entity_data in results['entities']:
                        if entity_data.get('pmid') == article.get('pmid'):
                            article_entities = entity_data.get('entities', {})
                            break
                
                if article.get('abstract'):
                    st.markdown("**Abstract:**")
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
                    else:
                        # No entities, display plain text
                        st.text_area("Abstract", article['abstract'], height=150, key=f"abstract_{i}", label_visibility="collapsed")

                # Detailed Extraction for this article
                if article_entities:
                    st.markdown("---")
                    st.markdown("### 📋 Detailed Extraction")
                    
                    entities = article_entities

                    # Demographics
                    demo = entities.get('demographics', {})
                    if demo and demo.get('sample_size', 0) > 0:
                        st.markdown("**Demographics:**")
                        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                        with d_col1:
                            st.metric("Sample Size", demo.get('sample_size', 0))
                        with d_col2:
                            st.metric("Age", demo.get('age', 'Unknown'))
                        with d_col3:
                            st.metric("Gender", demo.get('gender', 'Unknown'))
                        with d_col4:
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
                    
                    # Diseases
                    diseases = entities.get('diseases', [])
                    if diseases:
                        st.markdown(f"**Diseases ({len(diseases)}):**")
                        for d in diseases:
                            st.markdown(f"- {d}")


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
            st.subheader("Entity Occurrence Visualization")

            # Aggregate entities
            all_aes = {}
            all_diseases = {}
            all_genders = {}
            all_ethnicities = {}

            for entity_data in results['entities']:
                entities = entity_data['entities']

                for ae in entities.get('adverse_events', []):
                    event = ae.get('event', '')
                    if event:
                        all_aes[event] = all_aes.get(event, 0) + 1

                for disease in entities.get('diseases', []):
                    if disease:
                        all_diseases[disease] = all_diseases.get(disease, 0) + 1
                
                demo = entities.get('demographics', {})
                if demo:
                    gender = demo.get('gender', 'Unknown')
                    if gender and gender != 'Unknown':
                        all_genders[gender] = all_genders.get(gender, 0) + 1
                    
                    ethnicity = demo.get('ethnicity', 'Unknown')
                    if ethnicity and ethnicity != 'Unknown':
                        all_ethnicities[ethnicity] = all_ethnicities.get(ethnicity, 0) + 1

            # Create three columns for charts
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### ⚠️ Adverse Events")
                if all_aes:
                    ae_df = pd.DataFrame(list(all_aes.items()), columns=['Adverse Event', 'Count']).sort_values(by='Count', ascending=False)
                    
                    # Pie Chart
                    fig_pie = px.pie(ae_df.head(10), values='Count', names='Adverse Event', title='Top 10 Adverse Events')
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Bar Chart
                    fig_bar = px.bar(ae_df.head(10), x='Count', y='Adverse Event', orientation='h', title='Top 10 Adverse Events')
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No adverse events to visualize.")

            with col2:
                st.markdown("### 🏥 Diseases")
                if all_diseases:
                    disease_df = pd.DataFrame(list(all_diseases.items()), columns=['Disease', 'Count']).sort_values(by='Count', ascending=False)

                    # Pie Chart
                    fig_pie = px.pie(disease_df.head(10), values='Count', names='Disease', title='Top 10 Diseases')
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Bar Chart
                    fig_bar = px.bar(disease_df.head(10), x='Count', y='Disease', orientation='h', title='Top 10 Diseases')
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No diseases to visualize.")

            with col3:
                st.markdown("### 🧑‍🤝‍🧑 Demographics")
                if all_genders:
                    gender_df = pd.DataFrame(list(all_genders.items()), columns=['Gender', 'Count'])
                    fig_pie = px.pie(gender_df, values='Count', names='Gender', title='Gender Distribution')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No gender data to visualize.")

                if all_ethnicities:
                    ethnicity_df = pd.DataFrame(list(all_ethnicities.items()), columns=['Ethnicity', 'Count'])
                    fig_bar = px.bar(ethnicity_df, x='Count', y='Ethnicity', orientation='h', title='Ethnicity Distribution')
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No ethnicity data to visualize.")

        else:
            st.info("No entity extraction performed. Enable 'Extract Entities' in the sidebar.")

    with tab3:
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
