"""
Streamlit Web Interface for PubMed Research Agent
Refactored for better maintainability
"""

import streamlit as st
import os
import warnings
import time
from datetime import datetime
from dotenv import load_dotenv

# Import modular components
from components.sidebar import render_sidebar
from components.tabs import (
    render_article_tab_content,
    render_summary_card,
    render_entities_tab,
    render_export_tab
)

# Load environment variables
load_dotenv()

# Suppress llama.cpp Metal warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['GGML_METAL_LOG_LEVEL'] = '0'

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

# Render sidebar and get search parameters
query, start_date, end_date, max_results, search_button = render_sidebar()

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

            # Search PubMed for articles
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

                # Create tabs
                tab1, tab2, tab3 = st.tabs(["📄 Articles", "🧬 Entities", "💾 Export"])

                # Add placeholder messages to Entities and Export tabs
                with tab2:
                    entities_placeholder = st.empty()
                    entities_placeholder.info("⏳ Processing in progress... Entity statistics will be updated once all articles are processed.")

                with tab3:
                    export_placeholder = st.empty()
                    export_placeholder.info("⏳ Processing in progress... Export options will be available once all articles are processed.")

                # Articles tab content
                with tab1:
                    # Progress bar with timer
                    progress_col1, progress_col2 = st.columns([5, 1])
                    with progress_col1:
                        progress_bar = st.progress(0, text="Starting entity extraction...")
                    with progress_col2:
                        timer_placeholder = st.empty()

                    summary_container = st.empty()
                    articles_container = st.container()

                total_start_time = time.time()
                processing_times = []

                # Process each article
                for i, article in enumerate(articles, 1):
                    article_start_time = time.time()

                    # Calculate estimated time
                    if processing_times:
                        avg_time = sum(processing_times) / len(processing_times)
                        remaining = len(articles) - i + 1
                        est_mins = int((avg_time * remaining) // 60)
                        est_secs = int((avg_time * remaining) % 60)
                        est_text = f" (Est. {est_mins:02d}:{est_secs:02d} left)"
                    else:
                        est_text = ""

                    # Update progress
                    progress = (i - 1) / len(articles)
                    with progress_col1:
                        progress_bar.progress(
                            progress,
                            text=f"Processing {i}/{len(articles)}: {article.get('title', 'Unknown')[:50]}...{est_text}"
                        )

                    # Show processing indicator
                    with progress_col2:
                        timer_placeholder.markdown(
                            f"<div style='text-align: right; font-size: 13px; color: #666; margin-top: 8px;'>⏱️ ...</div>",
                            unsafe_allow_html=True
                        )

                    # Extract entities
                    abstract = article.get('abstract', '')
                    article_entities = None
                    
                    if abstract:
                        try:
                            entities = st.session_state.agent.extractor.extract(abstract)

                            # Calculate time taken
                            article_elapsed = time.time() - article_start_time
                            processing_times.append(article_elapsed)

                            article_mins = int(article_elapsed) // 60
                            article_secs = int(article_elapsed) % 60

                            # Show time taken
                            with progress_col2:
                                timer_placeholder.markdown(
                                    f"<div style='text-align: right; font-size: 13px; color: #10b981; margin-top: 8px;'>✓ {article_mins:02d}:{article_secs:02d}</div>",
                                    unsafe_allow_html=True
                                )

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

                    # Show completed article in Articles tab
                    with articles_container:
                        render_article_tab_content(article, article_entities, i)

                # Finalize progress and show total time
                total_elapsed = int(time.time() - total_start_time)
                total_mins = total_elapsed // 60
                total_secs = total_elapsed % 60

                with progress_col1:
                    progress_bar.progress(1.0, text="✅ All articles processed!")
                with progress_col2:
                    timer_placeholder.markdown(
                        f"<div style='text-align: right; font-size: 13px; color: #10b981; margin-top: 8px; font-weight: bold;'>✅ {total_mins:02d}:{total_secs:02d}</div>",
                        unsafe_allow_html=True
                    )

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

                # Count demographics only if they have meaningful values (not 'Unknown' or empty)
                unique_demographics = len(set(
                    f"{demo.get('age', '')}_{demo.get('gender', '')}_{demo.get('race', '')}"
                    for entity in results['entities']
                    for demo in [entity['entities'].get('demographics', {})]
                    if demo and any([
                        demo.get('age') and demo.get('age') != 'Unknown',
                        demo.get('gender') and demo.get('gender') not in ['Unknown', ''],
                        demo.get('race') and demo.get('race') not in ['Unknown', ''],
                        demo.get('pregnancy') and demo.get('pregnancy') != 'Unknown',
                        demo.get('bmi') and demo.get('bmi') != 'Unknown',
                        demo.get('sample_size') and demo.get('sample_size') > 0
                    ])
                )) if results.get('entities') else 0

                # Render summary card in Articles tab
                with summary_container:
                    render_summary_card(results, unique_drugs, unique_aes, unique_demographics)

                # Update Entities tab
                entities_placeholder.empty()
                with tab2:
                    render_entities_tab(results)

                # Update Export tab
                export_placeholder.empty()
                with tab3:
                    render_export_tab(results)

        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Footer
st.divider()
st.caption("PubMed Research Agent - Powered by NCBI E-utilities API")
