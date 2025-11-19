"""Sidebar Component for PubMed Research Agent"""

import streamlit as st
import os
from datetime import datetime, timedelta
from utils.model_scanner import scan_installed_gguf_models

try:
    from agent_gguf import PubMedResearchAgentGGUF
    GGUF_AVAILABLE = True
except:
    GGUF_AVAILABLE = False


def render_sidebar():
    """Render the sidebar with search query, model selection, and filters"""
    
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

        # Base model options
        model_options = {
            "Rule-based (Fast)": {"type": "rule", "name": None, "use_llm": False}
        }

        # Scan and add GGUF models
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

        # Ensure default value exists
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
        
        # Calculate default dates (current date and 3 months ago)
        current_date = datetime.now()
        three_months_ago = current_date - timedelta(days=90)
        
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

    return query, start_date, end_date, max_results, search_button
