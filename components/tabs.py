"""Tab Components for PubMed Research Agent"""

import streamlit as st
import pandas as pd
import plotly.express as px
import re
from datetime import datetime
from utils.entity_highlighter import highlight_entities_in_text


def render_article_tab_content(article, article_entities, article_num):
    """Render content for a single article in the Articles tab"""
    
    with st.expander(f"✅ [{article_num}] {article.get('title', 'No title')} ({article.get('year', 'N/A')})", expanded=False):
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

        abstract = article.get('abstract', '')
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
            
            entity_rows = _build_entity_count_table(article_entities, abstract)
            
            if entity_rows:
                entity_df = pd.DataFrame(entity_rows)
                st.dataframe(entity_df, use_container_width=True, hide_index=True)
            else:
                st.info("No entities extracted from this article.")


def _build_entity_count_table(article_entities, abstract):
    """Build entity count table for a single article"""
    entity_rows = []
    row_num = 1
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

    # Add demographics
    demo = article_entities.get('demographics', {})
    if demo:
        entity_rows.extend(_extract_demographics_for_table(demo, abstract, row_num))

    return entity_rows


def _extract_demographics_for_table(demo, abstract, row_num):
    """Extract demographics patterns for entity count table"""
    rows = []
    abstract_lower = abstract.lower()

    # Age patterns
    age = demo.get('age', '')
    if age and age != 'Unknown':
        rows.extend(_extract_age_patterns_for_table(age, abstract, abstract_lower, row_num))
        row_num += len(rows)

    # Gender patterns
    gender = demo.get('gender', '')
    if gender and gender != 'Unknown':
        gender_rows = _extract_gender_patterns_for_table(gender, abstract, abstract_lower, row_num)
        rows.extend(gender_rows)
        row_num += len(gender_rows)

    # Race patterns
    race = demo.get('race', demo.get('ethnicity', ''))
    if race and race != 'Unknown' and len(race) > 3:
        race_rows = _extract_race_patterns_for_table(race, abstract, abstract_lower, row_num)
        rows.extend(race_rows)
        row_num += len(race_rows)

    # Sample size patterns
    sample_size = demo.get('sample_size', 0)
    if sample_size and sample_size > 0:
        size_rows = _extract_sample_size_patterns_for_table(sample_size, abstract, abstract_lower, row_num)
        rows.extend(size_rows)

    return rows


def _extract_age_patterns_for_table(age, abstract, abstract_lower, row_num):
    """Extract age patterns for table"""
    rows = []
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
            rows.append({
                "No.": row_num,
                "Keyword": phrase,
                "Entity Type": "Demographics",
                "Count": count
            })
            row_num += 1

    return rows


def _extract_gender_patterns_for_table(gender, abstract, abstract_lower, row_num):
    """Extract gender patterns for table"""
    rows = []
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
                    rows.append({
                        "No.": row_num,
                        "Keyword": phrase,
                        "Entity Type": "Demographics",
                        "Count": count
                    })
                    row_num += 1

    return rows


def _extract_race_patterns_for_table(race, abstract, abstract_lower, row_num):
    """Extract race patterns for table"""
    rows = []
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
                    rows.append({
                        "No.": row_num,
                        "Keyword": phrase,
                        "Entity Type": "Demographics",
                        "Count": count
                    })
                    row_num += 1

    return rows


def _extract_sample_size_patterns_for_table(sample_size, abstract, abstract_lower, row_num):
    """Extract sample size patterns for table"""
    rows = []
    size_patterns = [
        rf'\b{sample_size}\s+(?:patients|participants|subjects)\b',
        rf'\bn\s*=\s*{sample_size}\b'
    ]
    for pattern in size_patterns:
        matches = list(re.finditer(pattern, abstract, re.IGNORECASE))
        for match in matches:
            phrase = match.group(0)
            count = abstract_lower.count(phrase.lower())
            rows.append({
                "No.": row_num,
                "Keyword": phrase,
                "Entity Type": "Demographics",
                "Count": count
            })
            row_num += 1
        if matches:
            break

    return rows


def render_summary_card(results, unique_drugs, unique_aes, unique_diseases, unique_demographics):
    """Render summary card with statistics"""
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


def render_entities_tab(results):
    """Render Entities tab content"""
    if not results.get('entities'):
        st.info("No entities extracted.")
        return

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


def render_export_tab(results):
    """Render Export tab content"""
    st.subheader("Export Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 CSV Data**")
        csv_data = _generate_csv_data(results)
        
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


def _generate_csv_data(results):
    """Generate CSV data from results"""
    import csv
    import io
    
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

    csv_buffer = io.StringIO()
    if csv_rows:
        writer = csv.DictWriter(csv_buffer, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
        return csv_buffer.getvalue()
    else:
        return "No data available"
