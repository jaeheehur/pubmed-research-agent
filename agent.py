"""
PubMed Research Agent
Main agent that combines PubMed search and entity extraction
"""

import logging
from typing import List, Dict, Any, Optional
from transformers import pipeline
from tools import PubMedSearcher, format_articles
from utils import EntityExtractor, ExtractedEntities

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PubMedResearchAgent:
    """
    Agent for PubMed research with entity extraction capabilities.

    This agent can:
    1. Search PubMed for relevant articles
    2. Extract medical entities (drugs, adverse events, demographics)
    3. Generate structured research reports
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_llm: bool = True,
        pubmed_email: Optional[str] = None
    ):
        """
        Initialize the research agent.

        Args:
            model_name: HuggingFace model name for LLM (if None and use_llm=True, uses rule-based extraction)
            use_llm: Whether to use LLM for entity extraction
            pubmed_email: Optional email for NCBI API (loaded from .env if not provided)
        """
        logger.info("Initializing PubMed Research Agent...")

        # Initialize PubMed searcher
        self.searcher = PubMedSearcher(email=pubmed_email)

        # Initialize LLM pipeline if requested
        self.use_llm = use_llm
        self.model = None

        if use_llm and model_name:
            try:
                logger.info(f"Loading model: {model_name}")

                # Check if CUDA is available for quantization
                import torch
                has_cuda = torch.cuda.is_available()

                if has_cuda:
                    # Use 8-bit quantization only on CUDA devices
                    logger.info("CUDA detected, using 8-bit quantization")
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,  # Use 8-bit quantization for speed
                        llm_int8_threshold=6.0
                    )

                    self.model = pipeline(
                        "text-generation",
                        model=model_name,
                        trust_remote_code=True,
                        device_map="auto",
                        model_kwargs={
                            "quantization_config": quantization_config,
                            "low_cpu_mem_usage": True
                        }
                    )
                    logger.info("Model loaded successfully with 8-bit quantization")
                else:
                    # No CUDA, load without quantization (for CPU or MPS)
                    logger.info("No CUDA detected, loading model without quantization")
                    self.model = pipeline(
                        "text-generation",
                        model=model_name,
                        trust_remote_code=True,
                        device_map="auto",  # Will use MPS on Mac or CPU
                        model_kwargs={"low_cpu_mem_usage": True}
                    )
                    logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.warning("Falling back to rule-based extraction")
                self.use_llm = False
        elif use_llm and not model_name:
            logger.warning("use_llm=True but no model_name provided, using rule-based extraction")
            self.use_llm = False

        # Initialize entity extractor
        self.extractor = EntityExtractor(model_pipeline=self.model)

        logger.info("Agent initialization complete")

    def search_and_extract(
        self,
        query: str,
        max_results: int = 50,
        max_records: int = 10,
        extract_entities: bool = True,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search PubMed and extract entities from results.

        Args:
            query: PubMed search query
            max_results: Maximum articles to search
            max_records: Maximum articles to return
            extract_entities: Whether to extract entities from abstracts
            start_year: Start year for publication date filter (optional)
            end_year: End year for publication date filter (optional)

        Returns:
            Dictionary containing articles and extracted entities
        """
        logger.info(f"Starting research query: {query}")

        # Search PubMed
        logger.info("Searching PubMed...")
        articles = self.searcher.search(
            query=query,
            max_results=max_results,
            max_records=max_records,
            rerank="referenced_by",
            start_year=start_year,
            end_year=end_year
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
            result["entities"] = self._extract_from_articles(articles)

        return result

    def _extract_from_articles(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract entities from multiple articles"""
        extracted_data = []

        for i, article in enumerate(articles, 1):
            logger.info(f"Processing article {i}/{len(articles)}: {article.get('pmid', 'Unknown')}")

            abstract = article.get('abstract', '')

            if not abstract:
                logger.warning(f"No abstract for article {article.get('pmid', 'Unknown')}")
                continue

            # Extract entities
            entities = self.extractor.extract(abstract, use_model=self.use_llm)

            extracted_data.append({
                "pmid": article.get('pmid'),
                "title": article.get('title'),
                "journal": article.get('journal'),
                "year": article.get('year'),
                "doi": article.get('doi'),
                "entities": entities.to_dict()
            })

        logger.info(f"Completed entity extraction for {len(extracted_data)} articles")
        return extracted_data

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
        import json

        # Convert ExtractedEntities to dict if needed
        data_copy = research_data.copy()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_copy, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON data saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize agent (without LLM for quick testing)
    agent = PubMedResearchAgent(use_llm=False)

    # Run a search
    query = "metformin adverse events type 2 diabetes"

    results = agent.search_and_extract(
        query=query,
        max_results=20,
        max_records=5,
        extract_entities=True
    )

    # Generate and print report
    report = agent.generate_report(results)
    print(report)

    # Optionally save
    # agent.save_report(results, "research_report.txt")
    # agent.save_json(results, "research_data.json")
