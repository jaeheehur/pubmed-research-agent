"""
PubMed Search Tool
Adapted from AWS Bedrock Agents Healthcare & Life Sciences repository
Original: https://github.com/aws-samples/amazon-bedrock-agents-healthcare-lifesciences
"""

import logging
from typing import List, Dict, Any, Optional
from xml.etree.ElementTree import Element
from defusedxml import ElementTree as ET
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedSearcher:
    """PubMed search tool for finding and ranking scientific articles."""

    def __init__(self, email: Optional[str] = None):
        """
        Initialize PubMed searcher.

        Args:
            email: Email address for NCBI (optional, can also set via PUBMED_EMAIL env var)
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email or os.getenv("PUBMED_EMAIL")

        if self.email:
            logger.info(f"PubMed API configured with email: {self.email}")
        else:
            logger.warning("No email configured for PubMed API. Please set PUBMED_EMAIL in .env file")

    def search(
        self,
        query: str,
        max_results: int = 100,
        max_records: Optional[int] = None,
        rerank: str = "referenced_by",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed for articles matching the query.

        Args:
            query: PubMed search query
            max_results: Maximum number of results to fetch from initial search
            max_records: Maximum number of articles to return in final results
            rerank: Reranking method ("referenced_by" or None)
            start_year: Start year for publication date filter (optional, deprecated - use start_date)
            end_year: End year for publication date filter (optional, deprecated - use end_date)
            start_date: Start date in YYYY/MM/DD format (optional)
            end_date: End date in YYYY/MM/DD format (optional)

        Returns:
            List of article dictionaries containing metadata and abstracts
        """
        # Add date filter to query if provided
        # Prefer start_date/end_date over start_year/end_year
        if start_date or end_date:
            date_filter = self._build_date_filter_with_dates(start_date, end_date)
            filtered_query = f"{query} AND {date_filter}"
            logger.info(f"Searching PubMed for: {query} (filtered: {filtered_query})")
        elif start_year or end_year:
            date_filter = self._build_date_filter(start_year, end_year)
            filtered_query = f"{query} AND {date_filter}"
            logger.info(f"Searching PubMed for: {query} (filtered: {filtered_query})")
        else:
            filtered_query = query
            logger.info(f"Searching PubMed for: {query}")

        # Search for article IDs
        pmids = self._search_pmids(filtered_query, max_results)

        if not pmids:
            logger.info("No articles found")
            return []

        # Fetch article details
        articles = self.fetch_articles(pmids)

        # Apply reranking if requested
        if rerank == "referenced_by":
            logger.info("Calculating citation relationships and ranking articles")
            articles = self._calculate_referenced_by_counts(articles)
            articles = self._rank_by_citations(articles)

        # Apply max_records limit
        if max_records is not None:
            articles = articles[:max_records]

        logger.info(f"Returning {len(articles)} articles")
        return articles

    def fetch_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed information about articles by PMID.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of article dictionaries
        """
        if not pmids:
            return []

        logger.info(f"Fetching {len(pmids)} PubMed articles")

        fetch_url = f"{self.base_url}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }

        if self.email:
            params["email"] = self.email

        response = requests.post(fetch_url, data=params, timeout=30)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)

        articles = []
        for article_element in root.findall(".//PubmedArticle"):
            try:
                article = self._extract_article_data(article_element)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue

        logger.info(f"Successfully fetched {len(articles)} articles")
        return articles

    def _search_pmids(self, query: str, max_results: int) -> List[str]:
        """Search for article PMIDs matching the query."""
        search_url = f"{self.base_url}/esearch.fcgi"

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }

        if self.email:
            params["email"] = self.email

        response = requests.post(search_url, data=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])

        logger.info(f"Found {len(pmids)} article PMIDs")
        return pmids

    def _build_date_filter(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> str:
        """Build date filter for PubMed query (year-based, deprecated)."""
        if start_year and end_year:
            return f"({start_year}[PDAT]:{end_year}[PDAT])"
        elif start_year:
            return f"{start_year}:3000[PDAT]"
        elif end_year:
            return f"1900:{end_year}[PDAT]"
        return ""

    def _build_date_filter_with_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """
        Build date filter for PubMed query with full dates (YYYY/MM/DD).
        Uses PDAT (Publication Date) which supports day-level precision.

        Args:
            start_date: Start date in YYYY/MM/DD format
            end_date: End date in YYYY/MM/DD format

        Returns:
            Date filter string for PubMed query
        """
        if start_date and end_date:
            return f"({start_date}[PDAT]:{end_date}[PDAT])"
        elif start_date:
            return f"{start_date}:3000[PDAT]"
        elif end_date:
            return f"1900/01/01:{end_date}[PDAT]"
        return ""

    def _extract_article_data(self, article_element: Element) -> Dict[str, Any]:
        """Extract article data from XML element."""
        article = {}

        # Extract PMID
        pmid_element = article_element.find(".//PMID")
        if pmid_element is not None and pmid_element.text:
            article["pmid"] = pmid_element.text

        # Extract title
        title_element = article_element.find(".//ArticleTitle")
        if title_element is not None:
            title_text = "".join(title_element.itertext()).strip()
            if title_text:
                article["title"] = title_text

        # Extract abstract
        abstract_parts = article_element.findall(".//AbstractText")
        if abstract_parts:
            abstract_texts = []
            for part in abstract_parts:
                text_content = "".join(part.itertext()).strip()
                if text_content:
                    abstract_texts.append(text_content)
            if abstract_texts:
                article["abstract"] = " ".join(abstract_texts)

        # Extract authors
        author_elements = article_element.findall(".//Author")
        if author_elements:
            authors = []
            for author in author_elements:
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")

                if last_name is not None and fore_name is not None:
                    if last_name.text and fore_name.text:
                        authors.append(f"{fore_name.text} {last_name.text}")
                elif last_name is not None and last_name.text:
                    authors.append(last_name.text)

            if authors:
                article["authors"] = authors

        # Extract journal info
        journal_element = article_element.find(".//Journal/Title")
        if journal_element is not None and journal_element.text:
            article["journal"] = journal_element.text

        # Extract publication year
        pub_date_element = article_element.find(".//PubDate/Year")
        if pub_date_element is not None and pub_date_element.text:
            article["year"] = pub_date_element.text

        # Extract DOI and PMC
        pubmed_data = article_element.find("PubmedData")
        if pubmed_data is not None:
            article_id_list = pubmed_data.find("ArticleIdList")
            if article_id_list is not None:
                for article_id in article_id_list.findall("ArticleId"):
                    id_type = article_id.get("IdType")
                    if id_type == "doi" and article_id.text:
                        article["doi"] = article_id.text
                    elif id_type == "pmc" and article_id.text:
                        article["pmc"] = article_id.text

        # Extract references
        reference_elements = article_element.findall(".//Reference")
        if reference_elements:
            references = []
            for ref in reference_elements:
                ref_pmid = ref.find(".//ArticleId[@IdType='pubmed']")
                if ref_pmid is not None and ref_pmid.text:
                    references.append(ref_pmid.text)
            if references:
                article["references"] = references

        return article

    def _calculate_referenced_by_counts(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate how many times each article is referenced by others."""
        # Build citation graph
        citation_graph = {}

        for article in articles:
            pmid = article.get("pmid")
            if pmid:
                citation_graph[pmid] = set()

        for article in articles:
            pmid = article.get("pmid")
            references = article.get("references", [])

            if not pmid:
                continue

            for ref_pmid in references:
                if ref_pmid and ref_pmid in citation_graph and ref_pmid != pmid:
                    citation_graph[ref_pmid].add(pmid)

        # Add citation counts to articles
        for article in articles:
            pmid = article.get("pmid")
            if pmid and pmid in citation_graph:
                article["referenced_by_count"] = len(citation_graph[pmid])
            else:
                article["referenced_by_count"] = 0

        return articles

    def _rank_by_citations(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank articles by citation count in descending order."""
        return sorted(
            articles,
            key=lambda a: (
                a.get("referenced_by_count", 0),
                int(a.get("pmid", "0")) if a.get("pmid", "").isdigit() else 0
            ),
            reverse=True
        )


def format_article(article: Dict[str, Any], index: Optional[int] = None) -> str:
    """
    Format an article as readable text.

    Args:
        article: Article dictionary
        index: Optional article number

    Returns:
        Formatted article string
    """
    lines = []

    if index is not None:
        lines.append(f"\n{'='*80}")
        lines.append(f"Article {index}")
        lines.append('='*80)

    lines.append(f"Title: {article.get('title', 'No title')}")

    authors = article.get('authors', [])
    if authors:
        if isinstance(authors, list):
            lines.append(f"Authors: {', '.join(authors[:3])}" +
                        (f" et al. ({len(authors)} total)" if len(authors) > 3 else ""))
        else:
            lines.append(f"Authors: {authors}")

    lines.append(f"Journal: {article.get('journal', 'Unknown')} ({article.get('year', 'N/A')})")

    if article.get('pmid'):
        lines.append(f"PMID: {article['pmid']}")
    if article.get('doi'):
        lines.append(f"DOI: {article['doi']}")
    if article.get('pmc'):
        lines.append(f"PMC: {article['pmc']}")

    if 'referenced_by_count' in article:
        lines.append(f"Cited by: {article['referenced_by_count']} articles in result set")

    abstract = article.get('abstract', 'No abstract available')
    if len(abstract) > 500:
        abstract = abstract[:497] + "..."
    lines.append(f"\nAbstract: {abstract}")

    return "\n".join(lines)


def format_articles(articles: List[Dict[str, Any]]) -> str:
    """Format multiple articles as readable text."""
    if not articles:
        return "No articles found."

    result = [f"Found {len(articles)} articles\n"]

    for i, article in enumerate(articles, 1):
        result.append(format_article(article, index=i))

    return "\n".join(result)


# Example usage
if __name__ == "__main__":
    searcher = PubMedSearcher()

    # Example search
    articles = searcher.search(
        query="GLP-1 receptor agonist",
        max_results=50,
        max_records=5,
        rerank="referenced_by"
    )

    print(format_articles(articles))
