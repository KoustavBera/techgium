"""
Chiranjeevi Medical Agent — Research Tools
===========================================
Plain-Python wrappers for Tavily web search and PubMed paper retrieval.
These are NOT LangChain tools — they are simple functions called by nodes.
"""

from __future__ import annotations

import requests
import xmltodict
from typing import List, Dict, Any

from agent.config import TAVILY_API_KEY


# ═══════════════════════════════════════════════════════════════════════
# Tavily Web Search
# ═══════════════════════════════════════════════════════════════════════

def search_tavily(query: str, max_results: int = 3) -> tuple[str, list[dict[str, str]]]:
    """Search the web via Tavily API and return formatted results + metadata.

    Parameters
    ----------
    query : str
        The medical query to search for.
    max_results : int
        Maximum number of results to return.

    Returns
    -------
    tuple[str, list[dict]]
        1. Formatted bullet-point results.
        2. List of source dicts: {"title": str, "url": str, "source": "Tavily"}
    """
    if not TAVILY_API_KEY:
        return "[Tavily] No API key configured — skipping web search.", []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )

        lines: list[str] = []
        sources: list[dict[str, str]] = []

        # Include the AI-generated answer if available
        if response.get("answer"):
            lines.append(f"**Tavily Summary**: {response['answer']}\n")

        # Include individual results and collect metadata
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            snippet = result.get("content", "No snippet")[:300]
            url = result.get("url", "")
            
            lines.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")
            sources.append({
                "title": title,
                "url": url,
                "source": "Tavily"
            })

        formatted_text = "\n".join(lines) if lines else "[Tavily] No results found."
        return formatted_text, sources

    except Exception as e:
        return f"[Tavily] Search failed: {e}", []


# ═══════════════════════════════════════════════════════════════════════
# PubMed Search (NCBI E-Utilities — free, no key required for <3 req/s)
# ═══════════════════════════════════════════════════════════════════════

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, max_results: int = 3) -> tuple[str, list[dict[str, str]]]:
    """Search PubMed for clinical papers and return formatted results + metadata.

    Uses the NCBI E-Utilities REST API.

    Returns
    -------
    tuple[str, list[dict]]
        1. Formatted list of papers.
        2. List of source dicts: {"title": str, "url": str, "source": "PubMed"}
    """
    try:
        # Step 1: Search for PMIDs
        search_params: Dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        search_resp = requests.get(PUBMED_ESEARCH_URL, params=search_params, timeout=10)
        search_resp.raise_for_status()
        pmids: List[str] = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return "[PubMed] No papers found for this query.", []

        # Step 2: Fetch article details via XML
        fetch_params: Dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        fetch_resp = requests.get(PUBMED_EFETCH_URL, params=fetch_params, timeout=15)
        fetch_resp.raise_for_status()

        parsed = xmltodict.parse(fetch_resp.text)
        articles_raw = parsed.get("PubmedArticleSet", {}).get("PubmedArticle", [])

        # Ensure it's always a list (single result comes as dict)
        if isinstance(articles_raw, dict):
            articles_raw = [articles_raw]

        lines: list[str] = []
        sources: list[dict[str, str]] = []

        for i, article in enumerate(articles_raw, 1):
            medline = article.get("MedlineCitation", {})
            art_info = medline.get("Article", {})

            title = art_info.get("ArticleTitle", "No title")
            # Handle structured vs plain abstract
            abstract_data = art_info.get("Abstract", {}).get("AbstractText", "")
            if isinstance(abstract_data, list):
                abstract = " ".join(
                    item if isinstance(item, str) else item.get("#text", "")
                    for item in abstract_data
                )
            elif isinstance(abstract_data, dict):
                abstract = abstract_data.get("#text", str(abstract_data))
            else:
                abstract = str(abstract_data)

            abstract = abstract[:400] + "..." if len(abstract) > 400 else abstract
            pmid = medline.get("PMID", {})
            pmid_val = pmid if isinstance(pmid, str) else pmid.get("#text", "N/A")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_val}/"

            lines.append(
                f"{i}. **{title}**\n"
                f"   {abstract}\n"
                f"   PMID: {pmid_val} | {url}"
            )
            sources.append({
                "title": title,
                "url": url,
                "source": "PubMed"
            })

        formatted_text = "\n".join(lines) if lines else "[PubMed] Could not parse results."
        return formatted_text, sources

    except Exception as e:
        return f"[PubMed] Search failed: {e}", []
