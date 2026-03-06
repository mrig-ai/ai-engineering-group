import asyncio
import os
from datetime import date, timedelta

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from langchain.tools import tool
from src.financial_analysis_agent.logger import logger

load_dotenv(".env")  # Force reload updated .env


@tool
async def api_news(query: str) -> dict:
    """
    Fetch news from newsapi API.

    Parameters:
    query (str): Query string

    Returns:
    dict: Response from newsapi API
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        logger.error("NEWS_API_KEY not set")
        raise ValueError("NEWS_API_KEY environment variable not set")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "searchIn": "title",
        "from": (date.today() - timedelta(days=2)).strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": api_key,
    }

    logger.info(f"Fetching news from newsapi API for query: '{query}'")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            total_in_db = data.get("totalResults", 0)
            actual_count = len(articles)
            logger.info(
                f"Query '{query}': {total_in_db} found in DB, but only {actual_count} retrieved (Limit: {params['pageSize']})"
            )
            return data
    except httpx.HTTPStatusError as e:
        logger.error(
            f"newsapi API returned HTTP error {e.response.status_code} for query: '{query}'"
        )
        raise
    except Exception:
        logger.exception(f"Error fetching news for query: '{query}'")
        raise


@tool
async def duckduckgo_search_tool(queries: list[str], max_results: int = 3) -> str:
    """
    Fetch search results from DuckDuckGo API.

    Parameters:
    queries (list[str]): List of query strings
    max_results (int): Maximum number of results to fetch from DuckDuckGo for each query. Defaults to 3.

    Returns:
    str: Report containing search results for each query.
    """
    logger.info(f"duckduckgo_search_tool received {len(queries)} queries: {queries}")

    async def fetch_single(q):
        try:
            with DDGS() as ddgs:
                return list(
                    ddgs.text(q, max_results=max_results, backend="lite", timelimit="w")
                )
        except Exception:
            return []

    tasks = [fetch_single(q) for q in queries]
    all_results = await asyncio.gather(*tasks)

    report = []
    for q, results in zip(queries, all_results):
        report.append(f"### Results for: {q}")
        if not results:
            report.append("No results found.")
        for r in results:
            report.append(f"- {r.get('title')}: {r.get('body')} ({r.get('href')})")

    return "\n".join(report)
