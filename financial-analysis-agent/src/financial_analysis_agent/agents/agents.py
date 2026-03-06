import asyncio
import time

from langchain.agents import create_agent
from src.financial_analysis_agent.config import llm_agent
from src.financial_analysis_agent.logger import logger
from src.financial_analysis_agent.prompts.prompts import (
    financial_analyst_prompt,
    macro_prompt,
    news_analyst_prompt,
    recomendation_prompt,
)
from src.financial_analysis_agent.tools.mcp import mcp
from src.financial_analysis_agent.tools.tools import api_news, duckduckgo_search_tool

_cached_tools = {}
_agents = {}
_tools_lock = asyncio.Lock()


async def get_mcp_tools():
    """Fetches and caches MCP tools only when a relevant agent needs them."""
    global _cached_tools
    async with _tools_lock:
        if not _cached_tools:
            logger.info("Starting MCP tools initialization (Lazy Load)...")
            start_mcp = time.time()
            try:
                financial_tools, recommendations_tool = await mcp()
                duration = time.time() - start_mcp

                _cached_tools["financial"] = financial_tools
                _cached_tools["recommendation"] = recommendations_tool

                logger.info(
                    f"MCP tools loaded successfully in {duration:.2f}s | "
                    f"Financial tools: {len(financial_tools)}, Recommendation tools: {len(recommendations_tool)}"
                )
            except Exception:
                logger.exception("Failed to load MCP tools")
                raise
    return _cached_tools


def create_logged_agent(name: str, model, tools: list, system_prompt_fn):
    """Internal helper to create an agent and log the time it takes."""
    start_time = time.time()
    logger.info(f"Initializing agent '{name}' with {len(tools)} tool(s)")
    try:
        agent = create_agent(model=model, tools=tools, system_prompt=system_prompt_fn())
        duration = time.time() - start_time
        logger.info(f"Agent '{name}' initialized successfully in {duration:.2f}s")
        return agent
    except Exception:
        logger.exception(f"Failed to initialize agent '{name}'")
        raise


async def get_macro_agent():
    """
    Returns the news agent.
    NOTE: Bypasses MCP initialization because news tools are local functions.
    """
    if "macro_agent" not in _agents:
        _agents["macro_agent"] = create_logged_agent(
            "macro_agent",
            model=llm_agent,
            tools=[duckduckgo_search_tool],
            system_prompt_fn=macro_prompt,
        )
    return _agents["macro_agent"]


async def get_news_agent():
    """
    Returns the news agent.
    NOTE: Bypasses MCP initialization because news tools are local functions.
    """
    if "news_agent" not in _agents:
        _agents["news_agent"] = create_logged_agent(
            "news_agent",
            model=llm_agent,
            tools=[api_news],
            system_prompt_fn=news_analyst_prompt,
        )
    return _agents["news_agent"]


async def get_financial_agent():
    """Returns the financial agent, loading MCP tools only if necessary."""
    if "financial_agent" not in _agents:
        tools_dict = await get_mcp_tools()
        _agents["financial_agent"] = create_logged_agent(
            "financial_agent",
            model=llm_agent,
            tools=tools_dict["financial"],
            system_prompt_fn=financial_analyst_prompt,
        )
    return _agents["financial_agent"]


async def get_recommendation_agent():
    """Returns the recommendation agent, loading MCP tools only if necessary."""
    if "recomendation_agent" not in _agents:
        tools_dict = await get_mcp_tools()
        _agents["recomendation_agent"] = create_logged_agent(
            "recomendation_agent",
            model=llm_agent,
            tools=tools_dict["recommendation"],
            system_prompt_fn=recomendation_prompt,
        )
    return _agents["recomendation_agent"]
