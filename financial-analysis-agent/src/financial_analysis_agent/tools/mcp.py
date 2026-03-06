from langchain_mcp_adapters.client import MultiServerMCPClient
from src.financial_analysis_agent.logger import logger  # your logger


async def mcp():
    logger.info("Initializing MultiServerMCPClient for yfinance")

    try:
        client = MultiServerMCPClient(
            {
                "yfinance": {
                    "transport": "stdio",
                    "command": "uv",
                    "args": [
                        "--directory",
                        "/home/leonelbaptista/Projects/POC/mcp/yahoo-finance-mcp",
                        "run",
                        "server.py",
                    ],
                },
            }
        )

        logger.info("Fetching tools from MCP client...")
        tools = await client.get_tools()
        logger.info(f"Retrieved {len(tools)} tools from MCP client")

        financial_tools = [
            tool
            for tool in tools
            if tool.name not in ["get_recommendations", "get_yahoo_finance_news"]
        ]
        logger.info(f"Filtered {len(financial_tools)} financial tools")

        recommendations_tool = [
            tool for tool in tools if tool.name in ["get_recommendations"]
        ]
        logger.info(f"Filtered {len(recommendations_tool)} recommendation tools")

        return financial_tools, recommendations_tool

    except Exception:
        logger.exception("Error initializing MCP client or fetching tools")
        raise
