from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.financial_analysis_agent.logger import logger

load_dotenv()
logger.info("Environment variables loaded via dotenv")

try:
    llm_agent = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
        max_retries=2,
        timeout=120,
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=120, max_retries=2)
    logger.info(
        "Dual-model architecture initialized: GPT-5-mini (Agent) | GPT-4o-mini (Router/Summary)"
    )
except Exception:
    logger.exception("Error initializing ChatOpenAI LLM")
    raise
