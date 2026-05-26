"""
MLflow RAG scorer definitions.

Provides factory functions for the three LLM-judge scorers used
to evaluate the chatbot's retrieval and generation quality.

Scorers
-------
- RelevanceToQuery   : Does the response address the user's question?
- RetrievalRelevance : Is each retrieved document relevant to the query?
- RetrievalGroundedness : Is the response grounded in the retrieved context?

All three are *trace-only* judges — no ground-truth labels required.
"""

from mlflow.genai.scorers import (
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    ToolCallEfficiency,
    Guidelines,
)
from src.evaluation.config import JUDGE_MODEL


def get_scorers(
    model: str = JUDGE_MODEL,
) -> list:
    """
    Return configured scorer instances.

    Parameters
    ----------
    model : str
        LLM endpoint powering the judges, in ``<provider>:/<model>`` format.

    Returns
    -------
    list
        List of scorer instances ready for ``mlflow.genai.evaluate()``.
    """
    return [
        RelevanceToQuery(model=model),
        RetrievalRelevance(model=model),
        RetrievalGroundedness(model=model),
    ]


def get_toolcall_scorers(
    model: str = JUDGE_MODEL,
) -> list:
    """
    Return configured scorer instances.

    Parameters
    ----------
    model : str
        LLM endpoint powering the judges, in ``<provider>:/<model>`` format.

    Returns
    -------
    list
        List of scorer instances ready for ``mlflow.genai.evaluate()``.
    """
    return [
        ToolCallEfficiency(model=model),
    ]


def get_llm_scorers(
    model: str = JUDGE_MODEL,
) -> list:
    """
    Return configured scorer instances.

    Parameters
    ----------
    model : str
        LLM endpoint powering the judges, in ``<provider>:/<model>`` format.

    Returns
    -------
    list
        List of scorer instances ready for ``mlflow.genai.evaluate()``.
    """
    tone_guidelines = Guidelines(
        name="tone",
        guidelines=[
            "The response must be friendly in tone.",
            "The response must sound like professional customer service.",
            "The response must be concise while still complete and clear.",
            "The response must never be dismissive, rude, overly terse, or abrupt.",
            "The response should avoid unnecessary filler phrases (e.g., 'Based on the information provided...').",
            "For greetings or small talk, the response should be warm and natural.",
        ],
    )

    language_guidelines = Guidelines(
        name="language_match",
        guidelines=[
            "The response must be written in exactly the same language as the user's question.",
            "If the question is in French, the response must be entirely in French.",
            "If the question is in German, the response must be entirely in German.",
            "If the question is in Portuguese, the response must be entirely in Portuguese.",
            "If the question is in Dutch, the response must be entirely in Dutch.",
            "If the question is in Italian, the response must be entirely in Italian.",
            "If the question is in Polish, the response must be entirely in Polish.",
            "Never respond in English when the user's question is in a different language.",
        ],
    )
    tone = Guidelines(
        name="tone",
        guidelines=tone_guidelines.guidelines,
        model=model,
    )
    language = Guidelines(
        name="language_match",
        guidelines=language_guidelines.guidelines,
        model=model,
    )

    return [tone, language]
