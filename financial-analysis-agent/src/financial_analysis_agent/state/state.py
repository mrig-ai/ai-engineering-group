import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentInput(TypedDict):
    """Simple input state for each subagent."""

    query: str


class AgentOutput(TypedDict):
    """Output from each subagent."""

    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""

    source: Literal[
        "financial_agent", "news_agent", "recomendation_agent", "macro_agent"
    ]
    query: str


class RouterState(TypedDict, total=False):
    messages: Annotated[list[dict], add_messages]
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str


class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""

    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )
