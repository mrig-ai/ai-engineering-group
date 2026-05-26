"""
RAG Evaluation Runner.

Runs the chatbot's actual agent pipeline against a set of test queries and
evaluates the results using MLflow's built-in LLM judges.

Usage
-----
    python -m evaluation.rag_evaluation          # run from src/
    python -m evaluation.rag_evaluation --help   # show options
"""

import argparse
import asyncio
import json
from pathlib import Path

from rich import print
import mlflow

from src.evaluation.config import JUDGE_MODEL, setup_mlflow
from src.evaluation.scorers import get_scorers
from langchain.messages import HumanMessage

# Import the main agent from the project
from src.chatbot.graphs.graph import get_answer

SAMPLE_QUERIES = [
    {"inputs": {"query": "How can I return an item on Zalando?"}},
    {"inputs": {"query": "What is Zalando's shipping policy?"}},
    {"inputs": {"query": "How do I track my order?"}},
    {"inputs": {"query": "What payment methods does Zalando accept?"}},
    {"inputs": {"query": "How do I cancel my order?"}},
]


def load_queries_from_json(path: Path) -> list[dict]:
    """Load a list of MLflow evaluation query dicts from a JSON file."""

    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    return data


@mlflow.trace(span_type="RETRIEVER", name="extract_retrieved_context")
def _extract_retrieved_context(query: str, messages: list) -> list:
    """
    Marked as a RETRIEVER span so RetrievalRelevance and RetrievalGroundedness
    scorers evaluate exactly this extracted context.
    """
    kb_marker = "KNOWLEDGE BASE CONTEXT:\n"
    retrieved = []
    for msg in messages:
        if msg.__class__.__name__ == "ToolMessage" and msg.content:
            raw = msg.content
            content = (
                raw[raw.index(kb_marker) + len(kb_marker) :]
                if kb_marker in raw
                else raw
            )
            retrieved.append({"page_content": content})
    return retrieved


PREDICT_TIMEOUT_SECONDS = 120


@mlflow.trace(span_type="PREDICTION", name="predict_fn_rag")
def predict_fn_rag(question: str) -> str:
    """
    End-to-end predict function that calls the real chatbot agent.

    Since the agent uses async LangChain methods, we run it inside
    an asyncio event loop so `predict_fn` remains a synchronous API
    for `mlflow.genai.evaluate`.
    """

    async def _invoke_agent():
        agent = await get_answer()
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [HumanMessage(content=question)]}),
            timeout=PREDICT_TIMEOUT_SECONDS,
        )
        messages = result.get("messages", [])
        output = result.get("output", "")
        if not output:
            for msg in reversed(messages):
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", [])
                if content and not tool_calls and msg.__class__.__name__ == "AIMessage":
                    output = content
                    break

        _extract_retrieved_context(question, messages)
        return output

    try:
        return asyncio.run(_invoke_agent())
    except asyncio.TimeoutError:
        print(
            f"Warning: query timed out after {PREDICT_TIMEOUT_SECONDS}s — skipping: {question!r}"
        )
        return ""


def run_evaluation(
    queries: list[dict] | None = None,
    judge_model: str = JUDGE_MODEL,
) -> "mlflow.models.EvaluationResult":
    setup_mlflow()

    data = queries or SAMPLE_QUERIES
    scorers = get_scorers(model=judge_model)

    print(f" Running evaluation on {len(data)} queries ...")
    print(f" Scorers: {[type(s).__name__ for s in scorers]}")
    print(f" Judge model: {judge_model}\n")

    # Collect predictions first so RetrievalRelevance and RetrievalGroundedness
    # score against the already-extracted retrieved_context, not live retriever traces.
    results = mlflow.genai.evaluate(
        data=data,
        predict_fn=predict_fn_rag,
        scorers=scorers,
    )

    print("\n Evaluation complete!")
    print(results.tables["eval_results"])

    return results


def run_evaluation_from_traces(
    traces: list,
    judge_model: str = JUDGE_MODEL,
) -> "mlflow.models.EvaluationResult":
    setup_mlflow()
    scorers = get_scorers(model=judge_model)

    print(f" Evaluating {len(traces)} existing traces ...")
    print(f" Scorers: {[type(s).__name__ for s in scorers]}")
    print(f" Judge model: {judge_model}\n")

    results = mlflow.genai.evaluate(
        data=traces,
        scorers=scorers,
    )

    print("\n Evaluation complete!")
    print(results.tables["eval_results"])
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation with MLflow judges.",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help=f"Judge LLM in <provider>:/<model> format (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        help="Path to a JSON file that defines the evaluation queries (default: built-in sample set)",
    )
    args = parser.parse_args()
    queries = None
    if args.dataset_json:
        queries = load_queries_from_json(args.dataset_json)
        print(f"Loaded {len(queries)} queries from {args.dataset_json}")
    run_evaluation(queries=queries["eval_data"], judge_model=args.judge_model)


if __name__ == "__main__":
    main()
