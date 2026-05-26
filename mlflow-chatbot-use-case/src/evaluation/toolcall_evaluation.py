"""MLflow wrapper to score the chatbot's tool usage via ToolCallEfficiency."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import mlflow
from langchain.messages import HumanMessage
from mlflow.tracing.fluent import get_current_active_span
from mlflow.tracing.utils import set_span_chat_tools
from src.evaluation.scorers import get_toolcall_scorers

from src.chatbot.graphs.graph import get_answer
from src.evaluation.config import JUDGE_MODEL, setup_mlflow

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "test_data" / "rag_eval_dataset.json"
)
PREDICT_TIMEOUT_SECONDS = 120


def load_tool_call_dataset(path: Path | str = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """Load and normalize the tool call evaluation data file."""

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Tool-call dataset not found at {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    entries = payload.get("tool_calls_eval_data", [])
    if not entries:
        raise ValueError(f"No tool call evaluation entries found in {dataset_path}")

    return entries


@mlflow.trace(span_type="PREDICTION", name="predict_fn_toolcall")
def predict_fn_toolcall(question: str) -> str:
    """Call the chatbot agent end-to-end for a single tool call evaluation query."""

    # current_span = get_current_active_span()
    # if current_span is not None:
    #     set_span_chat_tools(current_span, TOOL_DEFINITIONS)

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
        return output

    try:
        return asyncio.run(_invoke_agent())
    except asyncio.TimeoutError:
        print(
            f"Warning: query timed out after {PREDICT_TIMEOUT_SECONDS}s — skipping: {question!r}"
        )
        return ""


def run_toolcall_evaluation(
    dataset: list[dict[str, Any]], judge_model: str = JUDGE_MODEL
) -> mlflow.models.EvaluationResult:
    """Run the MLflow evaluation with the ToolCallEfficiency scorer."""

    setup_mlflow()
    scorers = get_toolcall_scorers(model=judge_model)

    print(f" Running ToolCallEfficiency scoring on {len(dataset)} queries...")
    print(f" Judge model: {judge_model}")

    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn_toolcall,
        scorers=scorers,
    )

    print("\nEvaluation complete. Summary:")
    print(results.tables["eval_results"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a ToolCallEfficiency-only evaluation."
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the JSON file containing the tool call evaluation data.",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help="Judge model for ToolCallEfficiency in <provider>:/<model> format.",
    )
    args = parser.parse_args()

    dataset = load_tool_call_dataset(args.dataset_json)
    run_toolcall_evaluation(dataset=dataset, judge_model=args.judge_model)


if __name__ == "__main__":
    main()
