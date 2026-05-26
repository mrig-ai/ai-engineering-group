"""MLflow evaluation for tone and language guidelines."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import mlflow
from langchain.messages import HumanMessage
from mlflow.genai.scorers import Correctness
from src.chatbot.graphs.graph import get_answer
from src.evaluation.config import JUDGE_MODEL, setup_mlflow
from src.evaluation.scorers import get_llm_scorers

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "test_data" / "rag_eval_dataset.json"
)
PREDICT_TIMEOUT_SECONDS = 120


def load_eval_dataset(path: Path | str = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """Load eval_data from the dataset JSON file."""
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    entries = payload.get("faq_dataset", [])
    if not entries:
        raise ValueError(f"No eval_data entries found in {dataset_path}")

    return entries


@mlflow.trace(span_type="PREDICTION", name="predict_fn_guidelines")
def predict_fn(question: str) -> str:
    """Call the chatbot agent end-to-end for a single query."""

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


def run_guidelines_evaluation(
    dataset: list[dict[str, Any]],
    judge_model: str = JUDGE_MODEL,
) -> mlflow.models.EvaluationResult:
    """Run MLflow evaluation with tone + language Guidelines scorers."""
    setup_mlflow()

    llm_scorers = get_llm_scorers(model=judge_model)
    llm_scorers.append(Correctness())
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=llm_scorers,
    )

    print("\nEvaluation complete. Summary:")
    print(results.tables["eval_results"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tone and language guidelines evaluation."
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the JSON file containing eval_data.",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help="Judge model in <provider>:/<model> format.",
    )
    args = parser.parse_args()

    dataset = load_eval_dataset(args.dataset_json)
    run_guidelines_evaluation(dataset=dataset, judge_model=args.judge_model)


if __name__ == "__main__":
    main()
