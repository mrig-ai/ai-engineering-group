import argparse
import os

import mlflow
from mlflow.genai.optimize import MetaPromptOptimizer

from src.prompt_management.prompts_utils import (
    answer_guidelines,
    multi_questions_guidelines,
)

uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(uri)
mlflow.set_experiment("chatbot-prompts")

mlflow.langchain.autolog(silent=True)

faq_prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/faq_prompt@dev")
queries_prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/multi_query_prompt@dev")

@mlflow.trace(name="prompt_optimizer")
def optimize_prompt(name: str) -> None:
    if name == "queries_prompt":
        mlflow.genai.optimize_prompts(
            predict_fn=lambda question: "",
            train_data=[],
            prompt_uris=[queries_prompt.uri],
            optimizer=MetaPromptOptimizer(
                reflection_model="openai:/gpt-5-mini",
                guidelines=multi_questions_guidelines,
            ),
            scorers=[],
        )
    else:
        mlflow.genai.optimize_prompts(
            predict_fn=lambda question: "",
            train_data=[],
            prompt_uris=[faq_prompt.uri],
            optimizer=MetaPromptOptimizer(
                reflection_model="openai:/gpt-5-mini",
                guidelines=answer_guidelines,
            ),
            scorers=[],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow Meta-Prompt Optimization.")

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=["queries_prompt", "faq_prompt"],
        help="The name of the prompt to optimize (queries_prompt or faq_prompt).",
    )

    args = parser.parse_args()

    optimize_prompt(args.name)

# uv run python -m src.prompt_management.prompt_optimizer --name queries_prompt
# uv run python -m src.prompt_management.prompt_optimizer --name faq_prompt
