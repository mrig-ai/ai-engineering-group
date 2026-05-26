import argparse
import json
from typing import Literal

import mlflow
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.messages import ToolMessage
from mlflow.genai.judges import make_judge

from src.chatbot.schema.basemodel import QueryExpansion
from src.prompt_management.models import llm
from src.prompt_management.prompts_utils import (
    answer_instructions,
    multi_questions_instruction,
)

with open("test_data/evaluation_dataset.json", "r") as file:
    data = json.load(file)

multiqueries_eval_data = data.get("multi_query_prompt_dataset")
faq_eval_data = data.get("faq_prompt_dataset")

faq_prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/faq_prompt@dev")
queries_prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/multi_query_prompt@dev")


def evaluate_prompt(name: str) -> None:
    if name == "queries_prompt":

        @mlflow.trace(name="eval_queries_prompt", trace_destination="chatbot-prompts")
        def predict_fn(question: str) -> str:
            langchain_prompt = queries_prompt.to_single_brace_format()
            prompt = langchain_prompt.format(question=question)
            structured_llm = llm.with_structured_output(QueryExpansion)
            result = structured_llm.invoke(prompt)
            return {
                "question": question,
                "queries": result.questions,
                "detected_language": result.detected_language,
            }

        answer_similarity = make_judge(
            name="answer_similarity",
            instructions=multi_questions_instruction,
            model="openai:/gpt-5-mini",
            feedback_value_type=Literal["yes", "no"],
        )

        mlflow.genai.evaluate(
            data=multiqueries_eval_data,
            predict_fn=predict_fn,
            scorers=[answer_similarity],
        )
    else:

        def dummy_tool():
            """Search Zalando's FAQ documents with the given user query."""
            return "No information found in knowledge base."

        @mlflow.trace(name="eval_faq_prompt", trace_destination="chatbot-prompts")
        def predict_fn(question: str) -> str:
            agent = create_agent(
                llm, system_prompt=faq_prompt.format(), tools=[dummy_tool]
            )
            result = agent.invoke({"messages": [HumanMessage(question)]})
            messages = result.get("messages", [])

            tool_was_called = any(isinstance(m, ToolMessage) for m in messages)

            final_answer = messages[-1].content if messages else ""

            return json.dumps(
                {
                    "question": question,
                    "final_answer": final_answer,
                    "tool_call": tool_was_called,
                }
            )

        answer_similarity = make_judge(
            name="answer_similarity",
            instructions=answer_instructions,
            model="openai:/gpt-5-mini",
            feedback_value_type=Literal["yes", "no"],
        )
        mlflow.genai.evaluate(
            data=faq_eval_data,
            predict_fn=predict_fn,
            scorers=[answer_similarity],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLflow evaluation for specific prompts."
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=["queries_prompt", "faq_prompt"],
        help="The name of the prompt to evaluate.",
    )

    args = parser.parse_args()

    evaluate_prompt(args.name)

# uv run python -m src.prompt_management.prompt_evaluation --name queries_prompt
# uv run python -m src.prompt_management.prompt_evaluation --name faq_prompt
