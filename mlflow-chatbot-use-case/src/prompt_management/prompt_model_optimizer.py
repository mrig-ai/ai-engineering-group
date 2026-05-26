import argparse  # noqa: I001
import mlflow
import os
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Equivalence

from src.prompt_management.models import llm, new_llm
from src.chatbot.rag.retrievers.retriever import get_chunks, get_docs
from src.chatbot.schema.basemodel import QueryExpansion

uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(uri)
mlflow.set_experiment("chatbot-prompts")

mlflow.langchain.autolog(silent=True)

multiqueries_eval_data = mlflow.genai.get_dataset(
    dataset_id="d-88d402315619419081c6471e842382f2"
)
faq_eval_data = mlflow.genai.get_dataset(
    dataset_id="d-f86e7d4782174b4ca75dbf439214ebd3"
)

faq_prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/faq_prompt@dev")
queries_prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/multi_query_prompt@dev")


def optimize_model_prompt(name: str) -> None:
    if name == "queries_prompt":

        @mlflow.trace
        def predict_fn(question: str) -> dict:
            prompt = queries_prompt.to_single_brace_format()
            structured_llm = new_llm.with_structured_output(QueryExpansion)
            result = structured_llm.invoke(prompt.format(question=question))
            return {
                "questions": result.questions,
                "detected_language": result.detected_language,
            }

        mlflow.genai.optimize_prompts(
            predict_fn=predict_fn,
            train_data=multiqueries_eval_data,
            prompt_uris=[queries_prompt.uri],
            optimizer=GepaPromptOptimizer(
                reflection_model="openai:/gpt-5-mini",
                max_metric_calls=25,
                display_progress_bar=True,
            ),
            scorers=[
                Equivalence(
                    model="openai:/gpt-5-mini",
                    inference_params={"reasoning_effort": "medium"},
                )
            ],
        )
    else:

        def get_multi_queries(question: str) -> QueryExpansion:
            """
            Process a User Question into three specific formats.

            :param query: The user query to process.
            :return: An object containing the original query and the processed queries.
            """
            prompt = queries_prompt.to_single_brace_format()
            prompt = prompt.format(question=question)
            structured_llm = llm.with_structured_output(QueryExpansion)
            result = structured_llm.invoke(prompt)
            return result

        @tool
        def search_zalando_faq(question: str) -> str:
            """
            Search Zalando's FAQ documents with the given user question.

            Args:
                question (str): The user question to search for.

            Returns:
                str: Context string containing results and language instructions.
            """
            expansion = get_multi_queries(question)
            search_queries = getattr(expansion, "questions", [question])
            target_lang = getattr(expansion, "detected_language", "the user's language")

            results = [get_docs(get_chunks(q)) for q in search_queries]

            unique_docs = sorted(set().union(*results), key=lambda x: x[0])

            if not unique_docs:
                return f"CRITICAL: Respond in {target_lang}.\n\nKNOWLEDGE BASE CONTEXT:\nNo context available."

            merged_parts = []
            for i, (page, content) in enumerate(unique_docs):
                if i == 0:
                    merged_parts.append(content)
                else:
                    prev_page = unique_docs[i - 1][0]
                    separator = "\n" if page == prev_page + 1 else "\n\n"
                    merged_parts.append(separator + content)

            context_str = "".join(merged_parts)

            return (
                f"CRITICAL: The user is speaking {target_lang}. "
                f"You MUST respond in {target_lang}.\n\n"
                f"KNOWLEDGE BASE CONTEXT:\n{context_str}"
            )

        @mlflow.trace
        def predict_fn(question: str) -> str:
            prompt = faq_prompt
            agent = create_agent(
                new_llm, system_prompt=prompt.format(), tools=[search_zalando_faq]
            )
            result = agent.invoke({"messages": [HumanMessage(question)]})
            messages = result.get("messages", [])
            final_answer = messages[-1].content if messages else ""

            return final_answer

        mlflow.genai.optimize_prompts(
            predict_fn=predict_fn,
            train_data=faq_eval_data,
            prompt_uris=[faq_prompt.uri],
            optimizer=GepaPromptOptimizer(
                reflection_model="openai:/gpt-5-mini",
                max_metric_calls=30,
                display_progress_bar=True,
            ),
            scorers=[
                Equivalence(
                    model="openai:/gpt-5-mini",
                    inference_params={"reasoning_effort": "medium"},
                )
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLflow GEPA Prompt Optimization for specific prompts."
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=["queries_prompt", "faq_prompt"],
        help="The name of the prompt template to optimize.",
    )

    args = parser.parse_args()

    optimize_model_prompt(args.name)

# uv run python -m src.prompt_management.prompt_model_optimizer --name queries_prompt
# uv run python -m src.prompt_management.prompt_model_optimizer --name faq_prompt
