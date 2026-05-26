import asyncio

import mlflow
from langchain.agents import create_agent
from langchain_core.tools import tool
from src.chatbot.agents.models import llm

# from src.chatbot.prompts.prompts import faq_prompt, multi_query_prompt
from src.chatbot.rag.retrievers.retriever import get_chunks, get_docs
from src.chatbot.schema.basemodel import QueryExpansion

faq_prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/faq_prompt@dev")
multi_query_prompt = mlflow.genai.load_prompt(
    name_or_uri="prompts:/multi_query_prompt@dev"
)


@mlflow.trace
async def get_multi_queries(query: str) -> QueryExpansion:
    """
    Process a User Question into three specific formats.

    :param query: The user query to process.
    :return: An object containing the original query and the processed queries.
    """
    prompt = multi_query_prompt.to_single_brace_format()
    structured_llm = llm.with_structured_output(QueryExpansion)
    result = await structured_llm.ainvoke(prompt.format(question=query))
    return result


@tool
@mlflow.trace(span_type="TOOL", name="search_zalando_faq")
async def search_zalando_faq(query: str) -> str:
    """
    Search Zalando's FAQ documents with the given user query.

    Args:
        query (str): The user query to search for.

    Returns:
        str: Context string containing results and language instructions.
    """
    expansion = await get_multi_queries(query)
    search_queries = getattr(expansion, "questions", [query])
    # print(f"Search queries: {search_queries}")
    target_lang = getattr(expansion, "detected_language", "the user's language")

    def retrieve_docs_sync(q: str) -> list[tuple[int, str]]:
        chunks = get_chunks(q)
        return get_docs(chunks)

    results = await asyncio.gather(
        *(asyncio.to_thread(retrieve_docs_sync, q) for q in search_queries)
    )

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


async def get_answer():
    agent = create_agent(
        llm, system_prompt=faq_prompt.format(), tools=[search_zalando_faq]
    )

    return agent
