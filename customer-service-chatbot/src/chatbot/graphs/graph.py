import asyncio

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.chatbot.agents.models import llm
from src.chatbot.prompts.prompts import faq_prompt
from src.chatbot.rag.retrievers.retriever import get_chunks, get_docs


class QueryExpansion(BaseModel):
    queries: list[str] = Field(min_items=3,
        max_items=3,
        description=(
            "A list of exactly 3 search queries: "
            "1. Literal: The user question translated to English (or spelling-fixed). "
            "2. SEO: The core keywords only, removing filler words. "
            "3. Variation: A 3-4 word version using synonyms, converting symbols like '+' or '&' into words (e.g., 'Plus', 'and')."
        )
    )
    detected_language: str = Field(
        description="The language name the user is speaking (e.g., German, English, French, Spanish, Portuguese)."  # noqa: E501
    )


async def get_multi_queries(query: str) -> QueryExpansion:
    """
    Process a User Question into three specific formats.

    :param query: The user query to process.
    :return: An object containing the original query and the processed queries.
    """
    
    prompt = f"""
    Act as a high-precision linguistic pre-processor for a search engine. 
    Your goal is to transform the **User Question** into optimized search queries while identifying the source language.

    ### CORE LOGIC:
    1. **Language Detection:** Identify the language of the User Question.
    2. **Standardization:** - If the question is NOT in English: Translate it to English first. 
    - If it IS in English: Fix typos but keep the original sentence structure intact.
    3. **Symbol Handling:** - Preserve special characters like '+' or '&' in technical/brand contexts for Queries 1 and 2.
    - Convert them to words (e.g., "plus", "and") ONLY for Query 3.

    ### USER QUESTION:
    "{query}"

    ### EXAMPLES:
    - User: "¿Cómo puedo contactar a Zalando Plus?"
    1. How can I contact Zalando Plus?
    2. contact Zalando Plus
    3. Zalando Plus membership support
    - User: "where is my order & shiping info"
    1. Where is my order & shipping info?
    2. order & shipping info
    3. order and delivery status
    """
    structured_llm = llm.with_structured_output(QueryExpansion)
    result = await structured_llm.ainvoke(prompt)
    return result


@tool
async def search_zalando_faq(query: str) -> str:
    """
    Search Zalando's FAQ documents with the given user query.

    Args:
        query (str): The user query to search for.

    Returns:
        str: Context string containing results and language instructions.
    """
    expansion = await get_multi_queries(query)
    search_queries = getattr(expansion, "queries", [query])
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
            # When you use one newline, the model sees the text as a single "chunk" of information. 
            # It keeps the logical flow tight. The "Separator" Signal: Most LLMs are 
            # trained on web data and Markdown. In those formats, \n\n is the universal 
            # signal that one topic has ended and a new one is beginning.
            merged_parts.append(separator + content)

    context_str = "".join(merged_parts)

    return (
        f"CRITICAL: The user is speaking {target_lang}. "
        f"You MUST respond in {target_lang}.\n\n"
        f"KNOWLEDGE BASE CONTEXT:\n{context_str}"
    )


async def get_answer():
    agent = create_agent(
        llm, 
        system_prompt=faq_prompt, 
        tools=[search_zalando_faq]
        )
    
    return agent
