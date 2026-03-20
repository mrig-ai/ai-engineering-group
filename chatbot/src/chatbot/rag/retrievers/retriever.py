from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from src.chatbot.rag.data_loader.data_loader import documents
from src.chatbot.rag.stores.embeddings import vector_store

VEC_STORE = vector_store
DOC_LOOKUP = {doc.metadata["page"]: doc.page_content for doc in documents}

ALL_PAGES = sorted(DOC_LOOKUP.keys())
MIN_PAGE = ALL_PAGES[0] if ALL_PAGES else 0
MAX_PAGE = ALL_PAGES[-1] if ALL_PAGES else 0


def get_chunks(
    input_text: str, vec_store: InMemoryVectorStore = VEC_STORE
) -> list[tuple[Document, float]]:
    """Retrieves relevant chunks from the vector store."""
    chunks = vec_store.similarity_search_with_score(input_text, k=2)
    return chunks


def get_docs(
    chunks: list[tuple[Document, float]], doc_lookup: dict = DOC_LOOKUP
) -> list[tuple[int, str]]:
    """
    Identifies unique pages and their immediate neighbors based on retrieved chunks.
    """
    pages_to_include = set()

    for chunk, _score in chunks:
        page = chunk.metadata.get("page")

        if page is not None and page in doc_lookup:
            neighbors = {
                p
                for p in {page - 1, page, page + 1}
                if MIN_PAGE <= p <= MAX_PAGE and p in doc_lookup
            }
            pages_to_include.update(neighbors)

    docs = [(p, doc_lookup[p]) for p in sorted(pages_to_include)]
    return docs
