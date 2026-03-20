import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_pdf_docs() -> list[Document]:
    file_path = "./data/zalando_cleaned.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20, separators=["\n\n", "\n", ". ", " "])  # noqa: E501
    all_splits = text_splitter.split_documents(docs)

    return all_splits

def get_processed_docs():
    raw = get_pdf_docs()
    formatted_documents = raw
    split_docs = split_documents(formatted_documents)
    return formatted_documents, split_docs



documents, split_docs = get_processed_docs()