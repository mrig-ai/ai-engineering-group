from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from src.chatbot.rag.data_loader.data_loader import split_docs

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store =InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=split_docs)