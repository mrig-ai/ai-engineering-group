from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    client="async"
)