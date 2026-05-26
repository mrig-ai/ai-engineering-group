import os

import mlflow
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_ADMIN_USER")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_ADMIN_PASSWORD")

uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(uri)
mlflow.set_experiment("chatbot-prompts")

mlflow.langchain.autolog(silent=True)


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    client="async",
    stream_usage=True,
    timeout=60,
    max_retries=3,
)
