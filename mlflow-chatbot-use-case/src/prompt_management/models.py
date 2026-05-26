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

mlflow.langchain.autolog()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
new_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
