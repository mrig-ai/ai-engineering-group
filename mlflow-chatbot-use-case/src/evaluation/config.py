import os

import mlflow
from dotenv import load_dotenv

load_dotenv()

# ── MLflow tracking ──────────────────────────────────────────────
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_ADMIN_USER", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_ADMIN_PASSWORD", "")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT = "chatbot-evaluation"

# ── Judge model ──────────────────────────────────────────────────
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai:/gpt-5.4-mini")


def setup_mlflow() -> None:
    """Initialize MLflow tracking for the evaluation experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    mlflow.autolog(disable=True)
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"
