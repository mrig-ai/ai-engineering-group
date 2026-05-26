# Chatbot 🤖

A developed RAG-based customer support chatbot, built with:

* **LangChain**
* **OpenAI GPT models**
* **In-memory vector store**
* **uv** (Python package manager)

The assistant answers strictly based on FAQ/Terms documents.

---

## 🏗️ Architecture Overview

### Core Stack

| Component | Technology |
| :--- | :--- |
| **LLM** | gpt-4.1-mini |
| **Embeddings** | text-embedding-3-small |
| **RAG** | LangChain |
| **Vector Store** | InMemoryVectorStore |
| **Dependency Manager** | uv |

---

## 📂 Project Structure

```text
src/chatbot/
│
├── agents/
│   └── models.py
│
├── graphs/
│   └── graph.py
│
├── prompts/
│   └── prompts.py
│
├── rag/
│   ├── data_loader/
│   ├── retrievers/
│   └── stores/
│
├── langgraph.json -> Entry point for the ui.
|
data/
    └── zalando_cleaned.pdf

```

## ⚙️ How It Works

The system follows a linear pipeline to ensure accuracy and safety:

1. **User input** 
2. **Context filtering**
3. **Query expansion** (multilingual → English SEO queries)
4. **Vector similarity search**
5. **Final LLM response**
6. **Response returned** in user's language


---

## 🚀 Installation

### 1️⃣ Clone the Repository
git clone [https://github.com/mrig-gmbh/chatbot.git](https://github.com/your-org/chatbot.git)

```bash
cd chatbot
```
### 2️⃣ Install Dependencies (uv)
If you don’t have **uv**:

```bash
pip install uv

### 3️⃣ Install all dependencies
```bash
uv sync

## 🔑 Environment Variables

Create a `.env` file in the project root:
- Define your model API keys in the `.env` file. 
Example:
```env
OPENAI_API_KEY=your_openai_api_key


## 🧩 LangGraph CLI Setup

### 1️⃣ Install the LangGraph CLI

> Python >= 3.11 is required.

Using **pip**:

```bash
pip install -U "langgraph-cli[inmem]"

Using **uv**:

```bash
uv add langgraph-cli[inmem]

- Then in the root of your project, run:
```bash
uv sync 

Then start the LangGraph server:
```bash
uv run langgraph dev --allow-blocking

```

- For more information about LangGraph server , visit [LangGraph](https://docs.langchain.com/oss/python/langgraph/local-server)


### When the server is up and running, you can start frontend:
- Go to https://agentchat.vercel.app/
- Press "Continue" to start the chatbot

## 🧪 Evaluation

The `src/evaluation` package provides MLflow-backed runners that exercise the real agent
through a set of canned queries and record judgements in the `chatbot-evaluation` experiment.
Each runner reuses the agent from `src.chatbot.graphs.graph.get_answer` so the results reflect
how the deployed workflow would behave.

1. **Environment setup** – add the same `.env` entry as the rest of the project and define the
   evaluation-specific helpers:

   ```env
   # point at your MLflow tracking server (leave empty for the default filesystem URI)
   MLFLOW_TRACKING_URI=
   MLFLOW_ADMIN_USER=
   MLFLOW_ADMIN_PASSWORD=
   # optional: which GenAI judge to use (default: openai:/gpt-5.4-mini)
   EVAL_JUDGE_MODEL=openai:/gpt-5.4-mini
   ```

2. **RAG evaluation** – uses `get_scorers()` (RelevanceToQuery, RetrievalRelevance, RetrievalGroundedness)
   to judge the response and the retrieved context. Run it from the repo root so that imports resolve:

   ```bash
   uv run python -m src.evaluation.rag_evaluation \
     --dataset-json test_data/rag_eval_dataset.json \
     --judge-model $EVAL_JUDGE_MODEL
   ```

   The default dataset (`test_data/rag_eval_dataset.json`) exposes two sections:
   `eval_data` (queries for the full RAG run) and `tool_calls_eval_data` (tool call expectations).
   You can point `--dataset-json` at any MLflow evaluation payload with `inputs` + `expectations` entries.

3. **Guidelines + tone scoring** – the `llm_evaluation` runner loads the same dataset but only
   evaluates language/tone criteria via `get_llm_scorers()`. It also accepts `--judge-model`
   (defaults to the same `$EVAL_JUDGE_MODEL`) so the same LLM handles all evaluations.

   ```bash
   uv run python -m src.evaluation.llm_evaluation \
     --dataset-json test_data/rag_eval_dataset.json \
     --judge-model $EVAL_JUDGE_MODEL
   ```

4. **Tool Call efficiency** – the `toolcall_evaluation` module drives MLflow metrics for tool usage.
   It also supports `--judge-model` and expects the `tool_calls_eval_data` list from the JSON file
   while using `get_toolcall_scorers()`:

   ```bash
   uv run python -m src.evaluation.toolcall_evaluation \
     --dataset-json test_data/rag_eval_dataset.json \
     --judge-model $EVAL_JUDGE_MODEL
   ```

### Running MLflow locally

- Start a tracking server from the repo root so that `mlruns.db` and `./mlruns` stay next to
  the code:

  ```bash
  mlflow server \
    --backend-store-uri sqlite:///<db name> \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 --port 5000
  ```

- Before running any evaluation script, export the URI so `mlflow` (and `mlflow.genai.evaluate`) log
  to that server:

  ```bash
  export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
  ```

- Keep the server running while you log runs, and open `http://127.0.0.1:5000` in a browser to
  inspect the experiments. Shutting down the server (Ctrl+C) preserves data in `mlruns.db`.

- If you prefer not to host the full server, run `mlflow ui --backend-store-uri sqlite:///mlruns.db`
  in another shell instead; it speaks the same API and reuses the same SQLite store.

After each run the output includes an `eval_results` table; the same data is persisted in your MLflow
tracking server so you can explore it via `mlflow ui` using the `MLFLOW_TRACKING_URI` you configured.

## Purge all deleted UI data from local database:
```bash
sqlite3 mlflow.db "
DELETE FROM latest_metrics WHERE run_uuid IN (SELECT run_uuid FROM runs WHERE lifecycle_stage = 'deleted');
DELETE FROM metrics WHERE run_uuid IN (SELECT run_uuid FROM runs WHERE lifecycle_stage = 'deleted');
DELETE FROM params WHERE run_uuid IN (SELECT run_uuid FROM runs WHERE lifecycle_stage = 'deleted');
DELETE FROM tags WHERE run_uuid IN (SELECT run_uuid FROM runs WHERE lifecycle_stage = 'deleted');
DELETE FROM runs WHERE lifecycle_stage = 'deleted' OR experiment_id IN (SELECT experiment_id FROM experiments WHERE lifecycle_stage = 'deleted');
DELETE FROM experiment_tags WHERE experiment_id IN (SELECT experiment_id FROM experiments WHERE lifecycle_stage = 'deleted');
DELETE FROM experiments WHERE lifecycle_stage = 'deleted';
VACUUM;
"
```
## 📝 License

MIT
