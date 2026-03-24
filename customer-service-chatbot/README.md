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

## 📝 License

MIT