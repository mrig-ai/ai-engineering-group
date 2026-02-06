# SQL Client – Multi-Agent System (MCP)

This project is a sophisticated multi-agent SQL assistant built using **LangGraph**, **LangChain**, and the **Model Context Protocol (MCP)**. It leverages specialized agents to orchestrate the lifecycle of a database request: from schema discovery and SQL generation to safe execution.

---

## 🏗️ Architecture

The system uses a **Model-as-a-Tool** architecture where a central **Orchestrator** manages three specialized agents wrapped as tools:

1.  **Schema Agent**: Inspects database tables and metadata using the SQLite MCP server.
2.  **SQL Generator**: Uses `gpt-5-nano` to write SQLite-compliant queries based on provided schema.
3.  **Read Agent**: A secured agent that only executes `SELECT` queries to ensure data safety.

---

## 📂 Project Structure

```text
sql-client/
├── graph.py          # Main LangGraph workflow & MCP Client logic
├── prompt.py         # System prompts for Orchestrator and Sub-Agents
├── langgraph.json    # LangGraph platform configuration
├── pyproject.toml    # dependencies (uv managed)
├── .python-version   # Python 3.12 requirement
└── .env              # OpenAI API Keys and environment variables
```
## 🚀 Getting Started

### 1. Prerequisites
*   **Python 3.12+**
*   **[uv](https://docs.astral.sh)**: Fast Python package manager.
*   **Node.js/npx**: Required for the SQLite MCP server ([@executeautomation/database-server](https://www.npmjs.com)).

### 2. Installation
```bash
# Clone the repository
git clone <repo-url>
cd sql-client

# Install dependencies using uv
uv sync

### 3. Configuration
Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_key_here

```

---
> **Note:** The system is configured to use the **gpt-5-nano** model (2026) with the **flex** service tier.



## 🛠️ Usage

### Run the LangGraph Server
Follow the [Local Server Tutorial](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/)
To start the backend server locally for development:

```bash
uv run langgraph dev
```

### Access the UI
You can interact with the graph through the following interfaces:
*   **AgentChat:** Connect your local server to [agentchat.vercel.app](https://agentchat.vercel.app).

---
## 🛠️ Build and Run with Docker
- docker compose up --build
*   **AgentChat:** Connect your local server to [agentchat.vercel.app](https://agentchat.vercel.app).
---

## 🤖 Agent Roles

| Agent | Responsibility |
| :--- | :--- |
| **Orchestrator** | Analyzes user intent and routes to sub-agents via specialized tools. |
| **Schema Agent** | Lists tables and inspects column metadata (SQLite only). |
| **SQL Generator** | Translates natural language to SQLite-compliant SQL using schema context. |
| **Read Agent** | Executes `SELECT` statements and returns raw results. |

---

## 🔒 Security & Logic
*   **Read-Only Enforcement:** The `execute_read_query_tool` explicitly blocks any query that does not start with `SELECT` to prevent data modification.
*   **Schema Isolation:** The SQL Generator does not have direct database access; it only receives relevant schema metadata from the Orchestrator.
*   **MCP Protocol:** Uses `MultiServerMCPClient` for standardized tool communication with the database layer.

---

## 🧠 Core Technologies

### **Orchestration & AI Frameworks**
**[LangGraph](https://langchain-ai.github.io)**: Powers the stateful multi-agent orchestration and workflow logic.
*   **[LangChain](https://www.langchain.com)**: Provides the foundational building blocks for LLM communication and tool integration.
*   **[Model Context Protocol (MCP)](https://modelcontextprotocol.io)**: Enables standardized communication between the Orchestrator and the SQLite database layer.

### **Large Language Models**
*   **GPT-5-Nano (2026)**: Utilized for high-speed, SQLite-compliant query generation via OpenAI's **Flex** service tier.
*   **Model-as-a-Tool Architecture**: Implements specialized sub-agents (Schema, SQL Generator, Read Agent) as modular tools.

### **Database & Backend**
*   **SQLite**: The target relational database for all data operations.
*   **[@executeautomation/database-server](https://www.npmjs.com)**: The Node.js-based MCP server providing secure database access.
*   **Node.js**: Runtime environment for the database MCP server.

### **Development Stack**
*   **Python 3.12+**: The core language used for the graph logic.
*   **[uv](https://docs.astral.sh)**: Used for high-speed Python dependency management and environment synchronization.
*   **[LangGraph Server](https://langchain-ai.github.ioconcepts/langgraph_server/)**: Local development server for testing and debugging agentic workflows.
*   **[AgentChat](https://agentchat.vercel.app)**: Frontend interface for real-time interaction with the multi-agent system.


## 📝 License

MIT
