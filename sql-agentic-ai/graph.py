from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from prompt import (
    LIST_TABLES_AND_STRUCTURE_PROMPT,
    READ_QUERY_PROMPT,
    SQL_QUERY_GENERATOR_PROMPT,
    ORCHESTRATOR_PROMPT,
)


class ColumnInfo(BaseModel):
    name: str
    type: str
    primary_key: bool = False
    nullable: bool = True
    default: Optional[Any] = None

class TableStructure(BaseModel):
    table_name: str
    columns: List[ColumnInfo]

class InspectStructureOutput(BaseModel):
    tables: List[str]
    structure: Optional[List[TableStructure]] = None

class GenerateSQLOutput(BaseModel):
    query: str

class QueryResponse(BaseModel):
    results: List[dict] = Field(description="The data returned from the database")


load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-mini"
    #  , service_tier="flex"
)

base_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(base_dir, "db", "chinook.db")

_CACHED_TOOLS = None


class Client(MultiServerMCPClient):
    _client_instance = None

    def __new__(cls, db_path):
        if cls._client_instance is None:
            cls._client_instance = super(Client, cls).__new__(cls)
            cls._client_instance._initialized = False
        return cls._client_instance

    def __init__(self, db_path):
        if getattr(self, "_initialized", False):
            return

        servers_config = {
            "sqlite-server": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@executeautomation/database-server", db_path],
            }
        }
        super().__init__(servers_config)
        self._initialized = True


mcp_client = Client(DB_PATH)


async def make_graph():
    global _CACHED_TOOLS

    if _CACHED_TOOLS is None:
        _CACHED_TOOLS = await mcp_client.get_tools()

    tools = _CACHED_TOOLS
    structure_tools = [t for t in tools if t.name in {"list_tables", "describe_table"}]
    read_query_tools = [t for t in tools if t.name == "read_query"]

    structure_agent = create_agent(
        model=llm,
        tools=structure_tools,
        system_prompt=LIST_TABLES_AND_STRUCTURE_PROMPT,
        response_format=InspectStructureOutput,
    )

    sql_generator_agent = create_agent(
        model=llm,
        system_prompt=SQL_QUERY_GENERATOR_PROMPT,
        response_format=GenerateSQLOutput,
    )

    read_query_agent = create_agent(
        model=llm,
        tools=read_query_tools,
        system_prompt=READ_QUERY_PROMPT,
        response_format=QueryResponse
    )

    @tool()
    async def inspect_structure_tool(question: str) -> Dict[str, Any]:
        """
        Inspect database structure, tables, and columns.

        Returns a structured Pydantic object with:
            tables: list of table names
            structure: list of table structures with columns
        """
        result = await structure_agent.ainvoke({"messages": [("user", question)]})
        structured_response = result["structured_response"]
        return structured_response

    @tool()
    async def generate_sql_tool(
        question: str,
        tables: list[str],
        structure: str,
    ) -> str:
        """
        Generate a read-only **SELECT** queries using tables names and structure metadata (SQLite-only).

        Returns a structured Pydantic object with:
            query: str
        """
        tables_normalized = [t.lower() for t in tables]
        prompt = f"""
            User question:
            {question}

            Available tables:
            {tables_normalized}

            Database structure:
            {structure}

            Rules:
            - SQLite syntax only
            - **Read-only SELECT** queries only
            - Generate exactly ONE query
            - DO NOT generate INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, WITH or multiple statements
            """
        sql_code = await sql_generator_agent.ainvoke({"messages": [("user", prompt)]})
        sql_structured_response = sql_code["structured_response"].query
        return sql_structured_response

    @tool()
    async def execute_read_query_tool(input: Any) -> List[Dict[str, Any]]:
        """
        Execute a read-only **SELECT** SQL query. 
        Accepts a SQL string or a dict like {'query': 'SELECT...'}.
        """
        query = ""
        if isinstance(input, dict):
            query = input.get("query", "")
        else:
            query = str(input)

        valid_starts = ("select",)
        if not query.lower().startswith(valid_starts):
            raise ValueError(f"Only SELECT or queries are allowed. Received: {query[:50]}...")
        
        final_query = await read_query_agent.ainvoke({"messages": [("user", query)]})

        final_query_structured_response = final_query["structured_response"].results
        return final_query_structured_response

    return create_agent(
        model=llm,
        tools=[inspect_structure_tool, generate_sql_tool, execute_read_query_tool],
        system_prompt=ORCHESTRATOR_PROMPT,
    )
