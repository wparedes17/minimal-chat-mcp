import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful data assistant that helps users explore and query database tables.

You have access to two tools:
- get_schema: Retrieves the schema (columns, types, structure) for a given table.
  Use this whenever the user mentions a table you have not inspected yet in this conversation.
- run_query: Executes a SQL query against the database and returns the results.

Guidelines:
1. Always inspect the table schema before writing queries if you don't already know its structure.
2. Build correct PostgreSQL SQL queries based on the schema information you retrieve.
3. If run_query returns an error, analyse the error message carefully and retry with a
   corrected query. You may retry up to 3 times before giving up and explaining the problem.
4. Only execute SELECT queries — never modify, insert, update, or delete data.
5. Present query results in a clear, readable way and include helpful context for the user.
"""

# ── Pydantic models ────────────────────────────────────────────────────────────


class ConversationOut(BaseModel):
    id: str
    created_at: str


class ChatInput(BaseModel):
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    response: str


class HistoryMessage(BaseModel):
    role: str   # "human" | "assistant"
    content: str


class HistoryResponse(BaseModel):
    conversation_id: str
    messages: list[HistoryMessage]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_headers() -> dict[str, str]:
    """
    Read MCP_HEADERS from the environment.
    Expected format: a JSON object, e.g.
        MCP_HEADERS={"Authorization": "Bearer token", "X-Api-Key": "abc"}
    Returns an empty dict when the variable is unset or empty.
    """
    raw = os.environ.get("MCP_HEADERS", "").strip()
    if not raw:
        return {}
    try:
        headers = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MCP_HEADERS is not valid JSON: {exc}") from exc
    if not isinstance(headers, dict):
        raise RuntimeError("MCP_HEADERS must be a JSON object, e.g. {\"Authorization\": \"Bearer …\"}")
    return headers


async def _check_health(health_url: str, headers: dict[str, str]) -> None:
    """Hit the MCP server's health endpoint before the agent is built."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(health_url, headers=headers, timeout=10)
        resp.raise_for_status()


# ── App lifespan: connect to MCP and build the agent once ──────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    mcp_url = os.environ.get("MCP_SERVER_URL")
    if not mcp_url:
        raise RuntimeError("MCP_SERVER_URL environment variable is required")

    headers = _parse_headers()

    # Optional: verify the MCP server is reachable before building the agent
    health_url = os.environ.get("MCP_HEALTH_URL", "").strip()
    if health_url:
        try:
            await _check_health(health_url, headers)
        except Exception as exc:
            raise RuntimeError(f"MCP health check failed ({health_url}): {exc}") from exc

    mcp_config = {
        "my-mcp": {
            "url": mcp_url,
            "transport": os.environ.get("MCP_TRANSPORT", "streamable_http"),
            # headers are forwarded with every tool call to the MCP server
            **({"headers": headers} if headers else {}),
        }
    }

    async with MultiServerMCPClient(mcp_config) as mcp_client:
        tools = await mcp_client.get_tools()

        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            temperature=0,
        )

        checkpointer = MemorySaver()

        # state_modifier prepends the system prompt to every agent invocation
        agent = create_react_agent(
            llm,
            tools,
            checkpointer=checkpointer,
            state_modifier=SystemMessage(content=SYSTEM_PROMPT),
        )

        app.state.agent = agent
        app.state.conversations: dict[str, dict] = {}
        yield


app = FastAPI(title="Minimal Chat MCP", version="0.1.0", lifespan=lifespan)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/conversations", response_model=ConversationOut, status_code=201)
async def create_conversation(request: Request):
    """
    Start a new conversation.
    Returns the conversation ID that must be used in all subsequent calls.
    """
    conv_id = str(uuid.uuid4())
    request.app.state.conversations[conv_id] = {
        "id": conv_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return request.app.state.conversations[conv_id]


@app.post("/conversations/{conv_id}/chat", response_model=ChatResponse)
async def chat(conv_id: str, body: ChatInput, request: Request):
    """
    Send a user message and get the agent's reply.

    The agent will:
    - Call get_schema when it needs to inspect a table.
    - Call run_query to fetch data; it retries up to 3 times on SQL errors.
    - Return a natural-language response with results or an explanation.
    """
    if conv_id not in request.app.state.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    agent = request.app.state.agent
    config = {
        "configurable": {"thread_id": conv_id},
        # Each ReAct step = 1 node execution.
        # 20 is enough for: schema fetch + up to 3 query attempts + reasoning steps.
        "recursion_limit": 20,
    }

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    last = result["messages"][-1]
    # AIMessage.content is a str in text responses; guard against content blocks
    if isinstance(last.content, str):
        response_text = last.content
    elif isinstance(last.content, list):
        # Extract text blocks from a mixed content list (e.g. Claude tool-use format)
        response_text = " ".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in last.content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    else:
        response_text = str(last.content)

    return ChatResponse(conversation_id=conv_id, response=response_text)


@app.get("/conversations/{conv_id}/history", response_model=HistoryResponse)
async def get_history(conv_id: str, request: Request):
    """
    Return the full human/assistant message history for a conversation.
    Internal tool calls and tool results are excluded.
    """
    if conv_id not in request.app.state.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    agent = request.app.state.agent
    state = await agent.aget_state({"configurable": {"thread_id": conv_id}})
    all_messages = state.values.get("messages", [])

    visible: list[HistoryMessage] = []
    for m in all_messages:
        if not isinstance(m, (HumanMessage, AIMessage)):
            continue
        # Skip AIMessages that only contain tool-call requests (no text for the user)
        content = m.content
        if isinstance(content, list):
            text_parts = [
                block["text"] if isinstance(block, dict) else str(block)
                for block in content
                if not isinstance(block, dict) or block.get("type") == "text"
            ]
            content = " ".join(text_parts).strip()
        if not content:
            continue
        visible.append(
            HistoryMessage(
                role="human" if isinstance(m, HumanMessage) else "assistant",
                content=content,
            )
        )

    return HistoryResponse(conversation_id=conv_id, messages=visible)


@app.delete("/conversations/{conv_id}", status_code=204)
async def delete_conversation(conv_id: str, request: Request):
    """
    Delete a conversation.
    The in-memory LangGraph checkpoint is not explicitly freed but will be
    garbage-collected when the MemorySaver reference is dropped on restart.
    """
    if conv_id not in request.app.state.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del request.app.state.conversations[conv_id]
