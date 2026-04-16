# minimal-chat-mcp

A minimal FastAPI service that wraps a LangGraph ReAct agent connected to a remote MCP server.  
The agent can inspect table schemas and run SQL queries on behalf of the user, retrying failed queries up to 3 times.

## Architecture

```
Client ──POST /conversations/{id}/chat──► FastAPI
                                              │
                                     LangGraph ReAct agent
                                      (MemorySaver, per thread_id)
                                              │
                                    ┌─────────┴──────────┐
                                get_schema           run_query
                                    └─────────┬──────────┘
                                         MCP Server
                                    (your existing API)
```

- **Conversations** are stored in memory keyed by UUID. State lives in LangGraph's `MemorySaver` (lost on restart).
- The **ReAct agent** uses `get_schema` and `run_query` MCP tools. It retries failed queries up to 3 times before explaining the issue to the user.
- The MCP server URL is the same value you use in Windsurf's `serverUrl`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and fill in MCP_SERVER_URL and ANTHROPIC_API_KEY
```

### `.env` variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MCP_SERVER_URL` | Yes | — | URL of your MCP server, e.g. `https://myhost:8080/mcp` |
| `ANTHROPIC_API_KEY` | Yes | — | Your Anthropic API key |
| `MCP_TRANSPORT` | No | `streamable_http` | `streamable_http` or `sse` |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Claude model ID |

## Run

```bash
uvicorn main:app --reload
```

## API

### `POST /conversations`
Create a new conversation. Returns the `id` to use in subsequent calls.

```bash
curl -X POST http://localhost:8000/conversations
# {"id": "abc-123", "created_at": "2026-04-16T..."}
```

### `POST /conversations/{id}/chat`
Send a message, get the agent's response.

```bash
curl -X POST http://localhost:8000/conversations/abc-123/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to explore table orders from schema sales"}'
```

```json
{
  "conversation_id": "abc-123",
  "response": "The `orders` table in the `sales` schema has the following columns: ..."
}
```

### `GET /conversations/{id}/history`
Get the full human/assistant message history.

```bash
curl http://localhost:8000/conversations/abc-123/history
```

### `DELETE /conversations/{id}`
Delete a conversation.

```bash
curl -X DELETE http://localhost:8000/conversations/abc-123
```

### `GET /health`
Liveness probe — returns `{"status": "ok"}`.

## Example session

```
POST /conversations          → id = "abc-123"

POST /conversations/abc-123/chat
  "I want help to explore table orders from schema sales"
  → "The orders table has columns: id (uuid), user_id (text), amount (numeric), created_at (timestamptz)..."

POST /conversations/abc-123/chat
  "Could you retrieve the last 5 results where user_id is 'user-test-h'?"
  → "Here are the 5 most recent orders for user-test-h: ..."
```

The agent automatically calls `get_schema` when it encounters an unfamiliar table, then builds the correct SQL and calls `run_query`. If the query fails, it analyses the error and retries (up to 3 times).
