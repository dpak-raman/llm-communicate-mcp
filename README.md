# llm-communicate-mcp

A local AI assistant powered by [Ollama](https://ollama.com) and [MCP](https://modelcontextprotocol.io), with a Telegram bot front-end.

## How it works

```
Telegram user
     │
     ▼
  bot.py              ← Telegram bot (python-telegram-bot)
     │
     ▼
  client.py
     │
     ├──► Tooling Agent (llama3.1:8b)
     │         decides if web search is needed
     │         calls web_search tool via MCP if required
     │              │
     │              ▼
     │         server.py  ← MCP server exposing web_search (DuckDuckGo)
     │
     └──► Chat Agent (llama3.2:3b)
               synthesises the final answer from search results (if any)
               or answers directly from its own knowledge
```

The pipeline uses **two separate Ollama models**:

| Agent | Default model | Role |
|---|---|---|
| Tooling Agent | `llama3.1:8b` | Decides whether to search the web; executes tool calls |
| Chat Agent | `llama3.2:3b` | Produces the final conversational answer |

Flow:
1. A user sends a message to the Telegram bot (or the CLI).
2. The **Tooling Agent** receives the query and decides if a web search is needed.
   - If yes, it calls `web_search` on the MCP server (backed by DuckDuckGo) and collects the results.
   - If no (greetings, math, coding questions, stable facts, etc.), it skips the search.
3. The **Chat Agent** synthesises a clean answer from the search results (if any) or from its own knowledge.
4. The answer is returned to the Telegram user (or printed to the terminal).

---

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.com) running locally
- A Telegram bot token from [@BotFather](https://t.me/botfather) *(only needed for the bot)*

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd llm-communicate-mcp
uv sync
```

### 2. Pull the Ollama models

```bash
ollama pull llama3.1:8b   # tooling agent
ollama pull llama3.2:3b   # chat agent
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | *(required for bot)* | Token from [@BotFather](https://t.me/botfather) |
| `OLLAMA_HOST` | `http://localhost:11434` | URL of your Ollama instance |
| `OLLAMA_TOOLING_MODEL` | `llama3.1:8b` | Model for routing & tool calls (needs function-calling support) |
| `OLLAMA_CHAT_MODEL` | `llama3.2:3b` | Model for the final conversational answer |

---

## Usage

### Run the Telegram bot

```bash
uv run python src/bot.py
```

Send any message to your bot on Telegram and it will reply with an AI-generated answer, searching the web only when necessary.

### Run from the terminal (no Telegram required)

```bash
uv run python src/client.py "Who won the 2025 Super Bowl?"
uv run python src/client.py "What is 2 + 2?"
uv run python src/client.py "How does TCP work?"
```

---

## Project structure

```
.
├── src/
│   ├── server.py        # MCP server — exposes web_search tool via DuckDuckGo
│   ├── client.py        # MCP client — two-LLM orchestration (tooling + chat)
│   └── bot.py           # Telegram bot — routes messages through the client
├── .env                 # Your local config (gitignored)
├── .env.example         # Template — copy this to .env
├── pyproject.toml
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mcp` | Model Context Protocol client/server |
| `ollama` | Python client for Ollama |
| `ddgs` | DuckDuckGo search |
| `python-telegram-bot` | Telegram bot framework |
| `python-dotenv` | Load `.env` file |
