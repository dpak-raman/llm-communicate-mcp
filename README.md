# llm-communicate-mcp

A local AI assistant powered by [Ollama](https://ollama.com) and [MCP](https://modelcontextprotocol.io), with a Telegram bot front-end.

**How it works:**

```
Telegram user
     │
     ▼
  bot.py          ← Telegram bot (python-telegram-bot)
     │
     ▼
  client.py       ← MCP client + Ollama orchestration
     │  \
     │   └──► Ollama (llama3.1:8b) ← local LLM
     │
     ▼
  server.py       ← MCP server exposing web_search tool (DuckDuckGo)
```

1. A user sends a message to the Telegram bot.
2. `bot.py` forwards the query to `client.py`.
3. `client.py` connects to the MCP server, discovers available tools, and sends the query to Ollama.
4. If Ollama decides to search the web, it calls the `web_search` tool on the MCP server (backed by DuckDuckGo).
5. The search results are fed back to Ollama, which generates a final answer.
6. The answer is sent back to the Telegram user.

You can also use the client directly from the terminal without the bot.

---

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.com) running locally with `llama3.1:8b` pulled
- A Telegram bot token from [@BotFather](https://t.me/botfather)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd llm-communicate-mcp
uv sync
```

### 2. Pull the Ollama model

```bash
ollama pull llama3.1:8b
```

### 3. Configure environment variables

Copy the example and fill in your token:

```bash
cp .env .env.local  # optional, .env is gitignored
```

Edit `.env`:

```
TELEGRAM_BOT_TOKEN=your_token_here
OLLAMA_MODEL=llama3.1:8b
```

- `TELEGRAM_BOT_TOKEN` — get one by messaging [@BotFather](https://t.me/botfather) on Telegram.
- `OLLAMA_MODEL` — any model available in your local Ollama instance (default: `llama3.1:8b`).

---

## Usage

### Run the Telegram bot

```bash
uv run python src/bot.py
```

Send any message to your bot on Telegram and it will reply with an AI-generated answer, searching the web when needed.

### Run from the terminal (no Telegram)

```bash
uv run python src/client.py "What is the latest Python release?"
uv run python src/client.py "What is 2 + 2?"
```

---

## Project structure

```
.
├── src/
│   ├── server.py   # MCP server — exposes web_search tool via DuckDuckGo
│   ├── client.py   # MCP client — orchestrates Ollama + tool calls
│   └── bot.py      # Telegram bot — routes messages through the client
├── .env            # Environment variables (gitignored)
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
