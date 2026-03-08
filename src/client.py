"""
MCP Client with two-LLM architecture:

  Tooling Agent  (OLLAMA_TOOLING_MODEL, default llama3.1:8b)
    - Receives the user query + MCP tools
    - Decides whether to call web_search and executes the tool loop
    - Returns the raw search context (if any)

  Chat Agent  (OLLAMA_CHAT_MODEL, default llama3.2:3b)
    - Receives the original query + any search context produced above
    - Synthesizes a clean, conversational final answer

Flow:
  1. Connect to MCP server, discover tools
  2. Tooling agent: query + tools → tool-call loop → collected search results
  3. Chat agent: query + search results → final answer

Run:
  uv run python src/client.py "What is the latest Python release?"
  uv run python src/client.py "What is 2 + 2?"
"""
import os
import asyncio
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(Path(__file__).parent.parent / ".env")

TOOLING_MODEL = os.getenv("OLLAMA_TOOLING_MODEL", "llama3.1:8b")
CHAT_MODEL    = os.getenv("OLLAMA_CHAT_MODEL",    "llama3.2:3b")
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11435")

ollama_client = ollama.Client(host=OLLAMA_HOST)


def mcp_tool_to_ollama(tool) -> dict:
    """Convert an MCP Tool object into Ollama's tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


# ── Agent 1: Tooling ──────────────────────────────────────────────────────────

async def tooling_agent(query: str, session, ollama_tools: list) -> list[str]:
    """
    Use the tooling LLM to decide whether web search is needed and execute it.

    Returns a list of raw search-result strings (may be empty if no tool was called).
    """
    today = __import__("datetime").date.today()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a routing and web search agent. Your job is to decide if the user's message requires a web search. "
                f"Today's date is {today}. "
                "Call the web_search tool ONLY when the query needs current, factual, time-sensitive information or for the one which you dont have data"
                "(e.g. news, prices, weather, ongoing events, specific facts). "
                "Do NOT call web_search for: greetings, casual conversation, opinions, general knowledge, "
                "math, coding questions, or anything you can answer without the internet. "
                "If no search is needed, do nothing — do not call any tool and do not respond. "
                "If a search IS needed, craft the best possible search query: extract the core intent, "
                "use specific keywords, include relevant context (e.g. product names, locations, timeframes), "
                "and avoid filler words. Prefer concise, search-engine-style queries over full sentences."
            ),
        },
        {"role": "user", "content": query},
    ]

    print(f"[Tooling Agent] Sending query to {TOOLING_MODEL}...")
    response = ollama_client.chat(model=TOOLING_MODEL, messages=messages, tools=ollama_tools)
    msg = response.message

    # Detect garbled tool-call in content (some Ollama builds emit JSON in content)
    if not msg.tool_calls and msg.content:
        try:
            parsed = json.loads(msg.content.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                print("[Tooling Agent] Garbled tool call detected — re-prompting without tools...")
                response = ollama_client.chat(model=TOOLING_MODEL, messages=messages)
                msg = response.message
        except (json.JSONDecodeError, ValueError):
            pass

    search_results: list[str] = []

    while msg.tool_calls:
        print(f"\n[Tooling Agent] Calling {len(msg.tool_calls)} tool(s):")
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

        for tc in msg.tool_calls:
            fn = tc.function
            print(f"  → {fn.name}  args: {fn.arguments}")

            result = await session.call_tool(fn.name, fn.arguments)
            tool_text = "\n".join(c.text for c in result.content if hasattr(c, "text"))

            print(f"  [Tool result preview] {tool_text[:300]}...")
            search_results.append(tool_text)

            messages.append({"role": "tool", "content": tool_text})

        # Let tooling LLM decide if it needs more searches
        response = ollama_client.chat(model=TOOLING_MODEL, messages=messages, tools=ollama_tools)
        msg = response.message

    return search_results


# ── Agent 2: Chat / Summarising ───────────────────────────────────────────────

def chat_agent(query: str, search_results: list[str]) -> str:
    """
    Use the chat LLM to produce a final conversational answer.

    If search_results are provided they are injected as context;
    otherwise the model answers from its own knowledge.
    """
    today = __import__("datetime").date.today()
    system_prompt = (
        "You are a helpful assistant for Deepak, who is from Tamil Nadu, India. "
        f"Today's date is {today}. "
        "When search results are provided, follow these steps: "
        "1. Read the question if it is answerable by you and not about current happening skip the search results and answer question directly"
        "2. Read every result and identify only those that directly address the question. "
        "3. Ignore results that are off-topic, promotional, or do not contribute useful information. "
        "4. Synthesise a clear, concise answer solely from the relevant results. "
        "If none of the results are relevant, say so honestly and answer from your own knowledge if you can. "
        "When no search results are provided, answer directly from your own knowledge. "
        "Always be concise, accurate, and conversational."
    )

    if search_results:
        context = "\n\n---\n\n".join(search_results)
        user_content = (
            f"Use the following search results to answer the question.\n\n"
            f"Search results:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        user_content = query

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    print(f"\n[Chat Agent] Synthesising answer with {CHAT_MODEL}...")
    response = ollama_client.chat(model=CHAT_MODEL, messages=messages)
    return response.message.content or ""


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run(query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            ollama_tools = [mcp_tool_to_ollama(t) for t in tools_result.tools]

            print(f"[Client] Connected to MCP server.")
            print(f"[Client] Available tools: {[t.name for t in tools_result.tools]}\n")

            # Step 1 — tooling agent gathers search results (if needed)
            search_results = await tooling_agent(query, session, ollama_tools)

            if search_results:
                print(f"\n[Client] Tooling agent collected {len(search_results)} result(s). Handing off to chat agent.")
            else:
                print("\n[Client] No tool calls made. Chat agent will answer from knowledge.")

            # Step 2 — chat agent produces the final answer
            answer = chat_agent(query, search_results)

    print(f"\n{'='*60}")
    print("Final Answer:")
    print(f"{'='*60}")
    print(answer)
    print()
    return answer


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python src/client.py \"<your question>\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    asyncio.run(run(query))


if __name__ == "__main__":
    main()
