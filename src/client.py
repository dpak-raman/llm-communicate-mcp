"""
MCP Client + Ollama orchestration.

Flow:
  1. Spawn the MCP server (server.py) as a subprocess via stdio transport
  2. Ask the server: "what tools do you have?" (MCP tool discovery)
  3. Convert MCP tool definitions → Ollama tool format
  4. Send user query + tools to Ollama (llama3.1:8b)
  5. If Ollama wants to call a tool → call it on the MCP server → get results
  6. Feed results back to Ollama → get final answer
  7. Print final answer

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

MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")


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


async def run(query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # ── Step 1: Connect to MCP server ───────────────────────────
    server_params = StdioServerParameters(
        command=sys.executable,  # same Python that's running this file
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── Step 2: Discover tools ───────────────────────────
            tools_result = await session.list_tools()
            ollama_tools = [mcp_tool_to_ollama(t) for t in tools_result.tools]

            print(f"[Client] Connected to MCP server.")
            print(f"[Client] Available tools: {[t.name for t in tools_result.tools]}\n")

            system_prompt = (
                "You are a helpful assistant for Deepak, who is from Tamil Nadu, India. "
                f"Today's date is {__import__('datetime').date.today()}. "
                "Use the search tool when the query involves: current events, news, "
                "prices, scores, weather, or anything that may have changed recently. "
                "Also use it for anything happening in the current year or recent months "
                "where your training data may be outdated. "
                "For greetings, math, timeless facts, or things you can answer with "
                "certainty — reply directly without calling any tool."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            # ── Step 3: Single LLM call with tools available (auto tool_choice) ──
            print(f"[Client] Sending query to {MODEL}...")
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=ollama_tools,
            )

            msg = response.message
            print(f"msg from ollama {msg}")

            # ── Step 4: Detect garbled tool-call in content ──────
            # Some Ollama models emit a JSON tool-call string in content
            # instead of using msg.tool_calls. Detect and re-prompt without tools.
            if not msg.tool_calls and msg.content:
                try:
                    parsed = json.loads(msg.content.strip())
                    if isinstance(parsed, dict) and "name" in parsed:
                        print(f"[Client] Model output garbled tool call — re-prompting without tools...")
                        response = ollama.chat(model=MODEL, messages=messages)
                        msg = response.message
                except (json.JSONDecodeError, ValueError):
                    pass

            # ── Step 5: Agentic tool-call loop ───────────────────
            # (loop for robustness; llama3.1 usually only needs one round)
            while msg.tool_calls:
                print(f"\n[Client] LLM wants to call {len(msg.tool_calls)} tool(s):")
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

                for tc in msg.tool_calls:
                    fn = tc.function
                    print(f"  → tool: {fn.name}  args: {fn.arguments}")

                    # ── Step 5: Call tool on MCP server ──────────
                    result = await session.call_tool(fn.name, fn.arguments)
                    tool_text = "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )

                    print(f"\n  [Client] Tool result preview (first 300 chars):")
                    print(f"  {tool_text[:300]}...")

                    # Append tool result so LLM can use it
                    messages.append({
                        "role": "tool",
                        "content": tool_text,
                    })

                # ── Step 6: Second LLM call with search context ──
                print(f"\n[Client] Sending search results back to {MODEL}...")
                response = ollama.chat(model=MODEL, messages=messages, tools=ollama_tools)
                msg = response.message

            # ── Step 7: Final answer ─────────────────────────────
            print(f"\n{'='*60}")
            print("Final Answer:")
            print(f"{'='*60}")
            print(msg.content)
            print()
            return msg.content or ""


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python src/client.py \"<your question>\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    asyncio.run(run(query))


if __name__ == "__main__":
    main()
