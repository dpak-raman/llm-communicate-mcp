"""
MCP Server: exposes a web_search tool backed by DuckDuckGo.

This server communicates over stdio — it is spawned as a subprocess by client.py.
The MCP protocol handles all the JSON-RPC framing; we just define our tool here.
"""
import asyncio
import sys
from ddgs import DDGS
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

server = Server("web-search-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Tell the client what tools this server offers."""
    return [
        types.Tool(
            name="web_search",
            description=(
                "Search the web for up-to-date information. "
                "Use this when you need current facts, news, or anything not in your training data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a tool call requested by the client."""
    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments["query"]
    print(f"  [MCP Server] Searching DuckDuckGo for: {query!r}", file=sys.stderr, flush=True)

    results = DDGS().text(query, max_results=  5 )

    print(f"  [MCP Server] Raw DDG results ({len(results) if results else 0} hits):", file=sys.stderr, flush=True)
    if results:
        for i, r in enumerate(results, 1):
            print(f"    {i}. {r['title']}", file=sys.stderr)
            print(f"       {r['href']}", file=sys.stderr)
            print(f"       {r['body'][:120]}", file=sys.stderr, flush=True)
    else:
        print("    (empty — DDG returned nothing)", file=sys.stderr, flush=True)

    if not results:
        text = "No results found."
    else:
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n  {r['body']}")
        text = "\n\n".join(lines)

    return [types.TextContent(type="text", text=text)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
