"""
MCP Server: exposes a web_search tool backed by DuckDuckGo,
and a run_powershell tool for local system queries.

This server communicates over stdio — it is spawned as a subprocess by client.py.
The MCP protocol handles all the JSON-RPC framing; we just define our tools here.
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
        ),
        types.Tool(
            name="run_powershell",
            description=(
                "Run a PowerShell command on the local Windows machine and return its output. "
                "Use this for local system queries: disk space, running processes, environment variables, "
                "installed software, file system checks, network info, or any other local system information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The PowerShell command to execute (read-only/query commands only)",
                    },
                },
                "required": ["command"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a tool call requested by the client."""
    if name == "web_search":
        return await _web_search(arguments)
    if name == "run_powershell":
        return await _run_powershell(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def _web_search(arguments: dict) -> list[types.TextContent]:
    query = arguments["query"]
    print(f"  [MCP Server] Searching DuckDuckGo for: {query!r}", file=sys.stderr, flush=True)

    results = DDGS().text(query, max_results=5)

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
        lines = [f"{i}. {r['title']}\n  {r['body']}" for i, r in enumerate(results, 1)]
        text = "\n\n".join(lines)

    return [types.TextContent(type="text", text=text)]


async def _run_powershell(arguments: dict) -> list[types.TextContent]:
    command = arguments["command"]
    print(f"  [MCP Server] Running PowerShell: {command!r}", file=sys.stderr, flush=True)

    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-NonInteractive", "-Command", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        return [types.TextContent(type="text", text="Error: PowerShell command timed out after 30 seconds.")]
    except FileNotFoundError:
        return [types.TextContent(type="text", text="Error: PowerShell not found on this system.")]

    output = stdout.decode(errors="replace").strip()
    error  = stderr.decode(errors="replace").strip()

    parts = []
    if output:
        parts.append(output)
    if error:
        parts.append(f"[stderr]\n{error}")
    text = "\n\n".join(parts) if parts else "(no output)"

    print(f"  [MCP Server] PowerShell result preview: {text[:200]}", file=sys.stderr, flush=True)
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
