"""
Calls the AItrajectIQ MCP endpoint the way Cursor or Claude Code would: initialize,
list the tools, then call a few of them and print what an assistant would receive.

    pip install "mcp>=1.10"
    $env:AITRAJECTIQ_API_KEY = "atq_..."          (bash: export AITRAJECTIQ_API_KEY=atq_...)
    python test_app/mcp_smoke.py [--url http://localhost:8000/mcp]
"""

import argparse
import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

CALLS = [
    ("get_project_overview", {"days": 7}),
    ("list_traces", {"since_hours": 168, "limit": 3}),
    ("get_costs", {"days": 7, "group_by": "model"}),
    ("get_metrics_summary", {"days": 7}),
    ("find_problem_spans", {"kind": "errors", "limit": 3}),
]


async def call_tools(url: str, api_key: str) -> tuple:
    """Server name, tool names, and (name, arguments, is_error, text) for each of CALLS."""
    headers = {"Authorization": f"Bearer {api_key}"}
    async with streamablehttp_client(url, headers=headers) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            results = []
            for name, arguments in CALLS:
                result = await session.call_tool(name, arguments)
                text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
                results.append((name, arguments, bool(result.isError), text))
            return init.serverInfo.name, [tool.name for tool in tools.tools], results


async def run(url: str, api_key: str) -> int:
    server, tool_names, results = await call_tools(url, api_key)
    print(f"Connected to {server}")
    print("Tools:", ", ".join(tool_names))
    for name, arguments, is_error, text in results:
        print(f"\n== {name}({arguments}) {'ERROR' if is_error else 'ok'}")
        print(text[:1500])
    return 1 if any(is_error for _, _, is_error, _ in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("AITRAJECTIQ_MCP_URL", "http://localhost:8000/mcp"))
    parser.add_argument("--api-key", default=os.environ.get("AITRAJECTIQ_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        print("Set AITRAJECTIQ_API_KEY or pass --api-key", file=sys.stderr)
        return 2
    return asyncio.run(run(args.url, args.api_key))


if __name__ == "__main__":
    sys.exit(main())
