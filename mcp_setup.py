"""
mcp_setup.py — Connect to the @modelcontextprotocol/server-filesystem MCP server.

Launches a local Node.js stdio MCP server scoped to one folder for safety.
Returns an MCPServerAdapter context-manager that yields ready-to-use tools:
  read_file, write_file, list_directory, move_file, etc.
"""

import os
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters


def get_fs_tools(folder: str) -> MCPServerAdapter:
    """Return an MCPServerAdapter (context-manager) scoped to *folder*.

    Usage::

        with get_fs_tools("./test_folder") as tools:
            # tools is a list of CrewAI-compatible tool objects
            crew = build_crew(tools)
            crew.kickoff()
    """
    abs_folder = os.path.abspath(folder)

    params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", abs_folder],
        env={**os.environ},          # inherit current env (PATH, etc.)
    )

    return MCPServerAdapter(params)
