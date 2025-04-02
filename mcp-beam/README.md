# Beam MCP Server

A Model Context Protocol (MCP) server for deploying Beam applications via the Beam CLI.

## Installation

```bash
uv sync 
```

Ensure you have Python 3.11+ and `uv` installed. Update the `uv` path in `server.py` to match your system (run `which uv` to find it).

## Usage

Install the server for use with Claude Desktop:

```bash
mcp install server.py
```

Once installed, interact with the server through the Claude Desktop app. If you encounter UV-related errors, update your MCP configuration (Claude Desktop app > Settings > Developer > Edit config) as follows:

```json
{
  "mcpServers": {
    "beam-mcp": {
      "command": "/path/to/your/uv",
      "args": [
        "--directory",
        "/path/to/your/server",
        "run",
        "server.py"
      ]
    }
  }
}
```
