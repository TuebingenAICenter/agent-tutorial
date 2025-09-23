from langchain_mcp_adapters.client import MultiServerMCPClient
from chat_with_X_utils.async_mcp_helper import MCPManager
import asyncio

# Use the MCP server config from the previous session
server_name = "filesystem_rag"  
mcp_config = {
    server_name: {
      "command": "uv",
      "args": [
        "run",
        "./hello-mcp/filesystem_rag/server_solution.py",
        "--directory=/tmp/filesystem_rag/"
      ],
      "transport": "stdio"
    },
    # It's possible to define multiple servers here and connect to them by name
}

mcp_client = MultiServerMCPClient(mcp_config)

manager = MCPManager(mcp_client)


async def run():
	# Start the MCP session and load the tools into Langgraph Tools
	session, rag_server_tools = await manager.start_session(server_name)

if __name__ == "__main__":
	asyncio.run(run())

