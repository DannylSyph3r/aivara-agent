"""
MCPToolset configuration — connects the agent to the Aivara MCP Server.

McpToolset is instantiated synchronously at module level as required by ADK
for deployment environments. Connection to the MCP server is lazy — it occurs
when tools are first needed, not at instantiation.

The Accept header is required by FastMCP's Streamable HTTP transport.
Without it the server returns 406 Not Acceptable.
"""
import logging

from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams

from config import AIVARA_MCP_API_KEY, AIVARA_MCP_URL

logger = logging.getLogger(__name__)

mcp_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=AIVARA_MCP_URL,
        headers={
            "X-Aivara-Key": AIVARA_MCP_API_KEY,
            "Accept": "application/json, text/event-stream",
        },
    )
)

logger.info("mcp_toolset_configured url=%s", AIVARA_MCP_URL)