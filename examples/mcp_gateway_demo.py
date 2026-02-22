"""
MCP Gateway Demo

Demonstrates connecting a SignalWire AI agent to MCP (Model Context Protocol)
servers through the mcp_gateway skill. The gateway bridges MCP tools so the
agent can use them as SWAIG functions.

Prerequisites:
    pip install "signalwire-agents[mcp-gateway]"
    # Start a gateway server: mcp-gateway -c config.json

Environment variables (or pass directly):
    MCP_GATEWAY_URL - URL of the running MCP gateway service
    MCP_GATEWAY_AUTH_USER - Basic auth username
    MCP_GATEWAY_AUTH_PASSWORD - Basic auth password
"""

import os
from signalwire_agents import AgentBase


class MCPGatewayAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="MCP Gateway Agent",
            route="/mcp-gateway"
        )

        self.add_language("English", "en-US", "inworld.Mark")

        self.prompt_add_section(
            "Role",
            "You are a helpful assistant with access to external tools provided "
            "through MCP servers. Use the available tools to help users accomplish "
            "their tasks."
        )

        # Connect to MCP gateway - tools are discovered automatically
        self.add_skill("mcp_gateway", {
            "gateway_url": os.environ.get("MCP_GATEWAY_URL", "http://localhost:8080"),
            "auth_user": os.environ.get("MCP_GATEWAY_AUTH_USER", "admin"),
            "auth_password": os.environ.get("MCP_GATEWAY_AUTH_PASSWORD", "changeme"),
            "services": [{"name": "todo"}]  # List which MCP services to expose
        })


if __name__ == "__main__":
    agent = MCPGatewayAgent()
    agent.run()
