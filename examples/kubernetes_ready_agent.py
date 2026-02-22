#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Kubernetes-Ready Agent Example

This demonstrates an agent configured for production Kubernetes deployment with:
- Health and readiness endpoints
- Structured JSON logging
- Graceful shutdown handling
- Environment variable configuration

Usage:
    # Use default port (8080)
    python kubernetes_ready_agent.py
    
    # Use custom port via environment variable
    PORT=8081 python kubernetes_ready_agent.py
    
    # Use custom port via command line
    python kubernetes_ready_agent.py --port 8081
"""

import os
import sys
import logging
from signalwire_agents import AgentBase

class KubernetesReadyAgent(AgentBase):
    def __init__(self, port=None):
        # Allow port override via parameter, environment, or default
        if port is None:
            port = int(os.environ.get("PORT", 8080))
            
        super().__init__(
            name="k8s-agent",
            route="/",  # Root route for simplicity
            host="0.0.0.0",  # Bind to all interfaces
            port=port
        )
        
        # Setup graceful shutdown for Kubernetes
        self.setup_graceful_shutdown()

        # Configure voice
        self.add_language(name="English", code="en-US", voice="inworld.Mark")

        # Configure prompt using the public prompt_add_section API
        self.prompt_add_section(
            "Role",
            body="You are a production-ready AI agent running in Kubernetes. "
                 "You can help users with general questions and demonstrate cloud-native deployment patterns."
        )

        # Log initialization
        self.log.info("kubernetes_agent_initialized",
                     port=self.port,
                     route=self.route)

    @AgentBase.tool(
        name="health_status",
        description="Get the health status of this agent"
    )
    def health_status(self, args, raw_data):
        from signalwire_agents.core.function_result import SwaigFunctionResult
        return SwaigFunctionResult(
            f"Agent {self.get_name()} is healthy, running on route {self.route} "
            f"port {self.port} in Kubernetes."
        )

if __name__ == "__main__":
    # Simple command line argument parsing
    port = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--port" and len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                print("Error: Port must be a number")
                sys.exit(1)
        elif sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)
    
    # Environment variable configuration
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    # Configure root logger level
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Create agent
    agent = KubernetesReadyAgent(port=port)
    
    # Log startup with environment info
    agent.log.info("agent_starting_production",
                   log_level=log_level,
                   port=agent.port,
                   health_endpoints=["/health", "/ready"])
    
    print(f"READY: Kubernetes-ready agent starting on port {agent.port}")
    print(f"HEALTH: Health check: http://localhost:{agent.port}/health")
    print(f"STATUS: Readiness check: http://localhost:{agent.port}/ready")
    print(f"LOG: Log level: {log_level}")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.log.info("agent_shutdown_requested")
        print("\nSTOPPED: Agent shutdown complete") 