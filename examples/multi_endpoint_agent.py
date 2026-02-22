#!/usr/bin/env python3
"""
Multi-Endpoint Agent Example

This agent demonstrates serving multiple endpoints:
- /swml - Voice AI SWML endpoint
- /swaig - SWAIG webhook callbacks (automatically at /swml/swaig)
- / - Web UI with hello world
- /api - JSON API endpoint
- /static/file.txt - Static file serving
"""

import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request, Response, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class MultiEndpointAgent(AgentBase):
    """Agent that serves both SWML for voice and web UI endpoints"""
    
    def __init__(self):
        # Initialize with route=/swml for the voice AI endpoint
        super().__init__(
            name="multi-endpoint",
            route="/swml",  # This sets the SWML endpoint at /swml
            host="0.0.0.0",
            port=8080
        )
        
        # Configure the voice AI agent
        self.prompt_add_section("Role", "You are a helpful voice assistant.")
        self.prompt_add_section("Instructions", bullets=[
            "Greet callers warmly",
            "Be concise in your responses",
            "Use the available functions when appropriate"
        ])
        
        # Add a simple greeting skill
        self.add_language("English", "en-US", "inworld.Mark")
        
        # Set up static files directory
        self.static_dir = Path(__file__).parent / "static_files"
        self.static_dir.mkdir(exist_ok=True)
        
        # Create a simple text file to serve
        file_txt = self.static_dir / "file.txt"
        file_txt.write_text("Hello World from static file!")
    
    @AgentBase.tool(
        name="get_time",
        description="Get the current time",
        parameters={}
    )
    def get_time(self, args, raw_data):
        """Simple SWAIG function for voice interactions"""
        now = datetime.now().strftime("%I:%M %p")
        return SwaigFunctionResult(f"The current time is {now}")
    
    def _register_routes(self, router: APIRouter):
        """
        Override route registration to add custom endpoints
        
        Note: The parent class already registers /swml routes,
        we're adding additional endpoints here
        """
        # First, register the parent SWML routes
        super()._register_routes(router)
        
        # Now add our custom UI and API endpoints
        # These will be available at the root level when we customize the app
    
    def get_app(self):
        """
        Override get_app to add custom root-level routes
        """
        if self._app is None:
            # Create the FastAPI app
            app = FastAPI(
                title="Multi-Endpoint Agent",
                description="Agent with SWML, UI, and API endpoints"
            )
            
            # Add CORS middleware
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Register health check endpoints
            @app.get("/health")
            async def health_check():
                return {"status": "healthy", "agent": self.get_name()}
            
            # Add root UI endpoint
            @app.get("/", response_class=HTMLResponse)
            async def root_ui():
                """Serve a simple HTML UI"""
                html_content = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Multi-Endpoint Agent</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            max-width: 800px;
                            margin: 0 auto;
                            padding: 20px;
                            background-color: #f5f5f5;
                        }
                        h1 {
                            color: #333;
                            text-align: center;
                        }
                        .endpoints {
                            background: white;
                            padding: 20px;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }
                        .endpoint {
                            margin: 10px 0;
                            padding: 10px;
                            background: #f9f9f9;
                            border-radius: 4px;
                        }
                        .endpoint code {
                            background: #e0e0e0;
                            padding: 2px 6px;
                            border-radius: 3px;
                        }
                        .hello-world {
                            text-align: center;
                            font-size: 24px;
                            color: #4CAF50;
                            margin: 30px 0;
                        }
                    </style>
                </head>
                <body>
                    <h1>Multi-Endpoint Agent</h1>
                    <div class="hello-world">Hello World!</div>
                    <div class="endpoints">
                        <h2>Available Endpoints:</h2>
                        <div class="endpoint">
                            <strong>Voice AI (SWML):</strong> <code>POST /swml</code>
                            <br>SignalWire voice AI endpoint
                        </div>
                        <div class="endpoint">
                            <strong>SWAIG Webhooks:</strong> <code>POST /swml/swaig</code>
                            <br>Function callbacks for voice AI
                        </div>
                        <div class="endpoint">
                            <strong>Web UI:</strong> <code>GET /</code>
                            <br>This page you're viewing now
                        </div>
                        <div class="endpoint">
                            <strong>API:</strong> <code>GET /api</code>
                            <br>JSON API endpoint
                        </div>
                        <div class="endpoint">
                            <strong>Static Files:</strong> <code>GET /static/file.txt</code>
                            <br>Serves static text file
                        </div>
                        <div class="endpoint">
                            <strong>Health Check:</strong> <code>GET /health</code>
                            <br>Service health status
                        </div>
                    </div>
                </body>
                </html>
                """
                return HTMLResponse(content=html_content)
            
            # Add API endpoint
            @app.get("/api")
            async def api_endpoint():
                """Return JSON response"""
                return JSONResponse(content={
                    "status": "ok",
                    "message": "Hello from API endpoint",
                    "timestamp": datetime.now().isoformat(),
                    "agent": self.get_name(),
                    "endpoints": {
                        "swml": "/swml",
                        "swaig": "/swml/swaig",
                        "ui": "/",
                        "api": "/api",
                        "static": "/static/file.txt"
                    }
                })
            
            # Add static file serving
            @app.get("/static/{file_path:path}")
            async def serve_static_file(file_path: str):
                """Serve static files"""
                file_full_path = self.static_dir / file_path
                
                # Security: ensure path doesn't escape static directory
                try:
                    file_full_path = file_full_path.resolve()
                    if not str(file_full_path).startswith(str(self.static_dir.resolve())):
                        return PlainTextResponse("Access denied", status_code=403)
                except Exception:
                    return PlainTextResponse("Invalid path", status_code=400)
                
                # Check if file exists
                if not file_full_path.exists() or not file_full_path.is_file():
                    return PlainTextResponse("File not found", status_code=404)
                
                # Read and return file content
                try:
                    content = file_full_path.read_text()
                    return PlainTextResponse(content)
                except Exception as e:
                    return PlainTextResponse(f"Error reading file: {e}", status_code=500)
            
            # Create router for SWML endpoints
            router = self.as_router()
            
            # Mount the SWML router at /swml
            # This gives us /swml and /swml/swaig endpoints
            app.include_router(router, prefix=self.route)
            
            self._app = app
        
        return self._app
    
    def serve(self, host=None, port=None):
        """Override serve to use our custom app"""
        import uvicorn
        
        host = host or self.host or "0.0.0.0"
        port = port or self.port or 8080
        
        # Get our custom app with all endpoints
        app = self.get_app()
        
        # Get auth credentials
        username, password, _ = self.get_basic_auth_credentials(include_source=True)
        
        print(f"\nMulti-Endpoint Agent starting...")
        print(f"Server: http://{host}:{port}")
        print(f"Basic Auth: {username}:{password}")
        print("\nEndpoints:")
        print(f"  Web UI:     http://{host}:{port}/")
        print(f"  API:        http://{host}:{port}/api")
        print(f"  Static:     http://{host}:{port}/static/file.txt")
        print(f"  SWML:       http://{host}:{port}/swml")
        print(f"  SWAIG:      http://{host}:{port}/swml/swaig")
        print(f"  Health:     http://{host}:{port}/health")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            uvicorn.run(app, host=host, port=port)
        except KeyboardInterrupt:
            print("\nStopping agent...")


if __name__ == "__main__":
    agent = MultiEndpointAgent()
    agent.serve()