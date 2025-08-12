"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import mimetypes
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

try:
    from fastapi import FastAPI, HTTPException, Request, Response, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, HTMLResponse
except ImportError:
    FastAPI = None
    HTTPException = None
    Request = None
    Response = None
    Depends = None
    CORSMiddleware = None
    HTTPBasic = None
    HTTPBasicCredentials = None
    StaticFiles = None
    FileResponse = None
    HTMLResponse = None

from signalwire_agents.core.security_config import SecurityConfig
from signalwire_agents.core.config_loader import ConfigLoader
from signalwire_agents.core.logging_config import get_logger

logger = get_logger("web_service")

class WebService:
    """Static file serving service with HTTP API"""
    
    def __init__(self, 
                 port: int = 8002,
                 directories: Dict[str, str] = None,
                 basic_auth: Optional[Tuple[str, str]] = None,
                 config_file: Optional[str] = None,
                 enable_directory_browsing: bool = False,
                 allowed_extensions: Optional[list] = None,
                 blocked_extensions: Optional[list] = None,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB default
                 enable_cors: bool = True):
        """
        Initialize WebService
        
        Args:
            port: Port to bind to (default: 8002)
            directories: Dict mapping URL paths to local directories
            basic_auth: Optional tuple of (username, password)
            config_file: Optional configuration file path
            enable_directory_browsing: Allow directory listing
            allowed_extensions: List of allowed file extensions (e.g., ['.html', '.css'])
            blocked_extensions: List of blocked extensions (e.g., ['.env', '.git'])
            max_file_size: Maximum file size in bytes to serve
            enable_cors: Enable CORS support
        """
        # Load configuration first
        self._load_config(config_file)
        
        # Override with constructor params if provided
        self.port = port
        self.enable_directory_browsing = enable_directory_browsing
        self.max_file_size = max_file_size
        self.enable_cors = enable_cors
        
        if directories is not None:
            self.directories = directories
        
        # Set up file extension filters
        self.allowed_extensions = allowed_extensions
        self.blocked_extensions = blocked_extensions or [
            '.env', '.git', '.gitignore', '.key', '.pem', '.crt',
            '.pyc', '__pycache__', '.DS_Store', '.swp'
        ]
        
        # Initialize mimetypes
        mimetypes.init()
        # Add custom MIME types if needed
        mimetypes.add_type('application/javascript', '.js')
        mimetypes.add_type('text/css', '.css')
        mimetypes.add_type('application/json', '.json')
        
        # Load security configuration
        self.security = SecurityConfig(config_file=config_file, service_name="web")
        self.security.log_config("WebService")
        
        # Set up authentication
        self._basic_auth = basic_auth or self.security.get_basic_auth()
        
        if FastAPI:
            self.app = FastAPI(
                title="SignalWire Web Service",
                description="Static file serving for SignalWire Agents"
            )
            self._setup_security()
            self._setup_routes()
            self._mount_directories()
        else:
            self.app = None
            logger.warning("FastAPI not available. HTTP service will not be available.")
    
    def _load_config(self, config_file: Optional[str]):
        """Load configuration from file if available"""
        # Initialize defaults
        self.directories = {}
        self.port = 8002
        
        # Find config file
        if not config_file:
            config_file = ConfigLoader.find_config_file("web")
        
        if not config_file:
            return
            
        # Load config
        config_loader = ConfigLoader([config_file])
        if not config_loader.has_config():
            return
            
        logger.info("loading_config_from_file", file=config_file)
        
        # Get service section
        service_config = config_loader.get_section('service')
        if service_config:
            if 'port' in service_config:
                self.port = int(service_config['port'])
            
            if 'directories' in service_config and isinstance(service_config['directories'], dict):
                self.directories = service_config['directories']
            
            if 'enable_directory_browsing' in service_config:
                self.enable_directory_browsing = bool(service_config['enable_directory_browsing'])
            
            if 'max_file_size' in service_config:
                self.max_file_size = int(service_config['max_file_size'])
            
            if 'allowed_extensions' in service_config:
                self.allowed_extensions = service_config['allowed_extensions']
            
            if 'blocked_extensions' in service_config:
                self.blocked_extensions = service_config['blocked_extensions']
    
    def _setup_security(self):
        """Setup security middleware and authentication"""
        if not self.app:
            return
            
        # Add CORS middleware if enabled
        if self.enable_cors and CORSMiddleware:
            self.app.add_middleware(
                CORSMiddleware,
                **self.security.get_cors_config()
            )
        
        # Add security headers middleware
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)
            
            # Add security headers
            is_https = request.url.scheme == "https"
            headers = self.security.get_security_headers(is_https)
            for header, value in headers.items():
                response.headers[header] = value
            
            # Add cache headers for static files
            if request.url.path.startswith(tuple(self.directories.keys())):
                # Cache static files for 1 hour
                response.headers["Cache-Control"] = "public, max-age=3600"
            
            return response
        
        # Add host validation middleware
        @self.app.middleware("http")
        async def validate_host(request: Request, call_next):
            host = request.headers.get("host", "").split(":")[0]
            if host and not self.security.should_allow_host(host):
                return Response(content="Invalid host", status_code=400)
            
            return await call_next(request)
    
    def _get_current_username(self, credentials: HTTPBasicCredentials = None) -> str:
        """Validate basic auth credentials"""
        if not credentials:
            return None
            
        correct_username, correct_password = self._basic_auth
        
        # Compare credentials
        import secrets
        username_correct = secrets.compare_digest(credentials.username, correct_username)
        password_correct = secrets.compare_digest(credentials.password, correct_password)
        
        if not (username_correct and password_correct):
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        return credentials.username
    
    def _is_file_allowed(self, file_path: Path) -> bool:
        """Check if file is allowed to be served"""
        # Check file size
        if file_path.stat().st_size > self.max_file_size:
            return False
        
        # Check extension and name
        ext = file_path.suffix.lower()
        name = file_path.name
        
        # Check blocked extensions and names
        for blocked in self.blocked_extensions:
            if blocked.startswith('.'):
                # Check both as extension and as full name (for files like .env, .gitignore)
                if ext == blocked or name == blocked:
                    return False
            else:
                if name == blocked or blocked in str(file_path):
                    return False
        
        # If allowed_extensions is set, only allow those
        if self.allowed_extensions:
            return ext in self.allowed_extensions
        
        return True
    
    def _generate_directory_listing(self, directory: Path, url_path: str) -> str:
        """Generate HTML directory listing"""
        items = []
        
        # Add parent directory link if not at root
        if url_path != '/':
            items.append('<li><a href="../">../</a></li>')
        
        # List directories first
        for item in sorted(directory.iterdir()):
            if item.name.startswith('.'):
                continue  # Skip hidden files
                
            if item.is_dir():
                items.append(f'<li>📁 <a href="{item.name}/">{item.name}/</a></li>')
        
        # Then list files
        for item in sorted(directory.iterdir()):
            if item.name.startswith('.'):
                continue
                
            if item.is_file() and self._is_file_allowed(item):
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                
                items.append(f'<li>📄 <a href="{item.name}">{item.name}</a> ({size_str})</li>')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Directory listing for {url_path}</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                ul {{ list-style: none; padding: 0; }}
                li {{ padding: 5px 0; }}
                a {{ text-decoration: none; color: #0066cc; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Directory listing for {url_path}</h1>
            <ul>
                {''.join(items)}
            </ul>
        </body>
        </html>
        """
        return html
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        if not self.app:
            return
        
        # Create security dependency if HTTPBasic is available
        security = HTTPBasic() if HTTPBasic else None
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "directories": list(self.directories.keys()),
                "ssl_enabled": self.security.ssl_enabled,
                "auth_required": bool(security),
                "directory_browsing": self.enable_directory_browsing
            }
        
        @self.app.get("/")
        async def root():
            """Root endpoint showing available directories"""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>SignalWire Web Service</title>
                <style>
                    body { font-family: sans-serif; margin: 40px; }
                    h1 { color: #333; }
                    ul { list-style: none; padding: 0; }
                    li { padding: 10px 0; }
                    a { text-decoration: none; color: #0066cc; font-size: 18px; }
                    a:hover { text-decoration: underline; }
                    .path { color: #666; font-size: 14px; }
                </style>
            </head>
            <body>
                <h1>SignalWire Web Service</h1>
                <h2>Available Directories:</h2>
                <ul>
            """
            
            for route, local_path in self.directories.items():
                html += f'<li>📁 <a href="{route}">{route}</a> <span class="path">→ {local_path}</span></li>'
            
            html += """
                </ul>
            </body>
            </html>
            """
            
            if HTMLResponse:
                return HTMLResponse(content=html)
            else:
                return {"directories": list(self.directories.keys())}
    
    def _mount_directories(self):
        """Mount static file directories"""
        if not self.app or not StaticFiles:
            return
        
        # Create security dependency if HTTPBasic is available
        security = HTTPBasic() if HTTPBasic else None
        
        for route, directory in self.directories.items():
            # Ensure directory exists
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue
            
            if not dir_path.is_dir():
                logger.warning(f"Path is not a directory: {directory}")
                continue
            
            # Normalize route
            if not route.startswith('/'):
                route = '/' + route
            
            logger.info(f"Mounting directory {directory} at route {route}")
            
            # Create custom static file handler with our security checks
            @self.app.get(f"{route}/{{file_path:path}}")
            async def serve_file(
                file_path: str,
                request: Request,
                credentials: HTTPBasicCredentials = None if not security else Depends(security),
                route=route,
                directory=directory
            ):
                """Serve files with security checks"""
                if security:
                    self._get_current_username(credentials)
                
                # Build full file path
                full_path = Path(directory) / file_path
                
                # Security: Prevent path traversal
                try:
                    full_path = full_path.resolve()
                    dir_path = Path(directory).resolve()
                    if not str(full_path).startswith(str(dir_path)):
                        raise HTTPException(status_code=403, detail="Access denied")
                except Exception:
                    raise HTTPException(status_code=403, detail="Invalid path")
                
                # Check if path exists
                if not full_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                
                # Handle directory requests
                if full_path.is_dir():
                    if not self.enable_directory_browsing:
                        # Try to serve index.html if it exists
                        index_path = full_path / "index.html"
                        if index_path.exists() and self._is_file_allowed(index_path):
                            return FileResponse(index_path)
                        else:
                            raise HTTPException(status_code=403, detail="Directory browsing disabled")
                    else:
                        # Generate directory listing
                        html = self._generate_directory_listing(full_path, request.url.path)
                        if HTMLResponse:
                            return HTMLResponse(content=html)
                        else:
                            raise HTTPException(status_code=403, detail="Directory browsing not available")
                
                # Check if file is allowed
                if not self._is_file_allowed(full_path):
                    raise HTTPException(status_code=403, detail="File type not allowed")
                
                # Serve the file
                mime_type = mimetypes.guess_type(str(full_path))[0] or 'application/octet-stream'
                
                if FileResponse:
                    return FileResponse(
                        full_path,
                        media_type=mime_type,
                        headers={
                            "Cache-Control": "public, max-age=3600",
                            "X-Content-Type-Options": "nosniff"
                        }
                    )
                else:
                    # Fallback if FileResponse not available
                    with open(full_path, 'rb') as f:
                        content = f.read()
                    return Response(
                        content=content,
                        media_type=mime_type,
                        headers={
                            "Cache-Control": "public, max-age=3600",
                            "X-Content-Type-Options": "nosniff"
                        }
                    )
    
    def add_directory(self, route: str, directory: str) -> None:
        """
        Add a new directory to serve
        
        Args:
            route: URL path to mount at (e.g., "/docs")
            directory: Local directory path to serve
        """
        # Normalize route
        if not route.startswith('/'):
            route = '/' + route
        
        # Verify directory exists
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        self.directories[route] = directory
        
        # If app is already running, mount the new directory
        if self.app:
            self._mount_directories()
    
    def remove_directory(self, route: str) -> None:
        """
        Remove a directory from being served
        
        Args:
            route: URL path to remove
        """
        # Normalize route
        if not route.startswith('/'):
            route = '/' + route
        
        if route in self.directories:
            del self.directories[route]
    
    def start(self, host: str = "0.0.0.0", port: Optional[int] = None,
              ssl_cert: Optional[str] = None, ssl_key: Optional[str] = None):
        """
        Start the service with optional HTTPS support
        
        Args:
            host: Host to bind to (default: "0.0.0.0")
            port: Port to bind to (default: self.port)
            ssl_cert: Path to SSL certificate file (overrides environment)
            ssl_key: Path to SSL key file (overrides environment)
        """
        if not self.app:
            raise RuntimeError("FastAPI not available. Cannot start HTTP service.")
        
        port = port or self.port
        
        # Get SSL configuration
        ssl_kwargs = {}
        if ssl_cert and ssl_key:
            # Use provided SSL files
            ssl_kwargs = {
                'ssl_certfile': ssl_cert,
                'ssl_keyfile': ssl_key
            }
        else:
            # Use security config SSL settings
            ssl_kwargs = self.security.get_ssl_context_kwargs()
        
        # Build startup URL
        scheme = "https" if ssl_kwargs else "http"
        startup_url = f"{scheme}://{host}:{port}"
        
        # Get auth credentials
        username, password = self._basic_auth
        
        # Log startup information
        logger.info(
            "starting_web_service",
            url=startup_url,
            ssl_enabled=bool(ssl_kwargs),
            directories=list(self.directories.keys()),
            username=username
        )
        
        # Print user-friendly startup message
        print(f"\nSignalWire Web Service starting...")
        print(f"URL: {startup_url}")
        print(f"Directories: {', '.join(self.directories.keys()) if self.directories else 'None'}")
        print(f"Basic Auth: {username}:{password}")
        print(f"Directory Browsing: {'Enabled' if self.enable_directory_browsing else 'Disabled'}")
        if ssl_kwargs:
            print(f"SSL: Enabled")
        print("")
        
        try:
            import uvicorn
            uvicorn.run(
                self.app, 
                host=host, 
                port=port,
                **ssl_kwargs
            )
        except ImportError:
            raise RuntimeError("uvicorn not available. Cannot start HTTP service.")
    
    def stop(self):
        """Stop the service (placeholder for cleanup)"""
        pass