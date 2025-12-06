#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Session Manager for MCP Gateway

Manages lifecycle of MCP server sessions tied to SignalWire call IDs.
Handles timeouts, cleanup, and resource limits.
"""

import threading
import time
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents an active MCP session"""
    session_id: str
    service_name: str
    process: Any  # MCPClient instance
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    timeout: int = 300  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired based on timeout"""
        return datetime.now() > self.last_accessed + timedelta(seconds=self.timeout)
    
    @property
    def is_alive(self) -> bool:
        """Check if the underlying MCP client is still running"""
        return self.process and self.process.process and self.process.process.poll() is None
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()


class SessionManager:
    """Manages MCP server sessions with automatic cleanup"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.RLock()
        self.cleanup_interval = config.get('session', {}).get('cleanup_interval', 60)
        self.max_sessions_per_service = config.get('session', {}).get('max_sessions_per_service', 100)
        self.default_timeout = config.get('session', {}).get('default_timeout', 300)
        self._shutdown = threading.Event()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"SessionManager initialized with cleanup_interval={self.cleanup_interval}s")
    
    def create_session(self, session_id: str, service_name: str, process: Any,
                      timeout: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> Session:
        """Create and register a new session"""
        with self.lock:
            # Check if session already exists
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, closing old session")
                self.close_session(session_id)
            
            # Check service limits
            service_count = sum(1 for s in self.sessions.values() if s.service_name == service_name)
            if service_count >= self.max_sessions_per_service:
                raise RuntimeError(f"Max sessions limit reached for service {service_name}")
            
            # Create new session
            session = Session(
                session_id=session_id,
                service_name=service_name,
                process=process,
                timeout=timeout or self.default_timeout,
                metadata=metadata or {}
            )
            
            self.sessions[session_id] = session
            logger.info(f"Created session {session_id} for service {service_name}")
            
            return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an active session by ID"""
        with self.lock:
            session = self.sessions.get(session_id)
            
            if session:
                if not session.is_alive:
                    logger.warning(f"Session {session_id} process is dead, removing")
                    self.close_session(session_id)
                    return None
                
                if session.is_expired:
                    logger.info(f"Session {session_id} has expired, removing")
                    self.close_session(session_id)
                    return None
                
                # Update last accessed time
                session.touch()
                
            return session
    
    def close_session(self, session_id: str) -> bool:
        """Close and remove a session"""
        with self.lock:
            session = self.sessions.pop(session_id, None)
            
            if not session:
                logger.warning(f"Attempted to close non-existent session {session_id}")
                return False
            
            # Terminate the MCP client
            if session.process:
                try:
                    # Stop the MCP client
                    session.process.stop()
                except Exception as e:
                    logger.error(f"Error stopping MCP client for session {session_id}: {e}")
            
            logger.info(f"Closed session {session_id}")
            return True
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active sessions with their info"""
        with self.lock:
            result = {}
            
            for session_id, session in list(self.sessions.items()):
                # Check if still valid
                if not session.is_alive or session.is_expired:
                    self.close_session(session_id)
                    continue
                
                result[session_id] = {
                    'service_name': session.service_name,
                    'created_at': session.created_at.isoformat(),
                    'last_accessed': session.last_accessed.isoformat(),
                    'timeout': session.timeout,
                    'metadata': session.metadata,
                    'time_remaining': max(0, (session.last_accessed + timedelta(seconds=session.timeout) - datetime.now()).total_seconds())
                }
            
            return result
    
    def get_service_session_count(self, service_name: str) -> int:
        """Get number of active sessions for a service"""
        with self.lock:
            return sum(1 for s in self.sessions.values() 
                      if s.service_name == service_name and s.is_alive and not s.is_expired)
    
    def _cleanup_loop(self):
        """Background thread that cleans up expired sessions"""
        logger.info("Session cleanup thread started")
        
        while not self._shutdown.is_set():
            try:
                # Wait with timeout so we can check shutdown flag
                if self._shutdown.wait(timeout=self.cleanup_interval):
                    break
                
                with self.lock:
                    expired_sessions = []
                    
                    for session_id, session in self.sessions.items():
                        if session.is_expired or not session.is_alive:
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        logger.info(f"Cleaning up expired session {session_id}")
                        self.close_session(session_id)
                    
                    if expired_sessions:
                        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                        
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
        
        logger.info("Session cleanup thread stopped")
    
    def shutdown(self):
        """Shutdown all sessions and cleanup"""
        logger.info("Shutting down SessionManager")
        
        # Signal cleanup thread to stop
        self._shutdown.set()
        
        with self.lock:
            # Close all sessions
            session_ids = list(self.sessions.keys())
            logger.info(f"Closing {len(session_ids)} active sessions")
            for session_id in session_ids:
                self.close_session(session_id)
        
        # Wait for cleanup thread to finish (with timeout)
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2.0)
            if self.cleanup_thread.is_alive():
                logger.warning("Cleanup thread did not stop gracefully")
        
        logger.info("SessionManager shutdown complete")