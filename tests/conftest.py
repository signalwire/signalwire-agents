"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Pytest configuration and shared fixtures for SignalWire AI Agents tests
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List
import json
import uuid
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main classes we'll be testing
from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult
from signalwire_agents.core.swaig_function import SWAIGFunction
from signalwire_agents.core.data_map import DataMap
from signalwire_agents.core.contexts import ContextBuilder
from signalwire_agents.core.swml_service import SWMLService


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="signalwire_tests_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    env_vars = {
        "SIGNALWIRE_PROJECT_ID": "test-project-id",
        "SIGNALWIRE_TOKEN": "test-token",
        "SIGNALWIRE_SPACE": "test.signalwire.com",
        "OPENAI_API_KEY": "test-openai-key",
        "TEST_ENV_VAR": "test-value"
    }
    
    # Patch os.environ
    original_environ = os.environ.copy()
    os.environ.update(env_vars)
    
    yield env_vars
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_environ)


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing"""
    return {
        'name': 'test_agent',
        'route': '/test',
        'host': '127.0.0.1',
        'port': 3001,
        'basic_auth': ('test_user', 'test_password'),
        'use_pom': False,  # Disable POM to avoid dependency issues in tests
        'suppress_logs': True
    }


@pytest.fixture
def mock_agent(mock_env_vars):
    """Create a mock agent for testing (schema validation disabled)"""
    with pytest.MonkeyPatch().context() as m:
        # Mock uvicorn to prevent actual server startup
        m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())

        agent = AgentBase(
            name="test_agent",
            route="/test",
            host="127.0.0.1",
            port=3001,
            suppress_logs=True,
            use_pom=False,
            schema_validation=False
        )

        # Mock the session manager to avoid initialization issues
        agent._session_manager = Mock()
        agent._session_manager.create_tool_token.return_value = "test-token"
        agent._session_manager.validate_tool_token.return_value = True

        return agent


@pytest.fixture
def sample_swaig_function():
    """Sample SWAIG function for testing"""
    def test_handler(param1: str, param2: int = 42):
        return {"result": f"Processed {param1} with {param2}"}
    
    return SWAIGFunction(
        name="test_function",
        description="A test function",
        parameters={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter", "default": 42}
            },
            "required": ["param1"]
        },
        handler=test_handler
    )


@pytest.fixture
def sample_post_data():
    """Sample POST data that would come from SignalWire"""
    return {
        "function": "test_function",
        "argument": {
            "parsed": [{"param1": "test_value", "param2": 100}]
        },
        "call_id": str(uuid.uuid4()),
        "meta_data": {
            "token": "test-token-123"
        },
        "global_data": {
            "user_id": "test-user-456"
        },
        "vars": {
            "userVariables": {
                "custom_var": "custom_value"
            }
        },
        "call": {
            "call_id": str(uuid.uuid4()),
            "state": "created",
            "direction": "inbound",
            "type": "webrtc",
            "from": "+15551234567",
            "to": "+15559876543"
        }
    }


@pytest.fixture
def mock_fastapi_request():
    """Create a mock FastAPI request for testing"""
    request = Mock()
    request.method = "POST"
    request.url = Mock()
    request.url.path = "/test"
    request.headers = {"content-type": "application/json"}
    request.json = Mock(return_value={"test": "data"})
    request.body = Mock(return_value=b'{"test": "data"}')
    return request


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    session_manager = Mock()
    session_manager.create_tool_token.return_value = "test-token-123"
    session_manager.validate_tool_token.return_value = True
    return session_manager


@pytest.fixture
def sample_swml_document():
    """Sample SWML document structure"""
    return {
        "version": "1.0.0",
        "sections": {
            "main": [
                {
                    "answer": {}
                },
                {
                    "ai": {
                        "prompt": "You are a helpful AI assistant.",
                        "SWAIG": {
                            "functions": []
                        }
                    }
                }
            ]
        }
    }


@pytest.fixture
def mock_skill():
    """Mock skill for testing skill manager"""
    from signalwire_agents.core.skill_base import SkillBase
    
    class MockSkill(SkillBase):
        SKILL_NAME = "mock_skill"
        SKILL_DESCRIPTION = "A mock skill for testing"
        SKILL_VERSION = "1.0.0"
        
        def setup(self):
            return True
            
        def register_tools(self):
            self.agent.define_tool(
                name="mock_tool",
                description="A mock tool",
                parameters={"type": "object", "properties": {}},
                handler=lambda: {"result": "mock"}
            )
    
    return MockSkill


@pytest.fixture
def temp_state_file(test_data_dir):
    """Temporary state file for testing state management"""
    state_file = test_data_dir / "test_state.json"
    yield state_file
    if state_file.exists():
        state_file.unlink()


@pytest.fixture
def sample_contexts():
    """Sample contexts for testing context system"""
    return [
        {
            "name": "greeting",
            "description": "Handle greetings",
            "steps": [
                {
                    "name": "detect_greeting",
                    "condition": "contains greeting words",
                    "action": "respond with greeting"
                }
            ]
        }
    ]


@pytest.fixture
def mock_swml_service():
    """Create a mock SWML service for testing (schema validation disabled)"""
    service = SWMLService(
        name="test_service",
        route="/test",
        host="127.0.0.1",
        port=3001,
        schema_validation=False
    )

    return service


@pytest.fixture
def mock_swaig_function():
    """Create a mock SWAIG function for testing"""
    return {
        "function": "test_function",
        "purpose": "Test function for unit tests",
        "argument": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Test parameter"},
                "param2": {"type": "integer", "description": "Test number"}
            },
            "required": ["param1"]
        }
    }


@pytest.fixture
def mock_post_data():
    """Create mock POST data for webhook testing"""
    return {
        "call_id": "test-call-123",
        "project_id": "test-project",
        "space_url": "test.signalwire.com",
        "from": "+15551234567",
        "to": "+15559876543",
        "direction": "inbound",
        "timestamp": "2024-01-01T12:00:00Z",
        "vars": {
            "user_id": "test-user-123",
            "session_id": "test-session-456"
        }
    }


@pytest.fixture
def sample_swml_response():
    """Sample SWML response for testing"""
    return {
        "version": "1.0.0",
        "sections": {
            "main": [
                {
                    "ai": {
                        "prompt": "You are a helpful assistant",
                        "SWAIG": {
                            "functions": []
                        }
                    }
                }
            ]
        }
    }


# Pytest hooks for better test organization
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark tests in integration/ directory as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests that use network fixtures as network tests
        if any(fixture in item.fixturenames for fixture in ["requests_mock", "httpx_mock"]):
            item.add_marker(pytest.mark.network)


# Test utilities
class TestUtils:
    """Utility functions for tests"""
    
    @staticmethod
    def create_mock_response(status_code: int = 200, json_data: Optional[Dict] = None):
        """Create a mock HTTP response"""
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.text = json.dumps(json_data or {})
        return response
    
    @staticmethod
    def assert_swml_structure(swml_dict: Dict[str, Any]):
        """Assert that a dictionary has valid SWML structure"""
        assert "version" in swml_dict
        assert "sections" in swml_dict
        assert "main" in swml_dict["sections"]
        assert isinstance(swml_dict["sections"]["main"], list)
    
    @staticmethod
    def assert_swaig_function_structure(func_dict: Dict[str, Any]):
        """Assert that a dictionary has valid SWAIG function structure"""
        required_fields = ["function", "purpose", "argument"]
        for field in required_fields:
            assert field in func_dict, f"Missing required field: {field}"


@pytest.fixture
def test_utils():
    """Provide test utilities"""
    return TestUtils 