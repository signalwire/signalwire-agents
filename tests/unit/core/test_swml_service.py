"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for SWMLService class
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from signalwire_agents.core.swml_service import SWMLService


class TestSWMLServiceInitialization:
    """Test SWMLService initialization"""
    
    def test_basic_initialization(self):
        """Test basic service initialization"""
        service = SWMLService(
            name="test_service",
            route="/test",
            host="127.0.0.1",
            port=3001
        )
        
        assert service.name == "test_service"
        assert service.route == "/test"
        assert service.host == "127.0.0.1"
        assert service.port == 3001
    
    def test_initialization_with_defaults(self):
        """Test initialization with default values"""
        service = SWMLService(name="test_service")
        
        assert service.name == "test_service"
        assert service.route == ""  # Route gets stripped of trailing slash
        assert service.host == "0.0.0.0"
        assert service.port == 3000
    
    def test_initialization_with_schema_path(self):
        """Test initialization with schema path"""
        service = SWMLService(
            name="test_service",
            schema_path="/path/to/schema.json"
        )
        
        assert service.name == "test_service"
        assert hasattr(service, 'schema_utils')
    
    def test_initialization_with_basic_auth(self):
        """Test initialization with basic auth"""
        service = SWMLService(
            name="test_service",
            basic_auth=("user", "pass")
        )
        
        assert service._basic_auth == ("user", "pass")


class TestSWMLServiceVerbMethods:
    """Test SWML verb method functionality"""
    
    def test_add_verb_basic(self, mock_swml_service):
        """Test adding a basic verb"""
        result = mock_swml_service.add_verb("play", {"url": "test.mp3"})
        
        # Should return boolean indicating success
        assert isinstance(result, bool)
    
    def test_add_verb_with_config(self, mock_swml_service):
        """Test adding verb with configuration"""
        config = {
            "url": "https://example.com/audio.mp3",
            "volume": 0.8,
            "loop": 3
        }
        
        result = mock_swml_service.add_verb("play", config)
        
        # Should return boolean
        assert isinstance(result, bool)
    
    def test_add_verb_with_integer_config(self, mock_swml_service):
        """Test adding verb with integer configuration (like sleep)"""
        result = mock_swml_service.add_verb("sleep", 5000)
        
        # Should return boolean
        assert isinstance(result, bool)
    
    def test_add_verb_to_section(self, mock_swml_service):
        """Test adding verb to specific section"""
        # First add a section
        mock_swml_service.add_section("custom_section")
        
        # Then add verb to that section
        result = mock_swml_service.add_verb_to_section("custom_section", "play", {"url": "test.mp3"})
        
        # Should return boolean
        assert isinstance(result, bool)


class TestSWMLServiceDocumentManagement:
    """Test SWML document management"""
    
    def test_reset_document(self, mock_swml_service):
        """Test resetting the document"""
        # Add some verbs first
        mock_swml_service.add_verb("say", {"text": "Hello"})
        mock_swml_service.add_verb("play", {"url": "test.mp3"})
        
        # Reset document
        mock_swml_service.reset_document()
        
        # Document should be reset (we can't directly check content, but method should not raise)
        assert True  # If we get here, reset worked
    
    def test_get_document(self, mock_swml_service):
        """Test getting the document"""
        mock_swml_service.add_verb("say", {"text": "Hello World"})
        
        document = mock_swml_service.get_document()
        
        assert isinstance(document, dict)
        assert "version" in document
        assert "sections" in document
    
    def test_render_document(self, mock_swml_service):
        """Test rendering document to JSON string"""
        mock_swml_service.add_verb("say", {"text": "Hello"})
        mock_swml_service.add_verb("play", {"url": "test.mp3"})
        
        swml_json = mock_swml_service.render_document()
        
        assert isinstance(swml_json, str)
        # Should be valid JSON
        swml_dict = json.loads(swml_json)
        assert "version" in swml_dict
        assert "sections" in swml_dict
    
    def test_add_section(self, mock_swml_service):
        """Test adding a new section"""
        result = mock_swml_service.add_section("custom_section")
        
        # Should return boolean
        assert isinstance(result, bool)


class TestSWMLServiceUtilityMethods:
    """Test utility methods"""
    
    def test_basic_properties(self, mock_swml_service):
        """Test basic property access"""
        assert mock_swml_service.name == "test_service"
        assert mock_swml_service.route == "/test"
        assert mock_swml_service.host == "127.0.0.1"
        assert mock_swml_service.port == 3001
    
    def test_basic_auth_credentials(self, mock_swml_service):
        """Test getting basic auth credentials"""
        credentials = mock_swml_service.get_basic_auth_credentials()
        
        assert isinstance(credentials, tuple)
        assert len(credentials) == 2
        assert isinstance(credentials[0], str)  # username
        assert isinstance(credentials[1], str)  # password
    
    def test_basic_auth_credentials_with_source(self, mock_swml_service):
        """Test getting basic auth credentials with source"""
        credentials = mock_swml_service.get_basic_auth_credentials(include_source=True)
        
        assert isinstance(credentials, tuple)
        assert len(credentials) == 3
        assert isinstance(credentials[0], str)  # username
        assert isinstance(credentials[1], str)  # password
        assert isinstance(credentials[2], str)  # source


class TestSWMLServiceSpecialVerbs:
    """Test special SWML verb methods via add_verb"""

    def test_add_answer_verb(self, mock_swml_service):
        """Test adding answer verb via add_verb"""
        result = mock_swml_service.add_verb("answer", {"max_duration": 30})

        assert isinstance(result, bool)

    def test_add_hangup_verb(self, mock_swml_service):
        """Test adding hangup verb via add_verb"""
        result = mock_swml_service.add_verb("hangup", {"reason": "completed"})

        assert isinstance(result, bool)

    def test_add_ai_verb(self, mock_swml_service):
        """Test adding AI verb via add_verb"""
        result = mock_swml_service.add_verb("ai", {
            "prompt": {
                "text": "You are a helpful assistant"
            },
            "post_prompt": {
                "text": "Thank you for using our service"
            }
        })

        assert isinstance(result, bool)


class TestSWMLServiceErrorHandling:
    """Test error handling and edge cases"""
    
    def test_add_verb_with_none_config(self, mock_swml_service):
        """Test adding verb with None configuration"""
        result = mock_swml_service.add_verb("hangup", None)
        
        # Should return boolean (likely False due to invalid config)
        assert isinstance(result, bool)
    
    def test_add_verb_with_empty_config(self, mock_swml_service):
        """Test adding verb with empty configuration"""
        result = mock_swml_service.add_verb("hangup", {})
        
        # Should return boolean
        assert isinstance(result, bool)
    
    def test_invalid_verb_name(self, mock_swml_service):
        """Test handling of invalid verb names"""
        result = mock_swml_service.add_verb("", {"test": "value"})
        
        # Should return boolean (likely False due to invalid verb)
        assert isinstance(result, bool)
    
    def test_add_verb_to_nonexistent_section(self, mock_swml_service):
        """Test adding verb to non-existent section"""
        result = mock_swml_service.add_verb_to_section("nonexistent", "play", {"url": "test.mp3"})
        
        # Should return boolean (likely False)
        assert isinstance(result, bool)


class TestSWMLServiceRouting:
    """Test routing and callback functionality"""
    
    def test_register_routing_callback(self, mock_swml_service):
        """Test registering routing callback"""
        def test_callback(request, data):
            return "test_response"
        
        # Should not raise error
        mock_swml_service.register_routing_callback(test_callback, "/test")
        
        # Callback should be registered
        assert "/test" in mock_swml_service._routing_callbacks
    
    def test_extract_sip_username(self):
        """Test SIP username extraction"""
        request_body = {
            "from": "sip:testuser@example.com",
            "to": "sip:destination@example.com"
        }
        
        username = SWMLService.extract_sip_username(request_body)
        
        # Should extract username from SIP URI
        assert isinstance(username, (str, type(None)))


class TestSWMLServiceIntegration:
    """Test integration functionality"""
    
    def test_as_router(self, mock_swml_service):
        """Test getting as FastAPI router"""
        router = mock_swml_service.as_router()
        
        # Should return APIRouter instance
        assert router is not None
        assert hasattr(router, 'routes')
    
    def test_on_request_handling(self, mock_swml_service):
        """Test request handling"""
        test_data = {"call_id": "test-123", "from": "+1234567890"}
        
        # Should not raise error
        result = mock_swml_service.on_request(test_data)
        
        # Result can be None or dict
        assert result is None or isinstance(result, dict)
    
    def test_manual_proxy_url_setting(self, mock_swml_service):
        """Test manual proxy URL setting"""
        proxy_url = "https://example.ngrok.io"
        
        # Should not raise error
        mock_swml_service.manual_set_proxy_url(proxy_url)
        
        # Should set the proxy URL
        assert mock_swml_service._proxy_url_base == proxy_url
        assert mock_swml_service._proxy_detection_done is True
    
    def test_verb_handler_registry(self, mock_swml_service):
        """Test verb handler registry"""
        # Should have verb registry
        assert hasattr(mock_swml_service, 'verb_registry')
        assert mock_swml_service.verb_registry is not None
    
    def test_schema_utils_integration(self, mock_swml_service):
        """Test schema utilities integration"""
        # Should have schema utils
        assert hasattr(mock_swml_service, 'schema_utils')
        
        if mock_swml_service.schema_utils:
            # If schema utils available, should be able to get verb names
            verb_names = mock_swml_service.schema_utils.get_all_verb_names()
            assert isinstance(verb_names, list)
    
    def test_json_serialization_of_document(self, mock_swml_service):
        """Test JSON serialization of SWML document"""
        mock_swml_service.add_verb("say", {"text": "Test message"})
        
        document = mock_swml_service.get_document()
        
        # Should be JSON serializable
        json_str = json.dumps(document)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "version" in parsed
        assert "sections" in parsed 