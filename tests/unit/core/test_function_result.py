"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for SwaigFunctionResult class
"""

import pytest
import json
from unittest.mock import Mock, patch

from signalwire_agents.core.function_result import SwaigFunctionResult


class TestSwaigFunctionResultBasic:
    """Test basic SwaigFunctionResult functionality"""
    
    def test_basic_response_creation(self):
        """Test creating a basic response"""
        result = SwaigFunctionResult(response="Hello, world!")
        
        assert result.response == "Hello, world!"
        assert result.action == []
        assert result.post_process is False
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        assert result_dict["response"] == "Hello, world!"
    
    def test_response_with_action(self):
        """Test creating response with action"""
        result = SwaigFunctionResult(response="Processing request")
        result.add_action("transfer", "+15551234567")
        
        assert result.response == "Processing request"
        assert len(result.action) == 1
        assert result.action[0] == {"transfer": "+15551234567"}
        
        result_dict = result.to_dict()
        assert result_dict["response"] == "Processing request"
        assert result_dict["action"] == [{"transfer": "+15551234567"}]
    
    def test_empty_response(self):
        """Test creating empty response"""
        result = SwaigFunctionResult()
        
        assert result.response == ""
        result_dict = result.to_dict()
        # Empty response gets default message
        assert result_dict["response"] == "Action completed."
    
    def test_post_process_setting(self):
        """Test setting post_process flag"""
        result = SwaigFunctionResult(post_process=True)
        result.add_action("test", "value")  # Need action for post_process to appear
        
        assert result.post_process is True
        
        result_dict = result.to_dict()
        assert result_dict["post_process"] is True


class TestSwaigFunctionResultActions:
    """Test action-related methods"""
    
    def test_add_action(self):
        """Test adding a single action"""
        result = SwaigFunctionResult()
        result.add_action("play", {"url": "https://example.com/audio.mp3"})
        
        assert len(result.action) == 1
        assert result.action[0] == {"play": {"url": "https://example.com/audio.mp3"}}
    
    def test_add_multiple_actions(self):
        """Test adding multiple actions"""
        result = SwaigFunctionResult()
        actions = [
            {"play": {"url": "https://example.com/audio.mp3"}},
            {"transfer": "+15551234567"}
        ]
        result.add_actions(actions)
        
        assert len(result.action) == 2
        assert result.action == actions
    
    def test_connect_action(self):
        """Test the connect action helper"""
        result = SwaigFunctionResult()
        result.connect("+15551234567", final=True)
        
        assert len(result.action) == 1
        action = result.action[0]
        assert "SWML" in action
        assert action["transfer"] == "true"
        
        swml = action["SWML"]
        assert swml["sections"]["main"][0]["connect"]["to"] == "+15551234567"
    
    def test_connect_with_from_addr(self):
        """Test connect action with from address"""
        result = SwaigFunctionResult()
        result.connect("+15551234567", final=False, from_addr="+15559876543")
        
        action = result.action[0]
        assert action["transfer"] == "false"
        
        connect_params = action["SWML"]["sections"]["main"][0]["connect"]
        assert connect_params["to"] == "+15551234567"
        assert connect_params["from"] == "+15559876543"


class TestSwaigFunctionResultSWMLMethods:
    """Test SWML-specific methods"""
    
    def test_say_method(self):
        """Test the say method"""
        result = SwaigFunctionResult()
        result.say("Hello there")
        
        assert len(result.action) == 1
        assert result.action[0] == {"say": "Hello there"}
    
    def test_hangup_method(self):
        """Test the hangup method"""
        result = SwaigFunctionResult()
        result.hangup()
        
        assert len(result.action) == 1
        assert result.action[0] == {"hangup": True}
    
    def test_hold_method(self):
        """Test the hold method"""
        result = SwaigFunctionResult()
        result.hold(timeout=60)
        
        assert len(result.action) == 1
        assert result.action[0] == {"hold": 60}
    
    def test_stop_method(self):
        """Test the stop method"""
        result = SwaigFunctionResult()
        result.stop()
        
        assert len(result.action) == 1
        assert result.action[0] == {"stop": True}
    
    def test_wait_for_user_method(self):
        """Test the wait_for_user method"""
        result = SwaigFunctionResult()
        result.wait_for_user(enabled=True, timeout=30)
        
        assert len(result.action) == 1
        action = result.action[0]
        assert "wait_for_user" in action
        # The actual implementation uses timeout as the value when both are provided
        assert action["wait_for_user"] == 30


class TestSwaigFunctionResultChaining:
    """Test method chaining functionality"""
    
    def test_method_chaining(self):
        """Test that methods return self for chaining"""
        result = SwaigFunctionResult("Initial response")
        
        chained = (result
                  .set_response("Updated response")
                  .set_post_process(True)
                  .add_action("play", {"url": "test.mp3"}))
        
        # Should return the same instance
        assert chained is result
        assert result.response == "Updated response"
        assert result.post_process is True
        assert len(result.action) == 1
    
    def test_complex_chaining(self):
        """Test complex method chaining"""
        result = (SwaigFunctionResult("Welcome")
                 .say("Please hold")
                 .add_action("play", {"url": "music.mp3"})
                 .set_post_process(True))
        
        assert result.response == "Welcome"
        assert result.post_process is True
        assert len(result.action) == 2


class TestSwaigFunctionResultAdvanced:
    """Test advanced functionality"""
    
    def test_update_global_data(self):
        """Test updating global data"""
        result = SwaigFunctionResult()
        result.update_global_data({"user_id": "123", "session": "abc"})
        
        assert len(result.action) == 1
        action = result.action[0]
        assert "set_global_data" in action
        # add_action("set_global_data", data) produces {"set_global_data": data}
        assert action["set_global_data"]["user_id"] == "123"
        assert action["set_global_data"]["session"] == "abc"
    
    def test_execute_swml(self):
        """Test executing custom SWML"""
        swml_content = {
            "sections": {
                "main": [{"play": {"url": "test.mp3"}}]
            }
        }
        
        result = SwaigFunctionResult()
        result.execute_swml(swml_content)
        
        assert len(result.action) == 1
        action = result.action[0]
        assert "SWML" in action
        assert action["SWML"] == swml_content
    
    def test_switch_context(self):
        """Test switching context"""
        result = SwaigFunctionResult()
        result.switch_context(
            system_prompt="New system prompt",
            user_prompt="New user prompt",
            consolidate=True
        )
        
        assert len(result.action) == 1
        action = result.action[0]
        assert "context_switch" in action
        # add_action("context_switch", data) produces {"context_switch": data}
        context_data = action["context_switch"]
        assert context_data["system_prompt"] == "New system prompt"
        assert context_data["user_prompt"] == "New user prompt"
        assert context_data["consolidate"] is True


class TestSwaigFunctionResultSerialization:
    """Test serialization and deserialization"""
    
    def test_to_dict_basic(self):
        """Test basic to_dict conversion"""
        result = SwaigFunctionResult(response="Test response")
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "response" in result_dict
        assert result_dict["response"] == "Test response"
    
    def test_to_dict_with_actions(self):
        """Test to_dict with actions"""
        result = SwaigFunctionResult("Test")
        result.add_action("play", {"url": "test.mp3"})
        
        result_dict = result.to_dict()
        
        assert "action" in result_dict
        assert isinstance(result_dict["action"], list)
        assert len(result_dict["action"]) == 1
    
    def test_to_dict_with_all_fields(self):
        """Test to_dict with all possible fields"""
        result = SwaigFunctionResult(
            response="Complete response",
            post_process=True
        )
        result.add_action("transfer", "+15551234567")
        
        result_dict = result.to_dict()
        
        assert result_dict["response"] == "Complete response"
        assert result_dict["post_process"] is True
        assert "action" in result_dict
        assert len(result_dict["action"]) == 1
    
    def test_json_serialization(self):
        """Test JSON serialization"""
        result = SwaigFunctionResult("Hello JSON")
        result.say("Additional message")
        
        result_dict = result.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed["response"] == "Hello JSON"


class TestSwaigFunctionResultErrorHandling:
    """Test error handling and edge cases"""
    
    def test_none_response(self):
        """Test handling of None response"""
        result = SwaigFunctionResult(response=None)
        # Should convert to empty string
        assert result.response == ""
    
    def test_empty_actions(self):
        """Test handling when no actions are present"""
        result = SwaigFunctionResult(response="No actions")
        result_dict = result.to_dict()
        
        # Should have response but no action key when no actions
        assert "response" in result_dict
        assert "action" not in result_dict or result_dict.get("action") == []
    
    def test_invalid_action_data(self):
        """Test adding action with various data types"""
        result = SwaigFunctionResult()
        
        # Should handle different data types
        result.add_action("test_string", "string_value")
        result.add_action("test_number", 42)
        result.add_action("test_boolean", True)
        result.add_action("test_object", {"key": "value"})
        result.add_action("test_array", [1, 2, 3])
        
        assert len(result.action) == 5
        assert result.action[0]["test_string"] == "string_value"
        assert result.action[1]["test_number"] == 42
        assert result.action[2]["test_boolean"] is True
        assert result.action[3]["test_object"] == {"key": "value"}
        assert result.action[4]["test_array"] == [1, 2, 3]


class TestSwaigFunctionResultFactoryMethods:
    """Test factory-like usage patterns"""
    
    def test_success_response(self):
        """Test creating success response"""
        result = SwaigFunctionResult("Operation successful")
        
        assert result.response == "Operation successful"
        result_dict = result.to_dict()
        assert result_dict["response"] == "Operation successful"
    
    def test_error_response(self):
        """Test creating error response"""
        result = SwaigFunctionResult("Error occurred")
        
        assert result.response == "Error occurred"
    
    def test_transfer_response(self):
        """Test creating transfer response"""
        result = SwaigFunctionResult("Transferring you now")
        result.connect("+15551234567")
        
        result_dict = result.to_dict()
        assert "action" in result_dict
        assert len(result_dict["action"]) == 1
    
    def test_information_response(self):
        """Test creating informational response"""
        result = SwaigFunctionResult("Here is the information you requested")
        
        assert "information" in result.response.lower()


class TestSwaigFunctionResultIntegration:
    """Test integration with other components"""
    
    def test_agent_integration(self):
        """Test integration with agent tools"""
        # This would typically be tested in integration tests
        # but we can test the interface here
        
        def mock_tool_handler():
            return SwaigFunctionResult("Tool executed successfully")
        
        result = mock_tool_handler()
        assert isinstance(result, SwaigFunctionResult)
        assert result.response == "Tool executed successfully"
    
    def test_datamap_integration(self):
        """Test integration with DataMap responses"""
        result = SwaigFunctionResult("DataMap response")
        result_dict = result.to_dict()
        
        # Should be compatible with DataMap expected format
        assert "response" in result_dict
        assert isinstance(result_dict, dict)
    
    def test_webhook_response_format(self):
        """Test webhook response format compatibility"""
        result = SwaigFunctionResult(
            response="Webhook processed",
            post_process=False
        )
        result.add_action("continue", True)
        
        result_dict = result.to_dict()
        
        # Should have the format expected by SignalWire
        assert "response" in result_dict
        assert "action" in result_dict
