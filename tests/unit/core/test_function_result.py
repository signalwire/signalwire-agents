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


# ---------------------------------------------------------------------------
# New test classes appended below for ~95% coverage
# ---------------------------------------------------------------------------


class TestSwmlTransfer:
    """Test swml_transfer() method"""

    def test_swml_transfer_final(self):
        """Test swml_transfer with final=True (permanent transfer)"""
        result = SwaigFunctionResult("Transferring")
        result.swml_transfer("https://example.com/swml", "Goodbye!", final=True)

        assert len(result.action) == 1
        action = result.action[0]
        assert action["transfer"] == "true"
        swml = action["SWML"]
        assert swml["version"] == "1.0.0"
        main_section = swml["sections"]["main"]
        assert main_section[0] == {"set": {"ai_response": "Goodbye!"}}
        assert main_section[1] == {"transfer": {"dest": "https://example.com/swml"}}

    def test_swml_transfer_temporary(self):
        """Test swml_transfer with final=False (temporary transfer)"""
        result = SwaigFunctionResult("Hold on")
        result.swml_transfer("sip:support@company.com", "Welcome back!", final=False)

        action = result.action[0]
        assert action["transfer"] == "false"
        main_section = action["SWML"]["sections"]["main"]
        assert main_section[1]["transfer"]["dest"] == "sip:support@company.com"

    def test_swml_transfer_default_final(self):
        """Test swml_transfer default final=True"""
        result = SwaigFunctionResult().swml_transfer("https://dest.com", "bye")
        assert result.action[0]["transfer"] == "true"

    def test_swml_transfer_chaining(self):
        """Test swml_transfer returns self for chaining"""
        result = SwaigFunctionResult("msg")
        ret = result.swml_transfer("dest", "resp")
        assert ret is result


class TestSwmlUserEvent:
    """Test swml_user_event() method"""

    def test_swml_user_event_basic(self):
        """Test sending a user event with event data dict"""
        event_data = {"type": "cards_dealt", "player_hand": ["Ace", "King"], "score": 21}
        result = SwaigFunctionResult("Blackjack!").swml_user_event(event_data)

        assert len(result.action) == 1
        action = result.action[0]
        assert "SWML" in action
        swml = action["SWML"]
        assert swml["version"] == "1.0.0"
        user_event = swml["sections"]["main"][0]["user_event"]
        assert user_event["event"] == event_data

    def test_swml_user_event_chaining(self):
        """Test swml_user_event returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.swml_user_event({"type": "test"})
        assert ret is result


class TestSwmlChangeStep:
    """Test swml_change_step() method"""

    def test_swml_change_step(self):
        """Test changing the conversation step"""
        result = SwaigFunctionResult("New hand").swml_change_step("betting")

        assert len(result.action) == 1
        assert result.action[0] == {"change_step": "betting"}

    def test_swml_change_step_chaining(self):
        """Test swml_change_step returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.swml_change_step("step1")
        assert ret is result


class TestSwmlChangeContext:
    """Test swml_change_context() method"""

    def test_swml_change_context(self):
        """Test changing the conversation context"""
        result = SwaigFunctionResult("Switching").swml_change_context("technical_support")

        assert len(result.action) == 1
        assert result.action[0] == {"change_context": "technical_support"}

    def test_swml_change_context_chaining(self):
        """Test swml_change_context returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.swml_change_context("ctx")
        assert ret is result


class TestExecuteSwml:
    """Test execute_swml() method"""

    def test_execute_swml_string_valid_json(self):
        """Test execute_swml with a valid JSON string input"""
        swml_json = json.dumps({"version": "1.0.0", "sections": {"main": []}})
        result = SwaigFunctionResult().execute_swml(swml_json)

        action = result.action[0]
        assert "SWML" in action
        assert action["SWML"]["version"] == "1.0.0"

    def test_execute_swml_string_invalid_json(self):
        """Test execute_swml with an invalid JSON string (falls back to raw_swml)"""
        result = SwaigFunctionResult().execute_swml("not valid json {{{")

        action = result.action[0]
        assert "SWML" in action
        assert action["SWML"]["raw_swml"] == "not valid json {{{"

    def test_execute_swml_dict_input(self):
        """Test execute_swml with a dict input"""
        swml_dict = {"version": "1.0.0", "sections": {"main": [{"play": "test.mp3"}]}}
        result = SwaigFunctionResult().execute_swml(swml_dict)

        action = result.action[0]
        assert action["SWML"] == swml_dict

    def test_execute_swml_dict_does_not_mutate_original(self):
        """Test execute_swml makes a copy of the dict to avoid mutating caller data"""
        original = {"version": "1.0.0", "sections": {"main": []}}
        result = SwaigFunctionResult().execute_swml(original, transfer=True)

        # The original dict should NOT have 'transfer' key added
        assert "transfer" not in original
        # But the action's SWML should have it
        assert result.action[0]["SWML"]["transfer"] == "true"

    def test_execute_swml_sdk_object_with_to_dict(self):
        """Test execute_swml with an SDK object that has to_dict()"""

        class MockSwmlObject:
            def to_dict(self):
                return {"version": "1.0.0", "sections": {"main": [{"ai": {}}]}}

        result = SwaigFunctionResult().execute_swml(MockSwmlObject())
        action = result.action[0]
        assert action["SWML"]["version"] == "1.0.0"
        assert action["SWML"]["sections"]["main"][0] == {"ai": {}}

    def test_execute_swml_invalid_type_raises_type_error(self):
        """Test execute_swml with invalid type raises TypeError"""
        with pytest.raises(TypeError, match="swml_content must be string, dict, or SWML object"):
            SwaigFunctionResult().execute_swml(12345)

    def test_execute_swml_invalid_type_list(self):
        """Test execute_swml with list raises TypeError"""
        with pytest.raises(TypeError):
            SwaigFunctionResult().execute_swml([1, 2, 3])

    def test_execute_swml_with_transfer_true(self):
        """Test execute_swml with transfer=True adds transfer key"""
        swml_dict = {"version": "1.0.0", "sections": {"main": []}}
        result = SwaigFunctionResult().execute_swml(swml_dict, transfer=True)

        action = result.action[0]
        assert action["SWML"]["transfer"] == "true"

    def test_execute_swml_with_transfer_false(self):
        """Test execute_swml with transfer=False does not add transfer key"""
        swml_dict = {"version": "1.0.0", "sections": {"main": []}}
        result = SwaigFunctionResult().execute_swml(swml_dict, transfer=False)

        action = result.action[0]
        assert "transfer" not in action["SWML"]

    def test_execute_swml_chaining(self):
        """Test execute_swml returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.execute_swml({"version": "1.0.0"})
        assert ret is result


class TestHold:
    """Test hold() method with timeout clamping"""

    def test_hold_default_timeout(self):
        """Test hold with default timeout of 300"""
        result = SwaigFunctionResult().hold()
        assert result.action[0] == {"hold": 300}

    def test_hold_custom_timeout(self):
        """Test hold with custom timeout"""
        result = SwaigFunctionResult().hold(timeout=120)
        assert result.action[0] == {"hold": 120}

    def test_hold_negative_timeout_clamped_to_zero(self):
        """Test hold with negative timeout is clamped to 0"""
        result = SwaigFunctionResult().hold(timeout=-50)
        assert result.action[0] == {"hold": 0}

    def test_hold_timeout_above_900_clamped(self):
        """Test hold with timeout above 900 is clamped to 900"""
        result = SwaigFunctionResult().hold(timeout=1500)
        assert result.action[0] == {"hold": 900}

    def test_hold_timeout_exactly_900(self):
        """Test hold with timeout exactly 900"""
        result = SwaigFunctionResult().hold(timeout=900)
        assert result.action[0] == {"hold": 900}

    def test_hold_timeout_exactly_zero(self):
        """Test hold with timeout exactly 0"""
        result = SwaigFunctionResult().hold(timeout=0)
        assert result.action[0] == {"hold": 0}

    def test_hold_chaining(self):
        """Test hold returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.hold()
        assert ret is result


class TestWaitForUser:
    """Test wait_for_user() method"""

    def test_wait_for_user_answer_first(self):
        """Test wait_for_user with answer_first mode"""
        result = SwaigFunctionResult().wait_for_user(answer_first=True)
        assert result.action[0] == {"wait_for_user": "answer_first"}

    def test_wait_for_user_enabled_only(self):
        """Test wait_for_user with enabled=True only"""
        result = SwaigFunctionResult().wait_for_user(enabled=True)
        assert result.action[0] == {"wait_for_user": True}

    def test_wait_for_user_enabled_false(self):
        """Test wait_for_user with enabled=False"""
        result = SwaigFunctionResult().wait_for_user(enabled=False)
        assert result.action[0] == {"wait_for_user": False}

    def test_wait_for_user_timeout_only(self):
        """Test wait_for_user with timeout only"""
        result = SwaigFunctionResult().wait_for_user(timeout=60)
        assert result.action[0] == {"wait_for_user": 60}

    def test_wait_for_user_no_args(self):
        """Test wait_for_user with no arguments defaults to True"""
        result = SwaigFunctionResult().wait_for_user()
        assert result.action[0] == {"wait_for_user": True}

    def test_wait_for_user_answer_first_takes_priority(self):
        """Test that answer_first takes priority over other args"""
        result = SwaigFunctionResult().wait_for_user(enabled=True, timeout=30, answer_first=True)
        assert result.action[0] == {"wait_for_user": "answer_first"}

    def test_wait_for_user_timeout_takes_priority_over_enabled(self):
        """Test that timeout takes priority over enabled when both set"""
        result = SwaigFunctionResult().wait_for_user(enabled=True, timeout=45)
        assert result.action[0] == {"wait_for_user": 45}

    def test_wait_for_user_chaining(self):
        """Test wait_for_user returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.wait_for_user()
        assert ret is result


class TestPlayBackgroundFile:
    """Test play_background_file() method"""

    def test_play_background_file_without_wait(self):
        """Test play_background_file with wait=False (default)"""
        result = SwaigFunctionResult().play_background_file("music.mp3")
        assert result.action[0] == {"playback_bg": "music.mp3"}

    def test_play_background_file_with_wait_true(self):
        """Test play_background_file with wait=True returns dict form"""
        result = SwaigFunctionResult().play_background_file("music.mp3", wait=True)
        assert result.action[0] == {"playback_bg": {"file": "music.mp3", "wait": True}}

    def test_play_background_file_with_wait_false_explicit(self):
        """Test play_background_file with explicit wait=False"""
        result = SwaigFunctionResult().play_background_file("video.mp4", wait=False)
        assert result.action[0] == {"playback_bg": "video.mp4"}

    def test_play_background_file_chaining(self):
        """Test play_background_file returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.play_background_file("test.mp3")
        assert ret is result


class TestStopBackgroundFile:
    """Test stop_background_file() method"""

    def test_stop_background_file(self):
        """Test stop_background_file adds correct action"""
        result = SwaigFunctionResult().stop_background_file()
        assert result.action[0] == {"stop_playback_bg": True}

    def test_stop_background_file_chaining(self):
        """Test stop_background_file returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.stop_background_file()
        assert ret is result


class TestRemoveGlobalData:
    """Test remove_global_data() method"""

    def test_remove_global_data_single_string(self):
        """Test remove_global_data with a single string key"""
        result = SwaigFunctionResult().remove_global_data("user_id")
        assert result.action[0] == {"unset_global_data": "user_id"}

    def test_remove_global_data_list_of_keys(self):
        """Test remove_global_data with a list of keys"""
        result = SwaigFunctionResult().remove_global_data(["user_id", "session", "token"])
        assert result.action[0] == {"unset_global_data": ["user_id", "session", "token"]}

    def test_remove_global_data_chaining(self):
        """Test remove_global_data returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.remove_global_data("key")
        assert ret is result


class TestSetMetadata:
    """Test set_metadata() method"""

    def test_set_metadata_dict(self):
        """Test set_metadata with a dict"""
        data = {"key1": "value1", "key2": 42}
        result = SwaigFunctionResult().set_metadata(data)
        assert result.action[0] == {"set_meta_data": data}

    def test_set_metadata_chaining(self):
        """Test set_metadata returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.set_metadata({"k": "v"})
        assert ret is result


class TestRemoveMetadata:
    """Test remove_metadata() method"""

    def test_remove_metadata_single_string(self):
        """Test remove_metadata with a single string key"""
        result = SwaigFunctionResult().remove_metadata("key1")
        assert result.action[0] == {"unset_meta_data": "key1"}

    def test_remove_metadata_list_of_keys(self):
        """Test remove_metadata with a list of keys"""
        result = SwaigFunctionResult().remove_metadata(["key1", "key2"])
        assert result.action[0] == {"unset_meta_data": ["key1", "key2"]}

    def test_remove_metadata_chaining(self):
        """Test remove_metadata returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.remove_metadata("k")
        assert ret is result


class TestPay:
    """Test pay() method"""

    def test_pay_default_params(self):
        """Test pay with only the required payment_connector_url"""
        result = SwaigFunctionResult().pay("https://pay.example.com/connector")

        assert len(result.action) == 1
        swml = result.action[0]["SWML"]
        main_section = swml["sections"]["main"]
        # First item is set ai_response
        assert "set" in main_section[0]
        assert "ai_response" in main_section[0]["set"]
        # Second item is pay
        pay_params = main_section[1]["pay"]
        assert pay_params["payment_connector_url"] == "https://pay.example.com/connector"
        assert pay_params["input"] == "dtmf"
        assert pay_params["payment_method"] == "credit-card"
        assert pay_params["timeout"] == "5"
        assert pay_params["max_attempts"] == "1"
        assert pay_params["security_code"] == "true"
        assert pay_params["postal_code"] == "true"
        assert pay_params["min_postal_code_length"] == "0"
        assert pay_params["token_type"] == "reusable"
        assert pay_params["currency"] == "usd"
        assert pay_params["language"] == "en-US"
        assert pay_params["voice"] == "woman"
        assert pay_params["valid_card_types"] == "visa mastercard amex"

    def test_pay_all_custom_params(self):
        """Test pay with all custom parameters"""
        result = SwaigFunctionResult().pay(
            payment_connector_url="https://pay.example.com",
            input_method="voice",
            status_url="https://status.example.com",
            payment_method="credit-card",
            timeout=10,
            max_attempts=3,
            security_code=False,
            postal_code="90210",
            min_postal_code_length=5,
            token_type="one-time",
            charge_amount="49.99",
            currency="eur",
            language="fr-FR",
            voice="man",
            description="Monthly subscription",
            valid_card_types="visa amex",
            ai_response="Payment processed."
        )

        pay_params = result.action[0]["SWML"]["sections"]["main"][1]["pay"]
        assert pay_params["input"] == "voice"
        assert pay_params["status_url"] == "https://status.example.com"
        assert pay_params["timeout"] == "10"
        assert pay_params["max_attempts"] == "3"
        assert pay_params["security_code"] == "false"
        assert pay_params["postal_code"] == "90210"
        assert pay_params["min_postal_code_length"] == "5"
        assert pay_params["token_type"] == "one-time"
        assert pay_params["charge_amount"] == "49.99"
        assert pay_params["currency"] == "eur"
        assert pay_params["language"] == "fr-FR"
        assert pay_params["voice"] == "man"
        assert pay_params["description"] == "Monthly subscription"
        assert pay_params["valid_card_types"] == "visa amex"

        ai_response = result.action[0]["SWML"]["sections"]["main"][0]["set"]["ai_response"]
        assert ai_response == "Payment processed."

    def test_pay_with_prompts_and_parameters(self):
        """Test pay with custom prompts and parameters"""
        prompts = [{"for": "payment-card-number", "actions": [{"type": "Say", "phrase": "Enter card"}]}]
        parameters = [{"name": "store_id", "value": "123"}]
        result = SwaigFunctionResult().pay(
            payment_connector_url="https://pay.example.com",
            prompts=prompts,
            parameters=parameters
        )

        pay_params = result.action[0]["SWML"]["sections"]["main"][1]["pay"]
        assert pay_params["prompts"] == prompts
        assert pay_params["parameters"] == parameters

    def test_pay_postal_code_boolean_false(self):
        """Test pay with postal_code as boolean False"""
        result = SwaigFunctionResult().pay(
            payment_connector_url="https://pay.example.com",
            postal_code=False
        )
        pay_params = result.action[0]["SWML"]["sections"]["main"][1]["pay"]
        assert pay_params["postal_code"] == "false"

    def test_pay_chaining(self):
        """Test pay returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.pay("https://pay.example.com")
        assert ret is result


class TestJoinConference:
    """Test join_conference() method"""

    def test_join_conference_simple_name_all_defaults(self):
        """Test join_conference with just a name (all defaults) uses simple form"""
        result = SwaigFunctionResult().join_conference("my-conference")

        swml = result.action[0]["SWML"]
        join_params = swml["sections"]["main"][0]["join_conference"]
        # Simple form: just the conference name string
        assert join_params == "my-conference"

    def test_join_conference_complex_params(self):
        """Test join_conference with non-default params uses object form"""
        result = SwaigFunctionResult().join_conference(
            name="team-meeting",
            muted=True,
            beep="onEnter",
            start_on_enter=False,
            end_on_exit=True,
            wait_url="https://example.com/hold-music",
            max_participants=50,
            record="record-from-start",
            region="us-east",
            trim="do-not-trim",
            coach="call-id-123",
            status_callback_event="start end",
            status_callback="https://example.com/callback",
            status_callback_method="GET",
            recording_status_callback="https://example.com/rec-callback",
            recording_status_callback_method="GET",
            recording_status_callback_event="in-progress",
            result={"key": "value"}
        )

        swml = result.action[0]["SWML"]
        join_params = swml["sections"]["main"][0]["join_conference"]
        assert isinstance(join_params, dict)
        assert join_params["name"] == "team-meeting"
        assert join_params["muted"] is True
        assert join_params["beep"] == "onEnter"
        assert join_params["start_on_enter"] is False
        assert join_params["end_on_exit"] is True
        assert join_params["wait_url"] == "https://example.com/hold-music"
        assert join_params["max_participants"] == 50
        assert join_params["record"] == "record-from-start"
        assert join_params["region"] == "us-east"
        assert join_params["trim"] == "do-not-trim"
        assert join_params["coach"] == "call-id-123"
        assert join_params["status_callback_event"] == "start end"
        assert join_params["status_callback"] == "https://example.com/callback"
        assert join_params["status_callback_method"] == "GET"
        assert join_params["recording_status_callback"] == "https://example.com/rec-callback"
        assert join_params["recording_status_callback_method"] == "GET"
        assert join_params["recording_status_callback_event"] == "in-progress"
        assert join_params["result"] == {"key": "value"}

    def test_join_conference_invalid_beep(self):
        """Test join_conference with invalid beep raises ValueError"""
        with pytest.raises(ValueError, match="beep must be one of"):
            SwaigFunctionResult().join_conference("conf", beep="invalid")

    def test_join_conference_max_participants_too_high(self):
        """Test join_conference with max_participants > 250 raises ValueError"""
        with pytest.raises(ValueError, match="max_participants must be a positive integer <= 250"):
            SwaigFunctionResult().join_conference("conf", max_participants=300)

    def test_join_conference_max_participants_zero(self):
        """Test join_conference with max_participants=0 raises ValueError"""
        with pytest.raises(ValueError, match="max_participants must be a positive integer <= 250"):
            SwaigFunctionResult().join_conference("conf", max_participants=0)

    def test_join_conference_max_participants_negative(self):
        """Test join_conference with negative max_participants raises ValueError"""
        with pytest.raises(ValueError, match="max_participants must be a positive integer <= 250"):
            SwaigFunctionResult().join_conference("conf", max_participants=-5)

    def test_join_conference_invalid_record(self):
        """Test join_conference with invalid record raises ValueError"""
        with pytest.raises(ValueError, match="record must be one of"):
            SwaigFunctionResult().join_conference("conf", record="always")

    def test_join_conference_invalid_trim(self):
        """Test join_conference with invalid trim raises ValueError"""
        with pytest.raises(ValueError, match="trim must be one of"):
            SwaigFunctionResult().join_conference("conf", trim="bad-value")

    def test_join_conference_empty_name(self):
        """Test join_conference with empty name raises ValueError"""
        with pytest.raises(ValueError, match="name cannot be empty"):
            SwaigFunctionResult().join_conference("", muted=True)

    def test_join_conference_whitespace_name(self):
        """Test join_conference with whitespace-only name raises ValueError"""
        with pytest.raises(ValueError, match="name cannot be empty"):
            SwaigFunctionResult().join_conference("   ", muted=True)

    def test_join_conference_invalid_status_callback_method(self):
        """Test join_conference with invalid status_callback_method raises ValueError"""
        with pytest.raises(ValueError, match="status_callback_method must be one of"):
            SwaigFunctionResult().join_conference("conf", status_callback_method="PUT")

    def test_join_conference_invalid_recording_status_callback_method(self):
        """Test join_conference with invalid recording_status_callback_method raises ValueError"""
        with pytest.raises(ValueError, match="recording_status_callback_method must be one of"):
            SwaigFunctionResult().join_conference("conf", recording_status_callback_method="DELETE")

    def test_join_conference_chaining(self):
        """Test join_conference returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.join_conference("conf")
        assert ret is result


class TestTap:
    """Test tap() method"""

    def test_tap_default_params(self):
        """Test tap with only required URI (all defaults)"""
        result = SwaigFunctionResult().tap("rtp://192.168.1.1:5000")

        swml = result.action[0]["SWML"]
        tap_params = swml["sections"]["main"][0]["tap"]
        assert tap_params["uri"] == "rtp://192.168.1.1:5000"
        # Default params should not be included
        assert "direction" not in tap_params
        assert "codec" not in tap_params
        assert "rtp_ptime" not in tap_params

    def test_tap_custom_params(self):
        """Test tap with all custom parameters"""
        result = SwaigFunctionResult().tap(
            uri="ws://example.com/tap",
            control_id="my-tap-1",
            direction="speak",
            codec="PCMA",
            rtp_ptime=30,
            status_url="https://example.com/status"
        )

        tap_params = result.action[0]["SWML"]["sections"]["main"][0]["tap"]
        assert tap_params["uri"] == "ws://example.com/tap"
        assert tap_params["control_id"] == "my-tap-1"
        assert tap_params["direction"] == "speak"
        assert tap_params["codec"] == "PCMA"
        assert tap_params["rtp_ptime"] == 30
        assert tap_params["status_url"] == "https://example.com/status"

    def test_tap_invalid_direction(self):
        """Test tap with invalid direction raises ValueError"""
        with pytest.raises(ValueError, match="direction must be one of"):
            SwaigFunctionResult().tap("rtp://1.2.3.4:5000", direction="invalid")

    def test_tap_invalid_codec(self):
        """Test tap with invalid codec raises ValueError"""
        with pytest.raises(ValueError, match="codec must be one of"):
            SwaigFunctionResult().tap("rtp://1.2.3.4:5000", codec="G729")

    def test_tap_invalid_rtp_ptime(self):
        """Test tap with invalid rtp_ptime raises ValueError"""
        with pytest.raises(ValueError, match="rtp_ptime must be a positive integer"):
            SwaigFunctionResult().tap("rtp://1.2.3.4:5000", rtp_ptime=0)

    def test_tap_negative_rtp_ptime(self):
        """Test tap with negative rtp_ptime raises ValueError"""
        with pytest.raises(ValueError, match="rtp_ptime must be a positive integer"):
            SwaigFunctionResult().tap("rtp://1.2.3.4:5000", rtp_ptime=-10)

    def test_tap_direction_hear(self):
        """Test tap with direction=hear"""
        result = SwaigFunctionResult().tap("rtp://1.2.3.4:5000", direction="hear")
        tap_params = result.action[0]["SWML"]["sections"]["main"][0]["tap"]
        assert tap_params["direction"] == "hear"

    def test_tap_chaining(self):
        """Test tap returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.tap("rtp://1.2.3.4:5000")
        assert ret is result


class TestStopTap:
    """Test stop_tap() method"""

    def test_stop_tap_with_control_id(self):
        """Test stop_tap with a control_id"""
        result = SwaigFunctionResult().stop_tap(control_id="my-tap-1")

        swml = result.action[0]["SWML"]
        stop_params = swml["sections"]["main"][0]["stop_tap"]
        assert stop_params["control_id"] == "my-tap-1"

    def test_stop_tap_without_control_id(self):
        """Test stop_tap without control_id (stops last tap)"""
        result = SwaigFunctionResult().stop_tap()

        swml = result.action[0]["SWML"]
        stop_params = swml["sections"]["main"][0]["stop_tap"]
        assert stop_params == {}

    def test_stop_tap_chaining(self):
        """Test stop_tap returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.stop_tap()
        assert ret is result


class TestRecordCall:
    """Test record_call() method"""

    def test_record_call_default_params(self):
        """Test record_call with all default parameters"""
        result = SwaigFunctionResult().record_call()

        swml = result.action[0]["SWML"]
        rec_params = swml["sections"]["main"][0]["record_call"]
        assert rec_params["stereo"] is False
        assert rec_params["format"] == "wav"
        assert rec_params["direction"] == "both"
        assert rec_params["beep"] is False
        assert rec_params["input_sensitivity"] == 44.0
        assert "control_id" not in rec_params

    def test_record_call_custom_params(self):
        """Test record_call with all custom parameters"""
        result = SwaigFunctionResult().record_call(
            control_id="rec-1",
            stereo=True,
            format="mp3",
            direction="speak",
            terminators="#",
            beep=True,
            input_sensitivity=50.0,
            initial_timeout=10.0,
            end_silence_timeout=5.0,
            max_length=600.0,
            status_url="https://example.com/rec-status"
        )

        rec_params = result.action[0]["SWML"]["sections"]["main"][0]["record_call"]
        assert rec_params["control_id"] == "rec-1"
        assert rec_params["stereo"] is True
        assert rec_params["format"] == "mp3"
        assert rec_params["direction"] == "speak"
        assert rec_params["terminators"] == "#"
        assert rec_params["beep"] is True
        assert rec_params["input_sensitivity"] == 50.0
        assert rec_params["initial_timeout"] == 10.0
        assert rec_params["end_silence_timeout"] == 5.0
        assert rec_params["max_length"] == 600.0
        assert rec_params["status_url"] == "https://example.com/rec-status"

    def test_record_call_invalid_format(self):
        """Test record_call with invalid format raises ValueError"""
        with pytest.raises(ValueError, match="format must be 'wav' or 'mp3'"):
            SwaigFunctionResult().record_call(format="ogg")

    def test_record_call_invalid_direction(self):
        """Test record_call with invalid direction raises ValueError"""
        with pytest.raises(ValueError, match="direction must be 'speak', 'listen', or 'both'"):
            SwaigFunctionResult().record_call(direction="left")

    def test_record_call_direction_listen(self):
        """Test record_call with direction=listen"""
        result = SwaigFunctionResult().record_call(direction="listen")
        rec_params = result.action[0]["SWML"]["sections"]["main"][0]["record_call"]
        assert rec_params["direction"] == "listen"

    def test_record_call_chaining(self):
        """Test record_call returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.record_call()
        assert ret is result


class TestStopRecordCall:
    """Test stop_record_call() method"""

    def test_stop_record_call_with_control_id(self):
        """Test stop_record_call with a control_id"""
        result = SwaigFunctionResult().stop_record_call(control_id="rec-1")

        swml = result.action[0]["SWML"]
        stop_params = swml["sections"]["main"][0]["stop_record_call"]
        assert stop_params["control_id"] == "rec-1"

    def test_stop_record_call_without_control_id(self):
        """Test stop_record_call without control_id"""
        result = SwaigFunctionResult().stop_record_call()

        swml = result.action[0]["SWML"]
        stop_params = swml["sections"]["main"][0]["stop_record_call"]
        assert stop_params == {}

    def test_stop_record_call_chaining(self):
        """Test stop_record_call returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.stop_record_call()
        assert ret is result


class TestSendSms:
    """Test send_sms() method"""

    def test_send_sms_with_body(self):
        """Test send_sms with body text"""
        result = SwaigFunctionResult().send_sms(
            to_number="+15551234567",
            from_number="+15559876543",
            body="Hello from AI"
        )

        swml = result.action[0]["SWML"]
        sms_params = swml["sections"]["main"][0]["send_sms"]
        assert sms_params["to_number"] == "+15551234567"
        assert sms_params["from_number"] == "+15559876543"
        assert sms_params["body"] == "Hello from AI"
        assert "media" not in sms_params

    def test_send_sms_with_media(self):
        """Test send_sms with media URLs"""
        result = SwaigFunctionResult().send_sms(
            to_number="+15551234567",
            from_number="+15559876543",
            media=["https://example.com/image.png"]
        )

        sms_params = result.action[0]["SWML"]["sections"]["main"][0]["send_sms"]
        assert "body" not in sms_params
        assert sms_params["media"] == ["https://example.com/image.png"]

    def test_send_sms_with_body_and_media(self):
        """Test send_sms with both body and media"""
        result = SwaigFunctionResult().send_sms(
            to_number="+15551234567",
            from_number="+15559876543",
            body="Check this out",
            media=["https://example.com/image.png"]
        )

        sms_params = result.action[0]["SWML"]["sections"]["main"][0]["send_sms"]
        assert sms_params["body"] == "Check this out"
        assert sms_params["media"] == ["https://example.com/image.png"]

    def test_send_sms_missing_both_raises_value_error(self):
        """Test send_sms with neither body nor media raises ValueError"""
        with pytest.raises(ValueError, match="Either body or media must be provided"):
            SwaigFunctionResult().send_sms(
                to_number="+15551234567",
                from_number="+15559876543"
            )

    def test_send_sms_with_tags_and_region(self):
        """Test send_sms with tags and region"""
        result = SwaigFunctionResult().send_sms(
            to_number="+15551234567",
            from_number="+15559876543",
            body="Tagged message",
            tags=["support", "urgent"],
            region="us-east"
        )

        sms_params = result.action[0]["SWML"]["sections"]["main"][0]["send_sms"]
        assert sms_params["tags"] == ["support", "urgent"]
        assert sms_params["region"] == "us-east"

    def test_send_sms_chaining(self):
        """Test send_sms returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.send_sms("+1555", "+1556", body="hi")
        assert ret is result


class TestSipRefer:
    """Test sip_refer() method"""

    def test_sip_refer_basic(self):
        """Test sip_refer basic usage"""
        result = SwaigFunctionResult().sip_refer("sip:alice@example.com")

        swml = result.action[0]["SWML"]
        refer_params = swml["sections"]["main"][0]["sip_refer"]
        assert refer_params["to_uri"] == "sip:alice@example.com"

    def test_sip_refer_chaining(self):
        """Test sip_refer returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.sip_refer("sip:bob@example.com")
        assert ret is result


class TestJoinRoom:
    """Test join_room() method"""

    def test_join_room_basic(self):
        """Test join_room basic usage"""
        result = SwaigFunctionResult().join_room("my-room")

        swml = result.action[0]["SWML"]
        join_params = swml["sections"]["main"][0]["join_room"]
        assert join_params["name"] == "my-room"

    def test_join_room_chaining(self):
        """Test join_room returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.join_room("room1")
        assert ret is result


class TestExecuteRpc:
    """Test execute_rpc() method"""

    def test_execute_rpc_method_only(self):
        """Test execute_rpc with just the method"""
        result = SwaigFunctionResult().execute_rpc(method="ping")

        swml = result.action[0]["SWML"]
        rpc_params = swml["sections"]["main"][0]["execute_rpc"]
        assert rpc_params["method"] == "ping"
        assert "call_id" not in rpc_params
        assert "node_id" not in rpc_params
        assert "params" not in rpc_params

    def test_execute_rpc_with_all_params(self):
        """Test execute_rpc with all parameters"""
        result = SwaigFunctionResult().execute_rpc(
            method="ai_message",
            params={"role": "system", "message_text": "Hello"},
            call_id="call-123",
            node_id="node-456"
        )

        rpc_params = result.action[0]["SWML"]["sections"]["main"][0]["execute_rpc"]
        assert rpc_params["method"] == "ai_message"
        assert rpc_params["call_id"] == "call-123"
        assert rpc_params["node_id"] == "node-456"
        assert rpc_params["params"] == {"role": "system", "message_text": "Hello"}

    def test_execute_rpc_with_call_id_only(self):
        """Test execute_rpc with call_id but no params"""
        result = SwaigFunctionResult().execute_rpc(method="status", call_id="call-789")

        rpc_params = result.action[0]["SWML"]["sections"]["main"][0]["execute_rpc"]
        assert rpc_params["call_id"] == "call-789"
        assert "params" not in rpc_params

    def test_execute_rpc_chaining(self):
        """Test execute_rpc returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.execute_rpc("test")
        assert ret is result


class TestRpcDial:
    """Test rpc_dial() method"""

    def test_rpc_dial_basic(self):
        """Test rpc_dial basic usage"""
        result = SwaigFunctionResult().rpc_dial(
            to_number="+15551234567",
            from_number="+15559876543",
            dest_swml="https://example.com/call-agent"
        )

        rpc_params = result.action[0]["SWML"]["sections"]["main"][0]["execute_rpc"]
        assert rpc_params["method"] == "dial"
        params = rpc_params["params"]
        assert params["dest_swml"] == "https://example.com/call-agent"
        assert params["devices"]["type"] == "phone"
        assert params["devices"]["params"]["to_number"] == "+15551234567"
        assert params["devices"]["params"]["from_number"] == "+15559876543"

    def test_rpc_dial_custom_device_type(self):
        """Test rpc_dial with custom device_type"""
        result = SwaigFunctionResult().rpc_dial(
            to_number="+15551234567",
            from_number="+15559876543",
            dest_swml="https://example.com/swml",
            device_type="sip"
        )

        params = result.action[0]["SWML"]["sections"]["main"][0]["execute_rpc"]["params"]
        assert params["devices"]["type"] == "sip"

    def test_rpc_dial_chaining(self):
        """Test rpc_dial returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.rpc_dial("+1555", "+1556", "https://dest.com")
        assert ret is result


class TestRpcAiMessage:
    """Test rpc_ai_message() method"""

    def test_rpc_ai_message_basic(self):
        """Test rpc_ai_message basic usage"""
        result = SwaigFunctionResult().rpc_ai_message(
            call_id="call-abc",
            message_text="Please take a message."
        )

        rpc_params = result.action[0]["SWML"]["sections"]["main"][0]["execute_rpc"]
        assert rpc_params["method"] == "ai_message"
        assert rpc_params["call_id"] == "call-abc"
        assert rpc_params["params"]["role"] == "system"
        assert rpc_params["params"]["message_text"] == "Please take a message."

    def test_rpc_ai_message_custom_role(self):
        """Test rpc_ai_message with custom role"""
        result = SwaigFunctionResult().rpc_ai_message(
            call_id="call-xyz",
            message_text="User said hello",
            role="user"
        )

        params = result.action[0]["SWML"]["sections"]["main"][0]["execute_rpc"]["params"]
        assert params["role"] == "user"

    def test_rpc_ai_message_chaining(self):
        """Test rpc_ai_message returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.rpc_ai_message("call-1", "msg")
        assert ret is result


class TestRpcAiUnhold:
    """Test rpc_ai_unhold() method"""

    def test_rpc_ai_unhold_basic(self):
        """Test rpc_ai_unhold basic usage"""
        result = SwaigFunctionResult().rpc_ai_unhold(call_id="call-abc")

        rpc_params = result.action[0]["SWML"]["sections"]["main"][0]["execute_rpc"]
        assert rpc_params["method"] == "ai_unhold"
        assert rpc_params["call_id"] == "call-abc"
        # empty dict params={} is falsy, so execute_rpc does not include "params" key
        assert "params" not in rpc_params

    def test_rpc_ai_unhold_chaining(self):
        """Test rpc_ai_unhold returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.rpc_ai_unhold("call-1")
        assert ret is result


class TestCreatePaymentPrompt:
    """Test create_payment_prompt() static method"""

    def test_create_payment_prompt_basic(self):
        """Test create_payment_prompt without card_type or error_type"""
        actions = [{"type": "Say", "phrase": "Enter your card number"}]
        prompt = SwaigFunctionResult.create_payment_prompt("payment-card-number", actions)

        assert prompt["for"] == "payment-card-number"
        assert prompt["actions"] == actions
        assert "card_type" not in prompt
        assert "error_type" not in prompt

    def test_create_payment_prompt_with_card_type(self):
        """Test create_payment_prompt with card_type"""
        actions = [{"type": "Say", "phrase": "Enter card"}]
        prompt = SwaigFunctionResult.create_payment_prompt(
            "payment-card-number", actions, card_type="visa mastercard"
        )

        assert prompt["card_type"] == "visa mastercard"

    def test_create_payment_prompt_with_error_type(self):
        """Test create_payment_prompt with error_type"""
        actions = [{"type": "Say", "phrase": "Invalid card"}]
        prompt = SwaigFunctionResult.create_payment_prompt(
            "payment-card-number", actions, error_type="invalid-card-number"
        )

        assert prompt["error_type"] == "invalid-card-number"

    def test_create_payment_prompt_with_both(self):
        """Test create_payment_prompt with both card_type and error_type"""
        actions = [{"type": "Say", "phrase": "Try again"}]
        prompt = SwaigFunctionResult.create_payment_prompt(
            "payment-card-number", actions,
            card_type="visa", error_type="timeout"
        )

        assert prompt["card_type"] == "visa"
        assert prompt["error_type"] == "timeout"


class TestCreatePaymentAction:
    """Test create_payment_action() static method"""

    def test_create_payment_action_say(self):
        """Test create_payment_action with Say type"""
        action = SwaigFunctionResult.create_payment_action("Say", "Enter card number")
        assert action == {"type": "Say", "phrase": "Enter card number"}

    def test_create_payment_action_play(self):
        """Test create_payment_action with Play type"""
        action = SwaigFunctionResult.create_payment_action("Play", "https://example.com/prompt.mp3")
        assert action == {"type": "Play", "phrase": "https://example.com/prompt.mp3"}


class TestCreatePaymentParameter:
    """Test create_payment_parameter() static method"""

    def test_create_payment_parameter_basic(self):
        """Test create_payment_parameter basic usage"""
        param = SwaigFunctionResult.create_payment_parameter("store_id", "abc-123")
        assert param == {"name": "store_id", "value": "abc-123"}

    def test_create_payment_parameter_empty_value(self):
        """Test create_payment_parameter with empty value"""
        param = SwaigFunctionResult.create_payment_parameter("key", "")
        assert param == {"name": "key", "value": ""}


class TestToDictEdgeCases:
    """Test to_dict() edge cases"""

    def test_to_dict_empty_result(self):
        """Test to_dict with no response and no actions returns default"""
        result = SwaigFunctionResult()
        d = result.to_dict()
        assert d == {"response": "Action completed."}

    def test_to_dict_response_only(self):
        """Test to_dict with response only"""
        result = SwaigFunctionResult("Hello")
        d = result.to_dict()
        assert d == {"response": "Hello"}
        assert "action" not in d
        assert "post_process" not in d

    def test_to_dict_actions_only(self):
        """Test to_dict with actions but empty response"""
        result = SwaigFunctionResult()
        result.add_action("hangup", True)
        d = result.to_dict()
        assert "action" in d
        assert d["action"] == [{"hangup": True}]
        # Empty string response should not appear
        assert "response" not in d

    def test_to_dict_post_process_without_actions_not_included(self):
        """Test to_dict: post_process=True but no actions means post_process should not appear"""
        result = SwaigFunctionResult("Response", post_process=True)
        d = result.to_dict()
        assert "post_process" not in d
        assert d["response"] == "Response"

    def test_to_dict_post_process_with_actions(self):
        """Test to_dict: post_process=True with actions includes post_process"""
        result = SwaigFunctionResult("Response", post_process=True)
        result.add_action("stop", True)
        d = result.to_dict()
        assert d["post_process"] is True

    def test_to_dict_post_process_false_with_actions(self):
        """Test to_dict: post_process=False with actions does not include post_process"""
        result = SwaigFunctionResult("Response", post_process=False)
        result.add_action("stop", True)
        d = result.to_dict()
        assert "post_process" not in d


class TestSetEndOfSpeechTimeout:
    """Test set_end_of_speech_timeout() method"""

    def test_set_end_of_speech_timeout(self):
        """Test setting end of speech timeout"""
        result = SwaigFunctionResult().set_end_of_speech_timeout(500)
        assert result.action[0] == {"end_of_speech_timeout": 500}

    def test_set_end_of_speech_timeout_chaining(self):
        """Test set_end_of_speech_timeout returns self"""
        result = SwaigFunctionResult()
        ret = result.set_end_of_speech_timeout(300)
        assert ret is result


class TestSetSpeechEventTimeout:
    """Test set_speech_event_timeout() method"""

    def test_set_speech_event_timeout(self):
        """Test setting speech event timeout"""
        result = SwaigFunctionResult().set_speech_event_timeout(1000)
        assert result.action[0] == {"speech_event_timeout": 1000}

    def test_set_speech_event_timeout_chaining(self):
        """Test set_speech_event_timeout returns self"""
        result = SwaigFunctionResult()
        ret = result.set_speech_event_timeout(200)
        assert ret is result


class TestToggleFunctions:
    """Test toggle_functions() method"""

    def test_toggle_functions(self):
        """Test toggling functions"""
        toggles = [
            {"function": "get_weather", "active": True},
            {"function": "book_flight", "active": False}
        ]
        result = SwaigFunctionResult().toggle_functions(toggles)
        assert result.action[0] == {"toggle_functions": toggles}

    def test_toggle_functions_chaining(self):
        """Test toggle_functions returns self"""
        result = SwaigFunctionResult()
        ret = result.toggle_functions([])
        assert ret is result


class TestEnableFunctionsOnTimeout:
    """Test enable_functions_on_timeout() method"""

    def test_enable_functions_on_timeout_default(self):
        """Test enable_functions_on_timeout with default True"""
        result = SwaigFunctionResult().enable_functions_on_timeout()
        assert result.action[0] == {"functions_on_speaker_timeout": True}

    def test_enable_functions_on_timeout_false(self):
        """Test enable_functions_on_timeout with False"""
        result = SwaigFunctionResult().enable_functions_on_timeout(False)
        assert result.action[0] == {"functions_on_speaker_timeout": False}

    def test_enable_functions_on_timeout_chaining(self):
        """Test enable_functions_on_timeout returns self"""
        result = SwaigFunctionResult()
        ret = result.enable_functions_on_timeout()
        assert ret is result


class TestEnableExtensiveData:
    """Test enable_extensive_data() method"""

    def test_enable_extensive_data_default(self):
        """Test enable_extensive_data with default True"""
        result = SwaigFunctionResult().enable_extensive_data()
        assert result.action[0] == {"extensive_data": True}

    def test_enable_extensive_data_false(self):
        """Test enable_extensive_data with False"""
        result = SwaigFunctionResult().enable_extensive_data(False)
        assert result.action[0] == {"extensive_data": False}

    def test_enable_extensive_data_chaining(self):
        """Test enable_extensive_data returns self"""
        result = SwaigFunctionResult()
        ret = result.enable_extensive_data()
        assert ret is result


class TestUpdateSettings:
    """Test update_settings() method"""

    def test_update_settings(self):
        """Test updating agent runtime settings"""
        settings = {
            "temperature": 0.7,
            "top-p": 0.9,
            "confidence": 0.8
        }
        result = SwaigFunctionResult().update_settings(settings)
        assert result.action[0] == {"settings": settings}

    def test_update_settings_chaining(self):
        """Test update_settings returns self"""
        result = SwaigFunctionResult()
        ret = result.update_settings({"temperature": 0.5})
        assert ret is result


class TestSimulateUserInput:
    """Test simulate_user_input() method"""

    def test_simulate_user_input(self):
        """Test simulating user input"""
        result = SwaigFunctionResult().simulate_user_input("I want to book a flight")
        assert result.action[0] == {"user_input": "I want to book a flight"}

    def test_simulate_user_input_chaining(self):
        """Test simulate_user_input returns self"""
        result = SwaigFunctionResult()
        ret = result.simulate_user_input("text")
        assert ret is result


class TestSwitchContextEdgeCases:
    """Test switch_context() edge cases"""

    def test_switch_context_simple_string_only(self):
        """Test switch_context with only system_prompt uses simple string form"""
        result = SwaigFunctionResult().switch_context(system_prompt="You are a helpful bot")
        assert result.action[0] == {"context_switch": "You are a helpful bot"}

    def test_switch_context_full_object_with_all_params(self):
        """Test switch_context with all parameters uses object form"""
        result = SwaigFunctionResult().switch_context(
            system_prompt="New prompt",
            user_prompt="User msg",
            consolidate=True,
            full_reset=True
        )

        ctx = result.action[0]["context_switch"]
        assert isinstance(ctx, dict)
        assert ctx["system_prompt"] == "New prompt"
        assert ctx["user_prompt"] == "User msg"
        assert ctx["consolidate"] is True
        assert ctx["full_reset"] is True

    def test_switch_context_with_full_reset_only(self):
        """Test switch_context with full_reset triggers object form"""
        result = SwaigFunctionResult().switch_context(full_reset=True)

        ctx = result.action[0]["context_switch"]
        assert isinstance(ctx, dict)
        assert ctx["full_reset"] is True
        assert "system_prompt" not in ctx

    def test_switch_context_system_and_user_prompt(self):
        """Test switch_context with system_prompt and user_prompt uses object form"""
        result = SwaigFunctionResult().switch_context(
            system_prompt="Sys",
            user_prompt="User"
        )

        ctx = result.action[0]["context_switch"]
        assert isinstance(ctx, dict)
        assert ctx["system_prompt"] == "Sys"
        assert ctx["user_prompt"] == "User"

    def test_switch_context_system_prompt_with_consolidate(self):
        """Test switch_context with system_prompt and consolidate uses object form"""
        result = SwaigFunctionResult().switch_context(
            system_prompt="New prompt",
            consolidate=True
        )

        ctx = result.action[0]["context_switch"]
        assert isinstance(ctx, dict)
        assert ctx["system_prompt"] == "New prompt"
        assert ctx["consolidate"] is True

    def test_switch_context_no_args(self):
        """Test switch_context with no args produces empty dict form"""
        result = SwaigFunctionResult().switch_context()
        ctx = result.action[0]["context_switch"]
        assert isinstance(ctx, dict)
        assert ctx == {}

    def test_switch_context_chaining(self):
        """Test switch_context returns self for chaining"""
        result = SwaigFunctionResult()
        ret = result.switch_context(system_prompt="test")
        assert ret is result
