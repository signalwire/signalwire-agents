"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for AgentBase class
"""

import pytest
import json
import uuid
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional

from signalwire_agents.core.agent_base import AgentBase



class TestAgentBaseInitialization:
    """Test AgentBase initialization"""

    def _create_mock_agent(self, **kwargs):
        """Helper to create a properly mocked agent"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            agent = AgentBase(schema_validation=False, **kwargs)
        return agent

    def test_basic_initialization(self):
        """Test basic AgentBase initialization"""
        agent = self._create_mock_agent(name="test_agent")

        assert agent.get_name() == "test_agent"
        assert agent.agent_id is not None
        assert agent._auto_answer is True
        assert agent._record_call is False
        assert agent._use_pom is True
        assert agent.native_functions == []

    def test_initialization_with_custom_params(self):
        """Test AgentBase initialization with custom parameters"""
        agent = self._create_mock_agent(
            name="custom_agent",
            route="/custom",
            host="127.0.0.1",
            port=8080,
            basic_auth=("user", "pass"),
            use_pom=False,
            auto_answer=False,
            record_call=True,
            agent_id="custom-id",
            native_functions=["func1", "func2"]
        )

        assert agent.get_name() == "custom_agent"
        assert agent.agent_id == "custom-id"
        assert agent._auto_answer is False
        assert agent._record_call is True
        assert agent._use_pom is False
        assert agent.pom is None
        assert agent.native_functions == ["func1", "func2"]

    def test_initialization_with_pom_import_error(self):
        """Test AgentBase initialization with POM import error"""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'signalwire_pom'")):
            with pytest.raises(ImportError, match="No module named 'signalwire_pom'"):
                self._create_mock_agent(name="test_agent", use_pom=True)


class TestAgentBasePromptMethods:
    """Test AgentBase prompt-related methods"""

    def setup_method(self):
        """Set up test fixtures"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            self.agent = AgentBase("test_agent", use_pom=False, schema_validation=False)

    def test_set_prompt_text(self):
        """Test setting prompt text"""
        result = self.agent.set_prompt_text("You are a helpful assistant")

        assert result is self.agent  # Should return self for chaining
        assert self.agent._prompt_manager._prompt_text == "You are a helpful assistant"

    def test_set_post_prompt(self):
        """Test setting post-prompt text"""
        result = self.agent.set_post_prompt("End of conversation")

        assert result is self.agent
        assert self.agent._prompt_manager._post_prompt_text == "End of conversation"

    def test_get_prompt_with_raw_text(self):
        """Test get_prompt with raw text"""
        self.agent.set_prompt_text("Raw prompt text")

        result = self.agent.get_prompt()

        assert result == "Raw prompt text"

    def test_get_prompt_without_text(self):
        """Test get_prompt without any text set"""
        result = self.agent.get_prompt()

        assert result == "You are test_agent, a helpful AI assistant."

    def test_get_post_prompt(self):
        """Test get_post_prompt"""
        self.agent.set_post_prompt("Post prompt text")

        result = self.agent.get_post_prompt()

        assert result == "Post prompt text"

    def test_get_post_prompt_none(self):
        """Test get_post_prompt when none set"""
        result = self.agent.get_post_prompt()

        assert result is None


class TestAgentBaseConfigurationMethods:
    """Test AgentBase configuration methods"""

    def setup_method(self):
        """Set up test fixtures"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            self.agent = AgentBase("test_agent", schema_validation=False)

    def test_add_hint(self):
        """Test adding hints"""
        result = self.agent.add_hint("Test hint")

        assert result is self.agent
        assert "Test hint" in self.agent._hints

    def test_add_hints(self):
        """Test adding multiple hints"""
        hints = ["Hint 1", "Hint 2"]
        result = self.agent.add_hints(hints)

        assert result is self.agent
        assert all(hint in self.agent._hints for hint in hints)

    def test_add_language(self):
        """Test adding language configuration"""
        result = self.agent.add_language("English", "en", "alice")

        assert result is self.agent
        assert len(self.agent._languages) == 1

        language = self.agent._languages[0]
        assert language["name"] == "English"
        assert language["code"] == "en"
        assert language["voice"] == "alice"

    def test_add_pronunciation(self):
        """Test adding pronunciation rules"""
        result = self.agent.add_pronunciation("AI", "Artificial Intelligence")

        assert result is self.agent
        assert len(self.agent._pronounce) == 1

        rule = self.agent._pronounce[0]
        assert rule["replace"] == "AI"
        assert rule["with"] == "Artificial Intelligence"

    def test_set_param(self):
        """Test setting parameters"""
        result = self.agent.set_param("temperature", 0.7)

        assert result is self.agent
        assert self.agent._params["temperature"] == 0.7

    def test_set_params(self):
        """Test setting multiple parameters"""
        params = {"temperature": 0.7, "max_tokens": 100}
        result = self.agent.set_params(params)

        assert result is self.agent
        assert self.agent._params == params

    def test_set_global_data(self):
        """Test setting global data"""
        data = {"user_id": "123"}
        result = self.agent.set_global_data(data)

        assert result is self.agent
        assert self.agent._global_data == data

    def test_update_global_data(self):
        """Test updating global data"""
        self.agent._global_data = {"existing": "value"}
        new_data = {"new_key": "new_value"}

        result = self.agent.update_global_data(new_data)

        assert result is self.agent
        assert self.agent._global_data == {"existing": "value", "new_key": "new_value"}

    def test_set_native_functions(self):
        """Test setting native functions"""
        functions = ["func1", "func2"]
        result = self.agent.set_native_functions(functions)

        assert result is self.agent
        assert self.agent.native_functions == functions

    def test_add_function_include(self):
        """Test adding function includes"""
        result = self.agent.add_function_include("http://example.com", ["func1"])

        assert result is self.agent
        assert len(self.agent._function_includes) == 1

        include = self.agent._function_includes[0]
        assert include["url"] == "http://example.com"
        assert include["functions"] == ["func1"]


class TestAgentBaseToolMethods:
    """Test AgentBase tool-related methods"""

    def setup_method(self):
        """Set up test fixtures"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            self.agent = AgentBase("test_agent", schema_validation=False)

    def test_define_tool(self):
        """Test defining a tool"""
        def test_handler(arg1: str, arg2: int) -> str:
            return f"{arg1}_{arg2}"

        parameters = {
            "type": "object",
            "properties": {
                "arg1": {"type": "string"},
                "arg2": {"type": "integer"}
            }
        }

        result = self.agent.define_tool("test_tool", "Test description", parameters, test_handler)

        assert result is self.agent
        assert "test_tool" in self.agent._tool_registry._swaig_functions

    def test_register_swaig_function(self):
        """Test registering SWAIG function from dictionary"""
        function_dict = {
            "function": "test_func",
            "description": "Test function",
            "parameters": {"type": "object"},
        }

        result = self.agent.register_swaig_function(function_dict)

        assert result is self.agent
        assert "test_func" in self.agent._tool_registry._swaig_functions

    def test_define_tools(self):
        """Test define_tools method"""
        # Add a tool first
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        self.agent._tool_registry._swaig_functions["test_tool"] = mock_tool

        tools = self.agent.define_tools()

        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_on_function_call(self):
        """Test on_function_call method"""
        result = self.agent.on_function_call("test_func", {"arg": "value"})

        assert result == {"response": "Function 'test_func' not found"}

    def test_on_summary(self):
        """Test on_summary method"""
        # This is a hook method that should be overridden by subclasses
        # Should not raise any exceptions
        self.agent.on_summary({"summary": "test"})


class TestAgentBaseAuthMethods:
    """Test AgentBase authentication methods"""

    def setup_method(self):
        """Set up test fixtures"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            self.agent = AgentBase("test_agent", basic_auth=("user", "pass"), schema_validation=False)

    def test_validate_basic_auth_success(self):
        """Test successful basic auth validation"""
        result = self.agent.validate_basic_auth("user", "pass")

        assert result is True

    def test_validate_basic_auth_failure(self):
        """Test failed basic auth validation"""
        result = self.agent.validate_basic_auth("wrong", "creds")

        assert result is False

    def test_validate_basic_auth_no_auth_configured(self):
        """Test basic auth validation when no auth is configured"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            agent = AgentBase("test_agent", schema_validation=False)  # No basic_auth
            agent._basic_auth = (None, None)  # Explicitly set no auth

        result = agent.validate_basic_auth("any", "creds")

        assert result is False

    def test_get_basic_auth_credentials(self):
        """Test getting basic auth credentials"""
        username, password = self.agent.get_basic_auth_credentials()

        assert username == "user"
        assert password == "pass"

    def test_get_basic_auth_credentials_with_source(self):
        """Test getting basic auth credentials with source"""
        username, password, source = self.agent.get_basic_auth_credentials(include_source=True)

        assert username == "user"
        assert password == "pass"
        assert source == "provided"


class TestAgentBaseURLMethods:
    """Test AgentBase URL-related methods"""

    def setup_method(self):
        """Set up test fixtures"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            self.agent = AgentBase("test_agent", host="localhost", port=3000, route="/test", schema_validation=False)

    def test_get_full_url_basic(self):
        """Test getting full URL without auth"""
        url = self.agent.get_full_url()

        assert url == "http://localhost:3000/test"

    def test_get_full_url_with_auth(self):
        """Test getting full URL with auth"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            agent = AgentBase("test_agent", host="localhost", port=3000, route="/test",
                            basic_auth=("user", "pass"), schema_validation=False)

        url = agent.get_full_url(include_auth=True)

        assert url == "http://user:pass@localhost:3000/test"

    def test_set_web_hook_url(self):
        """Test setting webhook URL"""
        result = self.agent.set_web_hook_url("http://example.com/webhook")

        assert result is self.agent
        assert self.agent._web_hook_url_override == "http://example.com/webhook"

    def test_set_post_prompt_url(self):
        """Test setting post-prompt URL"""
        result = self.agent.set_post_prompt_url("http://example.com/post")

        assert result is self.agent
        assert self.agent._post_prompt_url_override == "http://example.com/post"

    def test_manual_set_proxy_url(self):
        """Test manually setting proxy URL"""
        result = self.agent.manual_set_proxy_url("http://proxy.example.com")

        assert result is self.agent
        assert self.agent._proxy_url_base == "http://proxy.example.com"


class TestAgentBaseSkillMethods:
    """Test AgentBase skill-related methods"""

    def setup_method(self):
        """Set up test fixtures"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            self.agent = AgentBase("test_agent", schema_validation=False)

        # Replace the real skill_manager with a mock after creation
        self.mock_skill_manager_instance = Mock()
        self.agent.skill_manager = self.mock_skill_manager_instance

    def test_add_skill(self):
        """Test adding a skill"""
        self.mock_skill_manager_instance.load_skill.return_value = (True, None)

        result = self.agent.add_skill("test_skill", {"param": "value"})

        assert result is self.agent
        self.mock_skill_manager_instance.load_skill.assert_called_once_with("test_skill", params={"param": "value"})

    def test_remove_skill(self):
        """Test removing a skill"""
        result = self.agent.remove_skill("test_skill")

        assert result is self.agent
        self.mock_skill_manager_instance.unload_skill.assert_called_once_with("test_skill")

    def test_list_skills(self):
        """Test listing skills"""
        self.mock_skill_manager_instance.list_loaded_skills.return_value = ["skill1", "skill2"]

        result = self.agent.list_skills()

        assert result == ["skill1", "skill2"]
        self.mock_skill_manager_instance.list_loaded_skills.assert_called_once()

    def test_has_skill(self):
        """Test checking if skill exists"""
        self.mock_skill_manager_instance.has_skill.return_value = True

        result = self.agent.has_skill("test_skill")

        assert result is True
        self.mock_skill_manager_instance.has_skill.assert_called_once_with("test_skill")


class TestAgentBaseTokenMethods:
    """Test AgentBase token-related methods"""

    def setup_method(self):
        """Set up test fixtures"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            self.agent = AgentBase("test_agent", schema_validation=False)

        # Replace the real session_manager with a mock after creation
        self.mock_session_manager_instance = Mock()
        self.agent._session_manager = self.mock_session_manager_instance

    def test_create_tool_token(self):
        """Test creating tool token"""
        self.mock_session_manager_instance.create_tool_token.return_value = "test_token"

        token = self.agent._create_tool_token("test_tool", "call_123")

        assert token == "test_token"
        self.mock_session_manager_instance.create_tool_token.assert_called_once_with("test_tool", "call_123")

    def test_validate_tool_token(self):
        """Test validating tool token"""
        # Add a mock function to the agent's tool registry
        mock_func = Mock()
        mock_func.secure = True
        self.agent._tool_registry._swaig_functions["test_tool"] = mock_func

        self.mock_session_manager_instance.validate_tool_token.return_value = True

        result = self.agent.validate_tool_token("test_tool", "test_token", "call_123")

        assert result is True
        self.mock_session_manager_instance.validate_tool_token.assert_called_once_with("test_tool", "test_token", "call_123")


class TestAgentBaseMiscMethods:
    """Test miscellaneous AgentBase methods"""

    def setup_method(self):
        """Set up test fixtures"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            self.agent = AgentBase("test_agent", schema_validation=False)

    def test_get_name(self):
        """Test getting agent name"""
        name = self.agent.get_name()

        assert name == "test_agent"

    def test_set_dynamic_config_callback(self):
        """Test setting dynamic config callback"""
        def callback(request_data, call_data, meta_data, config):
            pass

        result = self.agent.set_dynamic_config_callback(callback)

        assert result is self.agent
        assert self.agent._dynamic_config_callback == callback

    def test_on_request(self):
        """Test on_request method"""
        # on_request calls on_swml_request which returns None when
        # no dynamic config callback is set
        result = self.agent.on_request({"test": "data"})

        # Default implementation should return None
        assert result is None

    def test_on_swml_request(self):
        """Test on_swml_request method"""
        # When no dynamic config callback is set, on_swml_request returns None
        result = self.agent.on_swml_request({"test": "data"})

        # Default implementation should return None
        assert result is None


class TestAgentBaseDeclarativePrompts:
    """Test AgentBase declarative prompt sections"""

    def test_process_prompt_sections_dict(self):
        """Test processing declarative prompt sections from dict"""
        class TestAgent(AgentBase):
            PROMPT_SECTIONS = {
                "Instructions": "Follow these rules",
                "Rules": ["Rule 1", "Rule 2"],
                "Complex": {
                    "body": "Complex section",
                    "bullets": ["Bullet 1", "Bullet 2"],
                    "numbered": True
                }
            }

        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            with patch.object(TestAgent, 'prompt_add_section') as mock_add_section:
                agent = TestAgent("test_agent", schema_validation=False)

                # Should have called prompt_add_section for each section
                assert mock_add_section.call_count == 3

    def test_process_prompt_sections_no_pom(self):
        """Test processing prompt sections when POM is disabled"""
        class TestAgent(AgentBase):
            PROMPT_SECTIONS = {"Test": "Content"}

        with pytest.MonkeyPatch().context() as m:
            m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
            with patch.object(TestAgent, 'prompt_add_section') as mock_add_section:
                agent = TestAgent("test_agent", use_pom=False, schema_validation=False)

                # Should not call prompt_add_section when POM is disabled
                mock_add_section.assert_not_called()


# ========================================================================
# NEW TEST CLASSES BELOW
# ========================================================================


def _make_agent(**kwargs):
    """Module-level helper to create a properly mocked agent."""
    with pytest.MonkeyPatch().context() as m:
        m.setattr("signalwire_agents.core.agent_base.uvicorn", Mock())
        agent = AgentBase(schema_validation=False, **kwargs)
    return agent


class TestRenderSwml:
    """Test _render_swml() output structure."""

    def _make(self, **kw):
        return _make_agent(name="render_test", use_pom=False, **kw)

    # -- basic document structure --

    def test_render_swml_returns_valid_json(self):
        agent = self._make()
        doc = json.loads(agent._render_swml())
        assert "version" in doc
        assert "sections" in doc
        assert "main" in doc["sections"]

    def test_render_swml_version_is_1_0_0(self):
        agent = self._make()
        doc = json.loads(agent._render_swml())
        assert doc["version"] == "1.0.0"

    def test_render_swml_main_section_is_list(self):
        agent = self._make()
        doc = json.loads(agent._render_swml())
        assert isinstance(doc["sections"]["main"], list)

    def test_render_swml_contains_answer_verb(self):
        agent = self._make()
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        assert "answer" in verb_names

    def test_render_swml_contains_ai_verb(self):
        agent = self._make()
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        assert "ai" in verb_names

    def test_render_swml_answer_before_ai(self):
        agent = self._make()
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        assert verb_names.index("answer") < verb_names.index("ai")

    def test_render_swml_no_answer_when_auto_answer_false(self):
        agent = self._make(auto_answer=False)
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        assert "answer" not in verb_names

    def test_render_swml_ai_section_has_prompt(self):
        agent = self._make()
        agent.set_prompt_text("Hello world")
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        ai_verb = [v for v in verbs if isinstance(v, dict) and "ai" in v][0]
        ai_config = ai_verb["ai"]
        assert "prompt" in ai_config

    def test_render_swml_ai_prompt_text(self):
        agent = self._make()
        agent.set_prompt_text("Be a helpful agent")
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        prompt = ai_verb["ai"]["prompt"]
        if isinstance(prompt, dict):
            assert "Be a helpful agent" in prompt.get("text", "")
        else:
            assert "Be a helpful agent" in prompt

    def test_render_swml_with_post_prompt(self):
        agent = self._make()
        agent.set_post_prompt("Summarize the conversation")
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        ai_config = ai_verb["ai"]
        assert "post_prompt" in ai_config

    def test_render_swml_with_hints(self):
        agent = self._make()
        agent.add_hints(["hint1", "hint2"])
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        assert "hints" in ai_verb["ai"]
        assert "hint1" in ai_verb["ai"]["hints"]
        assert "hint2" in ai_verb["ai"]["hints"]

    def test_render_swml_with_languages(self):
        agent = self._make()
        agent.add_language("English", "en", "alice")
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        assert "languages" in ai_verb["ai"]
        assert ai_verb["ai"]["languages"][0]["name"] == "English"

    def test_render_swml_with_params(self):
        agent = self._make()
        agent.set_params({"temperature": 0.5})
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        assert "params" in ai_verb["ai"]
        assert ai_verb["ai"]["params"]["temperature"] == 0.5

    def test_render_swml_with_global_data(self):
        agent = self._make()
        agent.set_global_data({"key": "value"})
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        assert "global_data" in ai_verb["ai"]
        assert ai_verb["ai"]["global_data"]["key"] == "value"

    def test_render_swml_with_pronunciation(self):
        agent = self._make()
        agent.add_pronunciation("SQL", "sequel")
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        assert "pronounce" in ai_verb["ai"]
        assert ai_verb["ai"]["pronounce"][0]["replace"] == "SQL"

    def test_render_swml_with_record_call(self):
        agent = self._make(record_call=True)
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        assert "record_call" in verb_names

    def test_render_swml_record_call_before_ai(self):
        agent = self._make(record_call=True)
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        assert verb_names.index("record_call") < verb_names.index("ai")

    def test_render_swml_record_call_format(self):
        agent = self._make(record_call=True, record_format="wav", record_stereo=False)
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        record_verb = [v for v in verbs if isinstance(v, dict) and "record_call" in v][0]
        assert record_verb["record_call"]["format"] == "wav"
        assert record_verb["record_call"]["stereo"] is False

    def test_render_swml_with_native_functions(self):
        agent = self._make(native_functions=["transfer", "check_time"])
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        swaig = ai_verb["ai"].get("SWAIG", {})
        assert "native_functions" in swaig
        assert "transfer" in swaig["native_functions"]

    def test_render_swml_with_function_includes(self):
        agent = self._make()
        agent.add_function_include("http://example.com/funcs", ["fn1"])
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        swaig = ai_verb["ai"].get("SWAIG", {})
        assert "includes" in swaig
        assert swaig["includes"][0]["url"] == "http://example.com/funcs"


class TestEphemeralCopy:
    """Test _create_ephemeral_copy()."""

    def _make(self, **kw):
        return _make_agent(name="ephemeral_test", use_pom=False, **kw)

    def test_ephemeral_copy_returns_agent(self):
        agent = self._make()
        copy = agent._create_ephemeral_copy()
        assert isinstance(copy, AgentBase)

    def test_ephemeral_copy_is_marked_ephemeral(self):
        agent = self._make()
        copy = agent._create_ephemeral_copy()
        assert getattr(copy, '_is_ephemeral', False) is True

    def test_ephemeral_copy_has_same_name(self):
        agent = self._make()
        copy = agent._create_ephemeral_copy()
        assert copy.name == agent.name

    def test_ephemeral_params_independent(self):
        agent = self._make()
        agent.set_params({"temperature": 0.5})
        copy = agent._create_ephemeral_copy()
        copy._params["temperature"] = 0.9
        assert agent._params["temperature"] == 0.5

    def test_ephemeral_hints_independent(self):
        agent = self._make()
        agent.add_hint("original hint")
        copy = agent._create_ephemeral_copy()
        copy._hints.append("new hint")
        assert "new hint" not in agent._hints

    def test_ephemeral_global_data_independent(self):
        agent = self._make()
        agent.set_global_data({"key": "original"})
        copy = agent._create_ephemeral_copy()
        copy._global_data["key"] = "modified"
        assert agent._global_data["key"] == "original"

    def test_ephemeral_languages_independent(self):
        agent = self._make()
        agent.add_language("English", "en", "alice")
        copy = agent._create_ephemeral_copy()
        copy._languages.append({"name": "French", "code": "fr", "voice": "bob"})
        assert len(agent._languages) == 1

    def test_ephemeral_pronounce_independent(self):
        agent = self._make()
        agent.add_pronunciation("AI", "A I")
        copy = agent._create_ephemeral_copy()
        copy._pronounce.append({"replace": "SQL", "with": "sequel"})
        assert len(agent._pronounce) == 1

    def test_ephemeral_function_includes_independent(self):
        agent = self._make()
        agent.add_function_include("http://example.com", ["fn1"])
        copy = agent._create_ephemeral_copy()
        copy._function_includes.append({"url": "http://other.com", "functions": ["fn2"]})
        assert len(agent._function_includes) == 1

    def test_ephemeral_pre_answer_verbs_independent(self):
        agent = self._make()
        agent.add_pre_answer_verb("sleep", {"time": 1000})
        copy = agent._create_ephemeral_copy()
        copy._pre_answer_verbs.append(("set", {"var": "x"}))
        assert len(agent._pre_answer_verbs) == 1

    def test_ephemeral_post_answer_verbs_independent(self):
        agent = self._make()
        agent.add_post_answer_verb("play", {"url": "say:hello"})
        copy = agent._create_ephemeral_copy()
        copy._post_answer_verbs.append(("sleep", {"time": 500}))
        assert len(agent._post_answer_verbs) == 1

    def test_ephemeral_post_ai_verbs_independent(self):
        agent = self._make()
        agent.add_post_ai_verb("hangup", {})
        copy = agent._create_ephemeral_copy()
        copy._post_ai_verbs.append(("request", {"url": "http://x.com"}))
        assert len(agent._post_ai_verbs) == 1

    def test_ephemeral_prompt_text_independent(self):
        agent = self._make()
        agent.set_prompt_text("original prompt")
        copy = agent._create_ephemeral_copy()
        copy._prompt_manager._prompt_text = "modified prompt"
        assert agent._prompt_manager._prompt_text == "original prompt"

    def test_ephemeral_post_prompt_text_independent(self):
        agent = self._make()
        agent.set_post_prompt("original post")
        copy = agent._create_ephemeral_copy()
        copy._prompt_manager._post_prompt_text = "modified post"
        assert agent._prompt_manager._post_prompt_text == "original post"

    def test_ephemeral_swaig_query_params_independent(self):
        agent = self._make()
        agent.add_swaig_query_params({"tier": "premium"})
        copy = agent._create_ephemeral_copy()
        copy._swaig_query_params["tier"] = "basic"
        assert agent._swaig_query_params["tier"] == "premium"

    def test_ephemeral_tool_registry_independent(self):
        agent = self._make()
        copy = agent._create_ephemeral_copy()
        assert id(copy._tool_registry) != id(agent._tool_registry)

    def test_ephemeral_prompt_manager_independent(self):
        agent = self._make()
        copy = agent._create_ephemeral_copy()
        assert id(copy._prompt_manager) != id(agent._prompt_manager)

    def test_ephemeral_skill_manager_independent(self):
        agent = self._make()
        copy = agent._create_ephemeral_copy()
        assert id(copy.skill_manager) != id(agent.skill_manager)

    def test_ephemeral_answer_config_independent(self):
        agent = self._make()
        agent.add_answer_verb({"max_duration": 3600})
        copy = agent._create_ephemeral_copy()
        copy._answer_config["max_duration"] = 7200
        assert agent._answer_config["max_duration"] == 3600


class TestVerbInsertion:
    """Test add_pre_answer_verb(), add_post_answer_verb(), add_post_ai_verb()."""

    def _make(self, **kw):
        return _make_agent(name="verb_test", use_pom=False, **kw)

    # -- pre-answer verbs --

    def test_add_pre_answer_verb_safe_verb(self):
        agent = self._make()
        result = agent.add_pre_answer_verb("sleep", {"time": 1000})
        assert result is agent
        assert len(agent._pre_answer_verbs) == 1
        assert agent._pre_answer_verbs[0] == ("sleep", {"time": 1000})

    def test_add_pre_answer_verb_multiple(self):
        agent = self._make()
        agent.add_pre_answer_verb("sleep", {"time": 500})
        agent.add_pre_answer_verb("set", {"var": "x", "val": "1"})
        assert len(agent._pre_answer_verbs) == 2

    def test_add_pre_answer_verb_unsafe_raises(self):
        agent = self._make()
        with pytest.raises(ValueError, match="not safe for pre-answer"):
            agent.add_pre_answer_verb("record_call", {"format": "mp4"})

    def test_add_pre_answer_verb_auto_answer_verb_with_flag(self):
        """play verb with auto_answer=False should not raise"""
        agent = self._make()
        agent.add_pre_answer_verb("play", {"urls": ["ring:us"], "auto_answer": False})
        assert len(agent._pre_answer_verbs) == 1

    def test_add_pre_answer_verb_auto_answer_verb_warns_without_flag(self):
        """play verb without auto_answer=False should still add (warns only)"""
        agent = self._make()
        agent.add_pre_answer_verb("play", {"urls": ["ring:us"]})
        assert len(agent._pre_answer_verbs) == 1

    def test_add_pre_answer_verb_transfer(self):
        agent = self._make()
        agent.add_pre_answer_verb("transfer", {"dest": "sip:foo@bar.com"})
        assert agent._pre_answer_verbs[0][0] == "transfer"

    def test_add_pre_answer_verb_connect_auto_answer_false(self):
        agent = self._make()
        agent.add_pre_answer_verb("connect", {"from": "+15551234567", "auto_answer": False})
        assert agent._pre_answer_verbs[0][0] == "connect"

    # -- post-answer verbs --

    def test_add_post_answer_verb_basic(self):
        agent = self._make()
        result = agent.add_post_answer_verb("play", {"url": "say:Welcome"})
        assert result is agent
        assert len(agent._post_answer_verbs) == 1

    def test_add_post_answer_verb_multiple(self):
        agent = self._make()
        agent.add_post_answer_verb("play", {"url": "say:Hello"})
        agent.add_post_answer_verb("sleep", {"time": 500})
        assert len(agent._post_answer_verbs) == 2
        assert agent._post_answer_verbs[0][0] == "play"
        assert agent._post_answer_verbs[1][0] == "sleep"

    # -- post-ai verbs --

    def test_add_post_ai_verb_basic(self):
        agent = self._make()
        result = agent.add_post_ai_verb("hangup", {})
        assert result is agent
        assert len(agent._post_ai_verbs) == 1

    def test_add_post_ai_verb_multiple(self):
        agent = self._make()
        agent.add_post_ai_verb("request", {"url": "http://api.com/log", "method": "POST"})
        agent.add_post_ai_verb("hangup", {})
        assert len(agent._post_ai_verbs) == 2

    # -- clearing verbs --

    def test_clear_pre_answer_verbs(self):
        agent = self._make()
        agent.add_pre_answer_verb("sleep", {"time": 500})
        result = agent.clear_pre_answer_verbs()
        assert result is agent
        assert len(agent._pre_answer_verbs) == 0

    def test_clear_post_answer_verbs(self):
        agent = self._make()
        agent.add_post_answer_verb("play", {"url": "say:test"})
        result = agent.clear_post_answer_verbs()
        assert result is agent
        assert len(agent._post_answer_verbs) == 0

    def test_clear_post_ai_verbs(self):
        agent = self._make()
        agent.add_post_ai_verb("hangup", {})
        result = agent.clear_post_ai_verbs()
        assert result is agent
        assert len(agent._post_ai_verbs) == 0

    # -- verb ordering in rendered SWML --

    def test_pre_answer_verbs_before_answer_in_swml(self):
        agent = self._make()
        agent.add_pre_answer_verb("sleep", {"time": 500})
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        assert verb_names.index("sleep") < verb_names.index("answer")

    def test_post_answer_verbs_between_answer_and_ai(self):
        agent = self._make()
        agent.add_post_answer_verb("play", {"url": "say:Welcome"})
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        answer_idx = verb_names.index("answer")
        play_idx = verb_names.index("play")
        ai_idx = verb_names.index("ai")
        assert answer_idx < play_idx < ai_idx

    def test_post_ai_verbs_after_ai(self):
        agent = self._make()
        agent.add_post_ai_verb("hangup", {})
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        assert verb_names.index("ai") < verb_names.index("hangup")

    def test_full_verb_ordering(self):
        """Test all five phases appear in correct order."""
        agent = self._make(record_call=True)
        agent.add_pre_answer_verb("sleep", {"time": 200})
        agent.add_post_answer_verb("play", {"url": "say:Welcome"})
        agent.add_post_ai_verb("hangup", {})
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        verb_names = [list(v.keys())[0] for v in verbs if isinstance(v, dict)]
        # sleep < answer < record_call < play < ai < hangup
        assert verb_names.index("sleep") < verb_names.index("answer")
        assert verb_names.index("answer") < verb_names.index("record_call")
        assert verb_names.index("record_call") < verb_names.index("play")
        assert verb_names.index("play") < verb_names.index("ai")
        assert verb_names.index("ai") < verb_names.index("hangup")

    def test_add_answer_verb_config(self):
        agent = self._make()
        result = agent.add_answer_verb({"max_duration": 3600})
        assert result is agent
        assert agent._answer_config == {"max_duration": 3600}

    def test_add_answer_verb_none_config(self):
        agent = self._make()
        agent.add_answer_verb(None)
        assert agent._answer_config == {}

    def test_answer_verb_config_in_swml(self):
        agent = self._make()
        agent.add_answer_verb({"max_duration": 3600})
        doc = json.loads(agent._render_swml())
        verbs = doc["sections"]["main"]
        answer_verb = [v for v in verbs if isinstance(v, dict) and "answer" in v][0]
        assert answer_verb["answer"]["max_duration"] == 3600


class TestSipRouting:
    """Test SIP routing methods."""

    def _make(self, **kw):
        return _make_agent(name="sip_test", use_pom=False, **kw)

    def test_register_sip_username(self):
        agent = self._make()
        result = agent.register_sip_username("alice")
        assert result is agent
        assert "alice" in agent._sip_usernames

    def test_register_sip_username_lowercased(self):
        agent = self._make()
        agent.register_sip_username("Alice")
        assert "alice" in agent._sip_usernames

    def test_register_multiple_sip_usernames(self):
        agent = self._make()
        agent.register_sip_username("alice")
        agent.register_sip_username("bob")
        assert "alice" in agent._sip_usernames
        assert "bob" in agent._sip_usernames

    def test_register_sip_username_deduplication(self):
        agent = self._make()
        agent.register_sip_username("alice")
        agent.register_sip_username("alice")
        assert len(agent._sip_usernames) == 1

    def test_auto_map_sip_usernames(self):
        agent = self._make()
        result = agent.auto_map_sip_usernames()
        assert result is agent
        # Agent name is "sip_test", so "sip_test" should be registered
        assert "sip_test" in agent._sip_usernames

    def test_auto_map_registers_route_variant(self):
        agent = _make_agent(name="myagent", route="/myroute", use_pom=False)
        agent.auto_map_sip_usernames()
        assert "myagent" in agent._sip_usernames
        assert "myroute" in agent._sip_usernames

    def test_auto_map_no_vowels_variant(self):
        """For names longer than 3 chars, a no-vowels variant should be registered."""
        agent = _make_agent(name="testing", use_pom=False)
        agent.auto_map_sip_usernames()
        assert "testing" in agent._sip_usernames
        # "tstng" is "testing" without vowels
        assert "tstng" in agent._sip_usernames

    def test_auto_map_short_name_no_vowel_variant(self):
        """Names <= 3 chars should NOT get a no-vowels variant."""
        agent = _make_agent(name="abc", use_pom=False)
        agent.auto_map_sip_usernames()
        assert "abc" in agent._sip_usernames
        # Should not have a no-vowels variant for short names
        assert len([u for u in agent._sip_usernames if u != "abc"]) <= 1  # only route variant

    def test_enable_sip_routing_returns_self(self):
        agent = self._make()
        result = agent.enable_sip_routing()
        assert result is agent

    def test_enable_sip_routing_auto_map_true(self):
        agent = self._make()
        agent.enable_sip_routing(auto_map=True)
        # Should have registered at least the agent name
        assert hasattr(agent, '_sip_usernames')
        assert "sip_test" in agent._sip_usernames

    def test_enable_sip_routing_auto_map_false(self):
        agent = self._make()
        agent.enable_sip_routing(auto_map=False)
        # With auto_map=False, _sip_usernames might not be populated
        usernames = getattr(agent, '_sip_usernames', set())
        # Should NOT have auto-mapped the agent name
        assert "sip_test" not in usernames


class TestUrlBuilding:
    """Test get_full_url() across different execution modes."""

    def _make(self, **kw):
        return _make_agent(name="url_test", use_pom=False, **kw)

    # -- Server mode (default) --

    def test_server_mode_basic_url(self):
        agent = self._make(host="localhost", port=3000, route="/test")
        url = agent.get_full_url()
        assert "localhost" in url
        assert "3000" in url
        assert "/test" in url

    def test_server_mode_with_auth(self):
        agent = _make_agent(
            name="url_test", host="localhost", port=3000,
            route="/test", basic_auth=("user", "pass"), use_pom=False
        )
        url = agent.get_full_url(include_auth=True)
        assert "user:pass@" in url

    def test_server_mode_without_auth(self):
        agent = _make_agent(
            name="url_test", host="localhost", port=3000,
            route="/test", basic_auth=("user", "pass"), use_pom=False
        )
        url = agent.get_full_url(include_auth=False)
        assert "user:pass@" not in url

    # -- Proxy URL --

    def test_proxy_url_takes_precedence(self):
        agent = self._make(host="localhost", port=3000)
        agent._proxy_url_base = "https://proxy.example.com/agent"
        url = agent.get_full_url()
        assert url == "https://proxy.example.com/agent"

    def test_proxy_url_with_auth(self):
        agent = _make_agent(
            name="url_test", host="localhost", port=3000,
            basic_auth=("u", "p"), use_pom=False
        )
        agent._proxy_url_base = "https://proxy.example.com/agent"
        url = agent.get_full_url(include_auth=True)
        assert "u:p@" in url

    def test_proxy_url_trailing_slash_stripped(self):
        agent = self._make()
        agent._proxy_url_base = "https://proxy.example.com/"
        url = agent.get_full_url()
        assert not url.endswith("/")

    # -- CGI mode --

    def test_cgi_mode_url(self):
        agent = self._make(route="/cgitest")
        env = {
            "GATEWAY_INTERFACE": "CGI/1.1",
            "HTTP_HOST": "example.com",
            "SCRIPT_NAME": "/cgi-bin/agent.py",
        }
        with patch.dict(os.environ, env, clear=False):
            # Ensure proxy URL is not set
            agent._proxy_url_base = None
            url = agent.get_full_url()
        assert "example.com" in url
        assert "/cgi-bin/agent.py" in url

    def test_cgi_mode_https(self):
        agent = self._make(route="/")
        env = {
            "GATEWAY_INTERFACE": "CGI/1.1",
            "HTTPS": "on",
            "HTTP_HOST": "secure.example.com",
            "SCRIPT_NAME": "/app",
        }
        with patch.dict(os.environ, env, clear=False):
            agent._proxy_url_base = None
            url = agent.get_full_url()
        assert url.startswith("https://")

    # -- Lambda mode --

    def test_lambda_mode_with_function_url(self):
        agent = self._make(route="/")
        env = {
            "AWS_LAMBDA_FUNCTION_NAME": "my-func",
            "AWS_LAMBDA_FUNCTION_URL": "https://abc123.lambda-url.us-east-1.on.aws",
        }
        with patch.dict(os.environ, env, clear=False):
            agent._proxy_url_base = None
            url = agent.get_full_url()
        assert "abc123.lambda-url.us-east-1.on.aws" in url

    def test_lambda_mode_fallback_construction(self):
        agent = self._make(route="/")
        env = {
            "AWS_LAMBDA_FUNCTION_NAME": "my-func",
            "AWS_REGION": "us-west-2",
        }
        # Remove AWS_LAMBDA_FUNCTION_URL if set
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AWS_LAMBDA_FUNCTION_URL", None)
            os.environ.pop("GATEWAY_INTERFACE", None)
            agent._proxy_url_base = None
            url = agent.get_full_url()
        assert "my-func" in url
        assert "us-west-2" in url

    # -- Google Cloud Function mode --

    def test_gcf_mode_url(self):
        agent = self._make(route="/")
        env = {
            "K_SERVICE": "my-service",
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "FUNCTION_REGION": "us-central1",
        }
        # Remove vars that would trigger other modes
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            os.environ.pop("LAMBDA_TASK_ROOT", None)
            os.environ.pop("GATEWAY_INTERFACE", None)
            agent._proxy_url_base = None
            url = agent.get_full_url()
        assert "my-project" in url
        assert "my-service" in url

    def test_gcf_mode_no_project(self):
        agent = self._make(route="/")
        env = {
            "K_SERVICE": "my-service",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            os.environ.pop("LAMBDA_TASK_ROOT", None)
            os.environ.pop("GATEWAY_INTERFACE", None)
            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            os.environ.pop("GCP_PROJECT", None)
            agent._proxy_url_base = None
            url = agent.get_full_url()
        # Falls back to localhost:8080
        assert "localhost" in url

    # -- Azure Function mode --

    def test_azure_mode_url(self):
        agent = self._make(route="/")
        env = {
            "FUNCTIONS_WORKER_RUNTIME": "python",
            "WEBSITE_SITE_NAME": "my-app",
            "AZURE_FUNCTION_NAME": "my-func",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            os.environ.pop("LAMBDA_TASK_ROOT", None)
            os.environ.pop("GATEWAY_INTERFACE", None)
            os.environ.pop("K_SERVICE", None)
            os.environ.pop("FUNCTION_TARGET", None)
            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            agent._proxy_url_base = None
            url = agent.get_full_url()
        assert "my-app.azurewebsites.net" in url
        assert "my-func" in url

    def test_azure_mode_no_app_name(self):
        agent = self._make(route="/")
        env = {
            "FUNCTIONS_WORKER_RUNTIME": "python",
            "AZURE_FUNCTION_NAME": "my-func",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            os.environ.pop("LAMBDA_TASK_ROOT", None)
            os.environ.pop("GATEWAY_INTERFACE", None)
            os.environ.pop("K_SERVICE", None)
            os.environ.pop("FUNCTION_TARGET", None)
            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            os.environ.pop("WEBSITE_SITE_NAME", None)
            os.environ.pop("AZURE_FUNCTIONS_APP_NAME", None)
            agent._proxy_url_base = None
            url = agent.get_full_url()
        # Falls back to localhost:7071
        assert "localhost:7071" in url

    # -- Manual proxy URL --

    def test_manual_set_proxy_url(self):
        agent = self._make()
        result = agent.manual_set_proxy_url("https://custom.example.com")
        assert result is agent
        assert agent._proxy_url_base == "https://custom.example.com"

    # -- Webhook URL overrides --

    def test_set_web_hook_url(self):
        agent = self._make()
        result = agent.set_web_hook_url("https://webhook.example.com/swaig")
        assert result is agent
        assert agent._web_hook_url_override == "https://webhook.example.com/swaig"

    def test_set_post_prompt_url(self):
        agent = self._make()
        result = agent.set_post_prompt_url("https://webhook.example.com/pp")
        assert result is agent
        assert agent._post_prompt_url_override == "https://webhook.example.com/pp"


class TestToolRegistration:
    """Test define_tool() with parameter schemas."""

    def _make(self, **kw):
        return _make_agent(name="tool_test", use_pom=False, **kw)

    def test_define_tool_basic(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        result = agent.define_tool(
            "my_tool",
            "A test tool",
            {"type": "object", "properties": {"name": {"type": "string"}}},
            handler
        )
        assert result is agent
        assert "my_tool" in agent._tool_registry._swaig_functions

    def test_define_tool_with_required(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        agent.define_tool(
            "req_tool",
            "Tool with required params",
            {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
            handler,
            required=["name"]
        )
        func = agent._tool_registry._swaig_functions["req_tool"]
        assert "name" in func.required

    def test_define_tool_secure_default(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        agent.define_tool("sec_tool", "Secure tool", {}, handler)
        func = agent._tool_registry._swaig_functions["sec_tool"]
        assert func.secure is True

    def test_define_tool_not_secure(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        agent.define_tool("unsec_tool", "Unsecured tool", {}, handler, secure=False)
        func = agent._tool_registry._swaig_functions["unsec_tool"]
        assert func.secure is False

    def test_define_tool_duplicate_raises(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        agent.define_tool("dup_tool", "First", {}, handler)
        with pytest.raises(ValueError, match="already exists"):
            agent.define_tool("dup_tool", "Second", {}, handler)

    def test_define_tool_with_webhook_url(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        agent.define_tool(
            "webhook_tool", "External tool", {}, handler,
            webhook_url="https://external.com/api"
        )
        func = agent._tool_registry._swaig_functions["webhook_tool"]
        assert func.webhook_url == "https://external.com/api"
        assert func.is_external is True

    def test_define_tool_description(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        agent.define_tool("desc_tool", "My description", {}, handler)
        func = agent._tool_registry._swaig_functions["desc_tool"]
        assert func.description == "My description"

    def test_define_tool_complex_parameters(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"]
        }
        agent.define_tool("complex_tool", "Complex params", params, handler)
        func = agent._tool_registry._swaig_functions["complex_tool"]
        assert func.parameters == params

    def test_define_tools_returns_list(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        agent.define_tool("tool_a", "Tool A", {}, handler)
        agent.define_tool("tool_b", "Tool B", {}, handler)
        tools = agent.define_tools()
        assert len(tools) == 2

    def test_register_swaig_function_raw_dict(self):
        agent = self._make()
        func_dict = {
            "function": "data_map_fn",
            "description": "A data map function",
            "parameters": {"type": "object", "properties": {}},
            "data_map": {"expressions": []}
        }
        result = agent.register_swaig_function(func_dict)
        assert result is agent
        assert "data_map_fn" in agent._tool_registry._swaig_functions

    def test_register_swaig_function_in_define_tools(self):
        agent = self._make()
        func_dict = {
            "function": "dm_fn",
            "description": "DM function",
            "parameters": {"type": "object"},
        }
        agent.register_swaig_function(func_dict)
        tools = agent.define_tools()
        assert len(tools) == 1

    def test_on_function_call_missing_function(self):
        agent = self._make()
        result = agent.on_function_call("nonexistent", {"arg": "val"})
        assert "not found" in result["response"]

    def test_on_function_call_data_map_function(self):
        agent = self._make()
        agent.register_swaig_function({
            "function": "dm_fn",
            "description": "DM",
            "parameters": {},
            "data_map": {}
        })
        result = agent.on_function_call("dm_fn", {})
        assert "Data map" in result["response"] or "data_map" in result["response"].lower()

    def test_on_function_call_success(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": f"got {args.get('x')}"}

        agent.define_tool("fn", "test", {"type": "object", "properties": {"x": {"type": "string"}}}, handler)
        result = agent.on_function_call("fn", {"x": "hello"})
        assert "got hello" in str(result)

    def test_on_function_call_handler_exception(self):
        agent = self._make()
        def bad_handler(args, raw):
            raise RuntimeError("boom")

        agent.define_tool("bad_fn", "bad", {}, bad_handler)
        result = agent.on_function_call("bad_fn", {})
        assert "Error" in result["response"] or "boom" in result["response"]

    def test_tool_appears_in_rendered_swml(self):
        agent = self._make()
        def handler(args, raw):
            return {"response": "ok"}

        agent.define_tool(
            "rendered_tool", "A tool for render test",
            {"type": "object", "properties": {"q": {"type": "string"}}},
            handler
        )
        doc = json.loads(agent._render_swml())
        ai_verb = [v for v in doc["sections"]["main"] if isinstance(v, dict) and "ai" in v][0]
        swaig = ai_verb["ai"].get("SWAIG", {})
        assert "functions" in swaig
        tool_names = [f["function"] for f in swaig["functions"]]
        assert "rendered_tool" in tool_names


class TestSkillIntegration:
    """Test add_skill() lifecycle."""

    def _make(self, **kw):
        return _make_agent(name="skill_test", use_pom=False, **kw)

    def test_add_skill_success(self):
        agent = self._make()
        agent.skill_manager = Mock()
        agent.skill_manager.load_skill.return_value = (True, "")
        result = agent.add_skill("test_skill", {"param": "value"})
        assert result is agent
        agent.skill_manager.load_skill.assert_called_once_with("test_skill", params={"param": "value"})

    def test_add_skill_failure_raises(self):
        agent = self._make()
        agent.skill_manager = Mock()
        agent.skill_manager.load_skill.return_value = (False, "Skill not found")
        with pytest.raises(ValueError, match="Failed to load skill"):
            agent.add_skill("bad_skill")

    def test_add_skill_no_params(self):
        agent = self._make()
        agent.skill_manager = Mock()
        agent.skill_manager.load_skill.return_value = (True, "")
        agent.add_skill("simple_skill")
        agent.skill_manager.load_skill.assert_called_once_with("simple_skill", params=None)

    def test_remove_skill(self):
        agent = self._make()
        agent.skill_manager = Mock()
        result = agent.remove_skill("some_skill")
        assert result is agent
        agent.skill_manager.unload_skill.assert_called_once_with("some_skill")

    def test_list_skills(self):
        agent = self._make()
        agent.skill_manager = Mock()
        agent.skill_manager.list_loaded_skills.return_value = ["skill_a", "skill_b"]
        result = agent.list_skills()
        assert result == ["skill_a", "skill_b"]

    def test_has_skill_true(self):
        agent = self._make()
        agent.skill_manager = Mock()
        agent.skill_manager.has_skill.return_value = True
        assert agent.has_skill("existing_skill") is True

    def test_has_skill_false(self):
        agent = self._make()
        agent.skill_manager = Mock()
        agent.skill_manager.has_skill.return_value = False
        assert agent.has_skill("missing_skill") is False

    def test_skill_manager_initialized(self):
        """Verify skill_manager is a SkillManager by default."""
        agent = self._make()
        from signalwire_agents.core.skill_manager import SkillManager
        assert isinstance(agent.skill_manager, SkillManager)

    def test_skill_manager_agent_reference(self):
        agent = self._make()
        assert agent.skill_manager.agent is agent


class TestSwaigQueryParams:
    """Test SWAIG query parameter methods."""

    def _make(self, **kw):
        return _make_agent(name="qp_test", use_pom=False, **kw)

    def test_add_swaig_query_params(self):
        agent = self._make()
        result = agent.add_swaig_query_params({"tier": "premium"})
        assert result is agent
        assert agent._swaig_query_params["tier"] == "premium"

    def test_add_swaig_query_params_merge(self):
        agent = self._make()
        agent.add_swaig_query_params({"a": "1"})
        agent.add_swaig_query_params({"b": "2"})
        assert agent._swaig_query_params == {"a": "1", "b": "2"}

    def test_add_swaig_query_params_overwrite(self):
        agent = self._make()
        agent.add_swaig_query_params({"a": "1"})
        agent.add_swaig_query_params({"a": "2"})
        assert agent._swaig_query_params["a"] == "2"

    def test_clear_swaig_query_params(self):
        agent = self._make()
        agent.add_swaig_query_params({"x": "y"})
        result = agent.clear_swaig_query_params()
        assert result is agent
        assert agent._swaig_query_params == {}

    def test_add_swaig_query_params_none(self):
        agent = self._make()
        agent.add_swaig_query_params(None)
        assert agent._swaig_query_params == {}

    def test_add_swaig_query_params_empty(self):
        agent = self._make()
        agent.add_swaig_query_params({})
        assert agent._swaig_query_params == {}


class TestDynamicConfig:
    """Test dynamic configuration callback."""

    def _make(self, **kw):
        return _make_agent(name="dynconfig_test", use_pom=False, **kw)

    def test_set_dynamic_config_callback(self):
        agent = self._make()
        def callback(qp, bp, h, a):
            pass
        result = agent.set_dynamic_config_callback(callback)
        assert result is agent
        assert agent._dynamic_config_callback is callback

    def test_dynamic_config_callback_initially_none(self):
        agent = self._make()
        assert agent._dynamic_config_callback is None

    def test_on_request_returns_none_without_callback(self):
        agent = self._make()
        result = agent.on_request({"test": "data"})
        assert result is None


class TestFindSummary:
    """Test _find_summary_in_post_data."""

    def _make(self, **kw):
        return _make_agent(name="summary_test", use_pom=False, **kw)

    def test_find_summary_none_body(self):
        agent = self._make()
        result = agent._find_summary_in_post_data(None, agent.log)
        assert result is None

    def test_find_summary_empty_body(self):
        agent = self._make()
        result = agent._find_summary_in_post_data({}, agent.log)
        assert result is None

    def test_find_summary_direct_key(self):
        agent = self._make()
        result = agent._find_summary_in_post_data({"summary": {"text": "hello"}}, agent.log)
        assert result == {"text": "hello"}

    def test_find_summary_from_post_prompt_data_parsed(self):
        agent = self._make()
        body = {"post_prompt_data": {"parsed": [{"outcome": "success"}]}}
        result = agent._find_summary_in_post_data(body, agent.log)
        assert result == {"outcome": "success"}

    def test_find_summary_from_post_prompt_data_raw_json(self):
        agent = self._make()
        body = {"post_prompt_data": {"raw": '{"outcome": "done"}'}}
        result = agent._find_summary_in_post_data(body, agent.log)
        assert result == {"outcome": "done"}

    def test_find_summary_from_post_prompt_data_raw_not_json(self):
        agent = self._make()
        body = {"post_prompt_data": {"raw": "just a string"}}
        result = agent._find_summary_in_post_data(body, agent.log)
        assert result == "just a string"

    def test_find_summary_no_matching_key(self):
        agent = self._make()
        result = agent._find_summary_in_post_data({"other": "data"}, agent.log)
        assert result is None


class TestOnSummary:
    """Test on_summary hook."""

    def _make(self, **kw):
        return _make_agent(name="summary_hook_test", use_pom=False, **kw)

    def test_on_summary_does_not_raise(self):
        agent = self._make()
        # Default does nothing
        agent.on_summary({"summary": "test"})

    def test_on_summary_with_raw_data(self):
        agent = self._make()
        agent.on_summary({"summary": "test"}, raw_data={"call_id": "123"})

    def test_on_summary_none(self):
        agent = self._make()
        agent.on_summary(None)


class TestAgentId:
    """Test agent_id generation."""

    def test_auto_generated_agent_id(self):
        agent = _make_agent(name="id_test", use_pom=False)
        assert agent.agent_id is not None
        # Should be a valid UUID
        uuid.UUID(agent.agent_id)

    def test_custom_agent_id(self):
        agent = _make_agent(name="id_test", agent_id="custom-123", use_pom=False)
        assert agent.agent_id == "custom-123"
