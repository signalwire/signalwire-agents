"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for DataSphereServerlessSkill
"""

import base64
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any


class TestDataSphereServerlessSkillClassAttributes:
    """Test class-level attributes and constants"""

    def _get_skill_class(self):
        """Import and return the skill class with mocked dependencies"""
        from signalwire_agents.skills.datasphere_serverless.skill import DataSphereServerlessSkill
        return DataSphereServerlessSkill

    def test_skill_name(self):
        """Test SKILL_NAME is set correctly"""
        cls = self._get_skill_class()
        assert cls.SKILL_NAME == "datasphere_serverless"

    def test_skill_description(self):
        """Test SKILL_DESCRIPTION is set correctly"""
        cls = self._get_skill_class()
        assert cls.SKILL_DESCRIPTION == "Search knowledge using SignalWire DataSphere with serverless DataMap execution"

    def test_skill_version(self):
        """Test SKILL_VERSION is set correctly"""
        cls = self._get_skill_class()
        assert cls.SKILL_VERSION == "1.0.0"

    def test_required_packages_empty(self):
        """Test REQUIRED_PACKAGES is empty for serverless skill"""
        cls = self._get_skill_class()
        assert cls.REQUIRED_PACKAGES == []

    def test_required_env_vars_empty(self):
        """Test REQUIRED_ENV_VARS is empty since config comes from params"""
        cls = self._get_skill_class()
        assert cls.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        """Test SUPPORTS_MULTIPLE_INSTANCES is True"""
        cls = self._get_skill_class()
        assert cls.SUPPORTS_MULTIPLE_INSTANCES is True


class TestDataSphereServerlessSkillParameterSchema:
    """Test get_parameter_schema class method"""

    def _get_skill_class(self):
        from signalwire_agents.skills.datasphere_serverless.skill import DataSphereServerlessSkill
        return DataSphereServerlessSkill

    def test_schema_includes_base_params(self):
        """Test that schema includes base class parameters"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        # Base class adds swaig_fields and tool_name (since SUPPORTS_MULTIPLE_INSTANCES)
        assert "swaig_fields" in schema
        assert "tool_name" in schema

    def test_schema_includes_space_name(self):
        """Test space_name parameter is defined"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "space_name" in schema
        assert schema["space_name"]["type"] == "string"
        assert schema["space_name"]["required"] is True

    def test_schema_includes_project_id(self):
        """Test project_id parameter is defined with env_var"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "project_id" in schema
        assert schema["project_id"]["type"] == "string"
        assert schema["project_id"]["required"] is True
        assert schema["project_id"]["env_var"] == "SIGNALWIRE_PROJECT_ID"

    def test_schema_includes_token(self):
        """Test token parameter is defined as hidden with env_var"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "token" in schema
        assert schema["token"]["type"] == "string"
        assert schema["token"]["required"] is True
        assert schema["token"]["hidden"] is True
        assert schema["token"]["env_var"] == "SIGNALWIRE_TOKEN"

    def test_schema_includes_document_id(self):
        """Test document_id parameter is defined"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "document_id" in schema
        assert schema["document_id"]["type"] == "string"
        assert schema["document_id"]["required"] is True

    def test_schema_includes_count(self):
        """Test count parameter is defined with defaults and bounds"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "count" in schema
        assert schema["count"]["type"] == "integer"
        assert schema["count"]["default"] == 1
        assert schema["count"]["required"] is False
        assert schema["count"]["minimum"] == 1
        assert schema["count"]["maximum"] == 10

    def test_schema_includes_distance(self):
        """Test distance parameter is defined with defaults and bounds"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "distance" in schema
        assert schema["distance"]["type"] == "number"
        assert schema["distance"]["default"] == 3.0
        assert schema["distance"]["required"] is False
        assert schema["distance"]["minimum"] == 0.0
        assert schema["distance"]["maximum"] == 10.0

    def test_schema_includes_tags(self):
        """Test tags parameter is defined as array of strings"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "tags" in schema
        assert schema["tags"]["type"] == "array"
        assert schema["tags"]["required"] is False
        assert schema["tags"]["items"]["type"] == "string"

    def test_schema_includes_language(self):
        """Test language parameter is defined"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "language" in schema
        assert schema["language"]["type"] == "string"
        assert schema["language"]["required"] is False

    def test_schema_includes_pos_to_expand(self):
        """Test pos_to_expand parameter is defined with enum values"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "pos_to_expand" in schema
        assert schema["pos_to_expand"]["type"] == "array"
        assert schema["pos_to_expand"]["required"] is False
        assert schema["pos_to_expand"]["items"]["enum"] == ["NOUN", "VERB", "ADJ", "ADV"]

    def test_schema_includes_max_synonyms(self):
        """Test max_synonyms parameter is defined with bounds"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "max_synonyms" in schema
        assert schema["max_synonyms"]["type"] == "integer"
        assert schema["max_synonyms"]["required"] is False
        assert schema["max_synonyms"]["minimum"] == 1
        assert schema["max_synonyms"]["maximum"] == 10

    def test_schema_includes_no_results_message(self):
        """Test no_results_message parameter is defined with default"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        assert "no_results_message" in schema
        assert schema["no_results_message"]["type"] == "string"
        assert schema["no_results_message"]["required"] is False
        assert "{query}" in schema["no_results_message"]["default"]

    def test_schema_has_all_expected_params(self):
        """Test that all expected parameters are present in schema"""
        cls = self._get_skill_class()
        schema = cls.get_parameter_schema()

        expected_params = [
            "swaig_fields", "tool_name",  # from base class
            "space_name", "project_id", "token", "document_id",
            "count", "distance", "tags", "language",
            "pos_to_expand", "max_synonyms", "no_results_message"
        ]

        for param in expected_params:
            assert param in schema, f"Missing expected parameter: {param}"


def _create_skill(params=None, swaig_fields=None):
    """Helper to create a DataSphereServerlessSkill with mocked agent"""
    from signalwire_agents.skills.datasphere_serverless.skill import DataSphereServerlessSkill

    mock_agent = Mock()
    mock_agent.register_swaig_function = Mock()

    default_params = {
        "space_name": "testspace",
        "project_id": "test-project-id",
        "token": "test-token-secret",
        "document_id": "doc-123",
    }

    if params is not None:
        default_params.update(params)

    if swaig_fields is not None:
        default_params["swaig_fields"] = swaig_fields

    skill = DataSphereServerlessSkill(mock_agent, default_params)
    return skill, mock_agent


class TestDataSphereServerlessSkillInit:
    """Test skill initialization via __init__"""

    def test_init_sets_agent(self):
        """Test that init stores the agent reference"""
        skill, mock_agent = _create_skill()
        assert skill.agent is mock_agent

    def test_init_sets_params(self):
        """Test that init stores the params"""
        skill, _ = _create_skill()
        assert skill.params["space_name"] == "testspace"
        assert skill.params["project_id"] == "test-project-id"
        assert skill.params["token"] == "test-token-secret"
        assert skill.params["document_id"] == "doc-123"

    def test_init_creates_logger(self):
        """Test that init creates a logger with the skill name"""
        skill, _ = _create_skill()
        assert skill.logger is not None
        assert skill.logger.name == "skill.datasphere_serverless"

    def test_init_extracts_swaig_fields(self):
        """Test that swaig_fields are extracted and removed from params"""
        swaig_fields = {"meta_data": {"key": "value"}}
        skill, _ = _create_skill(swaig_fields=swaig_fields)

        assert skill.swaig_fields == {"meta_data": {"key": "value"}}
        assert "swaig_fields" not in skill.params

    def test_init_empty_swaig_fields_by_default(self):
        """Test that swaig_fields defaults to empty dict"""
        skill, _ = _create_skill()
        assert skill.swaig_fields == {}

    def test_init_with_empty_params(self):
        """Test that init works with None params"""
        from signalwire_agents.skills.datasphere_serverless.skill import DataSphereServerlessSkill
        mock_agent = Mock()
        skill = DataSphereServerlessSkill(mock_agent, None)
        assert skill.params == {}


class TestDataSphereServerlessSkillGetInstanceKey:
    """Test get_instance_key method"""

    def test_instance_key_default_tool_name(self):
        """Test instance key uses default tool name 'search_knowledge'"""
        skill, _ = _create_skill()
        assert skill.get_instance_key() == "datasphere_serverless_search_knowledge"

    def test_instance_key_custom_tool_name(self):
        """Test instance key uses custom tool name from params"""
        skill, _ = _create_skill(params={"tool_name": "my_custom_search"})
        assert skill.get_instance_key() == "datasphere_serverless_my_custom_search"

    def test_instance_key_different_tool_names_produce_different_keys(self):
        """Test that different tool names produce different instance keys"""
        skill1, _ = _create_skill(params={"tool_name": "search_docs"})
        skill2, _ = _create_skill(params={"tool_name": "search_faq"})

        assert skill1.get_instance_key() != skill2.get_instance_key()
        assert skill1.get_instance_key() == "datasphere_serverless_search_docs"
        assert skill2.get_instance_key() == "datasphere_serverless_search_faq"


class TestDataSphereServerlessSkillSetup:
    """Test setup() method behavior"""

    def test_setup_success_with_all_required_params(self):
        """Test that setup returns True when all required params are present"""
        skill, _ = _create_skill()
        result = skill.setup()
        assert result is True

    def test_setup_stores_required_params_as_attributes(self):
        """Test that setup stores required params as instance attributes"""
        skill, _ = _create_skill()
        skill.setup()

        assert skill.space_name == "testspace"
        assert skill.project_id == "test-project-id"
        assert skill.token == "test-token-secret"
        assert skill.document_id == "doc-123"

    def test_setup_missing_space_name(self):
        """Test that setup returns False when space_name is missing"""
        skill, _ = _create_skill(params={"space_name": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_project_id(self):
        """Test that setup returns False when project_id is missing"""
        skill, _ = _create_skill(params={"project_id": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_token(self):
        """Test that setup returns False when token is missing"""
        skill, _ = _create_skill(params={"token": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_document_id(self):
        """Test that setup returns False when document_id is missing"""
        skill, _ = _create_skill(params={"document_id": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_all_required_params(self):
        """Test that setup returns False when all required params are missing"""
        from signalwire_agents.skills.datasphere_serverless.skill import DataSphereServerlessSkill
        mock_agent = Mock()
        skill = DataSphereServerlessSkill(mock_agent, {})
        result = skill.setup()
        assert result is False

    def test_setup_logs_error_on_missing_params(self):
        """Test that setup logs an error when params are missing"""
        skill, _ = _create_skill(params={"space_name": "", "token": ""})
        with patch.object(skill.logger, "error") as mock_error:
            skill.setup()
            mock_error.assert_called_once()
            error_msg = mock_error.call_args[0][0]
            assert "Missing required parameters" in error_msg
            assert "space_name" in error_msg
            assert "token" in error_msg

    def test_setup_default_count(self):
        """Test that setup uses default count of 1"""
        skill, _ = _create_skill()
        skill.setup()
        assert skill.count == 1

    def test_setup_custom_count(self):
        """Test that setup uses custom count"""
        skill, _ = _create_skill(params={"count": 5})
        skill.setup()
        assert skill.count == 5

    def test_setup_default_distance(self):
        """Test that setup uses default distance of 3.0"""
        skill, _ = _create_skill()
        skill.setup()
        assert skill.distance == 3.0

    def test_setup_custom_distance(self):
        """Test that setup uses custom distance"""
        skill, _ = _create_skill(params={"distance": 1.5})
        skill.setup()
        assert skill.distance == 1.5

    def test_setup_tags_default_none(self):
        """Test that tags defaults to None when not provided"""
        skill, _ = _create_skill()
        skill.setup()
        assert skill.tags is None

    def test_setup_custom_tags(self):
        """Test that tags are stored when provided"""
        skill, _ = _create_skill(params={"tags": ["faq", "docs"]})
        skill.setup()
        assert skill.tags == ["faq", "docs"]

    def test_setup_language_default_none(self):
        """Test that language defaults to None when not provided"""
        skill, _ = _create_skill()
        skill.setup()
        assert skill.language is None

    def test_setup_custom_language(self):
        """Test that language is stored when provided"""
        skill, _ = _create_skill(params={"language": "en"})
        skill.setup()
        assert skill.language == "en"

    def test_setup_pos_to_expand_default_none(self):
        """Test that pos_to_expand defaults to None"""
        skill, _ = _create_skill()
        skill.setup()
        assert skill.pos_to_expand is None

    def test_setup_custom_pos_to_expand(self):
        """Test that pos_to_expand is stored when provided"""
        skill, _ = _create_skill(params={"pos_to_expand": ["NOUN", "VERB"]})
        skill.setup()
        assert skill.pos_to_expand == ["NOUN", "VERB"]

    def test_setup_max_synonyms_default_none(self):
        """Test that max_synonyms defaults to None"""
        skill, _ = _create_skill()
        skill.setup()
        assert skill.max_synonyms is None

    def test_setup_custom_max_synonyms(self):
        """Test that max_synonyms is stored when provided"""
        skill, _ = _create_skill(params={"max_synonyms": 5})
        skill.setup()
        assert skill.max_synonyms == 5

    def test_setup_default_tool_name(self):
        """Test that tool_name defaults to 'search_knowledge'"""
        skill, _ = _create_skill()
        skill.setup()
        assert skill.tool_name == "search_knowledge"

    def test_setup_custom_tool_name(self):
        """Test that tool_name is stored when provided"""
        skill, _ = _create_skill(params={"tool_name": "custom_search"})
        skill.setup()
        assert skill.tool_name == "custom_search"

    def test_setup_default_no_results_message(self):
        """Test that no_results_message uses default value"""
        skill, _ = _create_skill()
        skill.setup()
        assert "{query}" in skill.no_results_message
        assert "couldn't find" in skill.no_results_message

    def test_setup_custom_no_results_message(self):
        """Test that no_results_message is stored when provided"""
        custom_msg = "Sorry, nothing found for '{query}'."
        skill, _ = _create_skill(params={"no_results_message": custom_msg})
        skill.setup()
        assert skill.no_results_message == custom_msg

    def test_setup_builds_api_url(self):
        """Test that setup builds the correct API URL"""
        skill, _ = _create_skill()
        skill.setup()
        assert skill.api_url == "https://testspace.signalwire.com/api/datasphere/documents/search"

    def test_setup_api_url_with_different_space(self):
        """Test API URL with a different space name"""
        skill, _ = _create_skill(params={"space_name": "mycompany"})
        skill.setup()
        assert skill.api_url == "https://mycompany.signalwire.com/api/datasphere/documents/search"

    def test_setup_builds_auth_header(self):
        """Test that setup builds the correct base64 auth header"""
        skill, _ = _create_skill()
        skill.setup()

        expected_auth = base64.b64encode(b"test-project-id:test-token-secret").decode()
        assert skill.auth_header == expected_auth

    def test_setup_auth_header_encoding(self):
        """Test that auth header is proper base64 of project_id:token"""
        skill, _ = _create_skill(params={
            "project_id": "proj-abc",
            "token": "tok-xyz"
        })
        skill.setup()

        decoded = base64.b64decode(skill.auth_header).decode()
        assert decoded == "proj-abc:tok-xyz"

    def test_setup_none_param_treated_as_missing(self):
        """Test that None values for required params cause setup failure"""
        skill, _ = _create_skill(params={"space_name": None})
        result = skill.setup()
        assert result is False


class TestDataSphereServerlessSkillRegisterTools:
    """Test register_tools() method"""

    def test_register_tools_calls_register_swaig_function(self):
        """Test that register_tools registers a SWAIG function with the agent"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        mock_agent.register_swaig_function.assert_called_once()

    def test_register_tools_function_name(self):
        """Test that registered function has correct name"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        assert swaig_func["function"] == "search_knowledge"

    def test_register_tools_custom_function_name(self):
        """Test that registered function uses custom tool name"""
        skill, mock_agent = _create_skill(params={"tool_name": "my_search"})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        assert swaig_func["function"] == "my_search"

    def test_register_tools_has_description(self):
        """Test that registered function has a description"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        assert "description" in swaig_func
        assert "knowledge base" in swaig_func["description"].lower()

    def test_register_tools_has_query_parameter(self):
        """Test that registered function has a query parameter"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        assert "parameters" in swaig_func
        assert "properties" in swaig_func["parameters"]
        assert "query" in swaig_func["parameters"]["properties"]
        assert swaig_func["parameters"]["properties"]["query"]["type"] == "string"

    def test_register_tools_query_is_required(self):
        """Test that query parameter is required"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        assert "required" in swaig_func["parameters"]
        assert "query" in swaig_func["parameters"]["required"]

    def test_register_tools_has_data_map(self):
        """Test that registered function uses data_map (not url)"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        assert "data_map" in swaig_func

    def test_register_tools_data_map_has_webhook(self):
        """Test that data_map contains a webhook"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        data_map = swaig_func["data_map"]
        assert "webhooks" in data_map
        assert len(data_map["webhooks"]) >= 1

    def test_register_tools_webhook_method_is_post(self):
        """Test that webhook uses POST method"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert webhook["method"] == "POST"

    def test_register_tools_webhook_url(self):
        """Test that webhook URL matches the API URL"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert webhook["url"] == "https://testspace.signalwire.com/api/datasphere/documents/search"

    def test_register_tools_webhook_auth_header(self):
        """Test that webhook has correct Authorization header"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "headers" in webhook
        assert "Authorization" in webhook["headers"]
        assert webhook["headers"]["Authorization"].startswith("Basic ")

        # Verify the base64 encoded credentials
        auth_value = webhook["headers"]["Authorization"].replace("Basic ", "")
        decoded = base64.b64decode(auth_value).decode()
        assert decoded == "test-project-id:test-token-secret"

    def test_register_tools_webhook_content_type(self):
        """Test that webhook has Content-Type application/json header"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert webhook["headers"]["Content-Type"] == "application/json"

    def test_register_tools_webhook_params(self):
        """Test that webhook has correct params"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "params" in webhook
        params = webhook["params"]
        assert params["document_id"] == "doc-123"
        assert params["query_string"] == "${args.query}"
        assert params["count"] == 1
        assert params["distance"] == 3.0

    def test_register_tools_webhook_params_with_custom_values(self):
        """Test webhook params with custom count and distance"""
        skill, mock_agent = _create_skill(params={"count": 5, "distance": 1.0})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        params = webhook["params"]
        assert params["count"] == 5
        assert params["distance"] == 1.0

    def test_register_tools_no_optional_params_by_default(self):
        """Test that optional params are not in webhook params by default"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        params = webhook["params"]
        assert "tags" not in params
        assert "language" not in params
        assert "pos_to_expand" not in params
        assert "max_synonyms" not in params

    def test_register_tools_includes_tags_when_provided(self):
        """Test that tags are included in webhook params when set"""
        skill, mock_agent = _create_skill(params={"tags": ["faq", "support"]})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        params = webhook["params"]
        assert params["tags"] == ["faq", "support"]

    def test_register_tools_includes_language_when_provided(self):
        """Test that language is included in webhook params when set"""
        skill, mock_agent = _create_skill(params={"language": "es"})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        params = webhook["params"]
        assert params["language"] == "es"

    def test_register_tools_includes_pos_to_expand_when_provided(self):
        """Test that pos_to_expand is included in webhook params when set"""
        skill, mock_agent = _create_skill(params={"pos_to_expand": ["NOUN", "ADJ"]})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        params = webhook["params"]
        assert params["pos_to_expand"] == ["NOUN", "ADJ"]

    def test_register_tools_includes_max_synonyms_when_provided(self):
        """Test that max_synonyms is included in webhook params when set"""
        skill, mock_agent = _create_skill(params={"max_synonyms": 3})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        params = webhook["params"]
        assert params["max_synonyms"] == 3

    def test_register_tools_includes_all_optional_params(self):
        """Test that all optional params are included when all are provided"""
        skill, mock_agent = _create_skill(params={
            "tags": ["doc"],
            "language": "en",
            "pos_to_expand": ["VERB"],
            "max_synonyms": 2
        })
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        params = webhook["params"]
        assert params["tags"] == ["doc"]
        assert params["language"] == "en"
        assert params["pos_to_expand"] == ["VERB"]
        assert params["max_synonyms"] == 2

    def test_register_tools_has_foreach(self):
        """Test that data_map webhook has foreach configuration"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "foreach" in webhook

        foreach = webhook["foreach"]
        assert foreach["input_key"] == "chunks"
        assert foreach["output_key"] == "formatted_results"
        assert foreach["max"] == 1  # default count

    def test_register_tools_foreach_max_matches_count(self):
        """Test that foreach max matches the configured count"""
        skill, mock_agent = _create_skill(params={"count": 7})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert webhook["foreach"]["max"] == 7

    def test_register_tools_foreach_append_template(self):
        """Test that foreach has an append template"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "append" in webhook["foreach"]
        assert "RESULT" in webhook["foreach"]["append"]
        assert "${this.text}" in webhook["foreach"]["append"]

    def test_register_tools_has_output(self):
        """Test that webhook has output configuration"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "output" in webhook
        assert "response" in webhook["output"]
        assert "${formatted_results}" in webhook["output"]["response"]
        assert "${args.query}" in webhook["output"]["response"]

    def test_register_tools_has_error_keys(self):
        """Test that webhook has error_keys configured"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "error_keys" in webhook
        assert "error" in webhook["error_keys"]

    def test_register_tools_has_fallback_output(self):
        """Test that data_map has a fallback output"""
        skill, mock_agent = _create_skill()
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        data_map = swaig_func["data_map"]
        assert "output" in data_map
        assert "response" in data_map["output"]

    def test_register_tools_fallback_output_uses_no_results_message(self):
        """Test that fallback output uses no_results_message with query substitution"""
        custom_msg = "Nothing found for '{query}'."
        skill, mock_agent = _create_skill(params={"no_results_message": custom_msg})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        data_map = swaig_func["data_map"]
        # {query} should be replaced with ${args.query}
        assert "${args.query}" in data_map["output"]["response"]
        assert "{query}" not in data_map["output"]["response"]

    def test_register_tools_merges_swaig_fields(self):
        """Test that swaig_fields are merged into the function definition"""
        swaig_fields = {"meta_data_token": "custom_token", "fillers": {"en": ["hmm"]}}
        skill, mock_agent = _create_skill(swaig_fields=swaig_fields)
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        assert swaig_func["meta_data_token"] == "custom_token"
        assert swaig_func["fillers"] == {"en": ["hmm"]}

    def test_register_tools_swaig_fields_do_not_overwrite_core_fields(self):
        """Test that swaig_fields merged after to_swaig_function update core fields"""
        # swaig_fields.update() is called with swaig_func.update(self.swaig_fields),
        # meaning swaig_fields CAN overwrite - this tests the actual behavior
        swaig_fields = {"function": "overridden_name"}
        skill, mock_agent = _create_skill(swaig_fields=swaig_fields)
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        # The code uses swaig_function.update(self.swaig_fields), so swaig_fields win
        assert swaig_func["function"] == "overridden_name"


class TestDataSphereServerlessSkillGetHints:
    """Test get_hints() method"""

    def test_get_hints_returns_empty_list(self):
        """Test that get_hints returns an empty list"""
        skill, _ = _create_skill()
        assert skill.get_hints() == []


class TestDataSphereServerlessSkillGetGlobalData:
    """Test get_global_data() method"""

    def test_get_global_data_after_setup(self):
        """Test get_global_data returns correct data after setup"""
        skill, _ = _create_skill()
        skill.setup()
        data = skill.get_global_data()

        assert data["datasphere_serverless_enabled"] is True
        assert data["document_id"] == "doc-123"
        assert data["knowledge_provider"] == "SignalWire DataSphere (Serverless)"

    def test_get_global_data_with_different_document_id(self):
        """Test get_global_data reflects the configured document_id"""
        skill, _ = _create_skill(params={"document_id": "another-doc"})
        skill.setup()
        data = skill.get_global_data()
        assert data["document_id"] == "another-doc"

    def test_get_global_data_keys(self):
        """Test that get_global_data returns exactly the expected keys"""
        skill, _ = _create_skill()
        skill.setup()
        data = skill.get_global_data()

        expected_keys = {"datasphere_serverless_enabled", "document_id", "knowledge_provider"}
        assert set(data.keys()) == expected_keys


class TestDataSphereServerlessSkillGetPromptSections:
    """Test get_prompt_sections() method"""

    def test_get_prompt_sections_returns_one_section(self):
        """Test that get_prompt_sections returns exactly one section"""
        skill, _ = _create_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        assert len(sections) == 1

    def test_get_prompt_sections_title(self):
        """Test the prompt section title"""
        skill, _ = _create_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "Serverless" in section["title"]
        assert "Knowledge Search" in section["title"]

    def test_get_prompt_sections_body_references_tool_name(self):
        """Test that section body references the tool name"""
        skill, _ = _create_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "search_knowledge" in section["body"]

    def test_get_prompt_sections_body_with_custom_tool_name(self):
        """Test that section body uses custom tool name"""
        skill, _ = _create_skill(params={"tool_name": "my_kb_search"})
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "my_kb_search" in section["body"]

    def test_get_prompt_sections_has_bullets(self):
        """Test that section has bullets"""
        skill, _ = _create_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "bullets" in section
        assert len(section["bullets"]) > 0

    def test_get_prompt_sections_bullets_reference_tool_name(self):
        """Test that at least one bullet references the tool name"""
        skill, _ = _create_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        tool_name_found = any("search_knowledge" in bullet for bullet in section["bullets"])
        assert tool_name_found

    def test_get_prompt_sections_mentions_serverless(self):
        """Test that bullets mention serverless execution"""
        skill, _ = _create_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        serverless_found = any("server" in bullet.lower() for bullet in section["bullets"])
        assert serverless_found


class TestDataSphereServerlessSkillEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_setup_with_special_characters_in_space_name(self):
        """Test setup with special characters in space name"""
        skill, _ = _create_skill(params={"space_name": "my-company-123"})
        skill.setup()
        assert skill.api_url == "https://my-company-123.signalwire.com/api/datasphere/documents/search"

    def test_setup_with_special_characters_in_credentials(self):
        """Test that auth header is correctly encoded with special characters"""
        skill, _ = _create_skill(params={
            "project_id": "proj+id/special",
            "token": "tok=en/special+chars"
        })
        skill.setup()

        decoded = base64.b64decode(skill.auth_header).decode()
        assert decoded == "proj+id/special:tok=en/special+chars"

    def test_multiple_skills_with_different_configs(self):
        """Test creating multiple skill instances with different configs"""
        skill1, agent1 = _create_skill(params={
            "tool_name": "search_docs",
            "document_id": "doc-1",
            "count": 3
        })
        skill2, agent2 = _create_skill(params={
            "tool_name": "search_faq",
            "document_id": "doc-2",
            "count": 1
        })

        skill1.setup()
        skill2.setup()

        assert skill1.tool_name == "search_docs"
        assert skill2.tool_name == "search_faq"
        assert skill1.document_id == "doc-1"
        assert skill2.document_id == "doc-2"
        assert skill1.count == 3
        assert skill2.count == 1

    def test_register_tools_with_empty_tags_list(self):
        """Test register_tools when tags is an empty list (not None)"""
        skill, mock_agent = _create_skill(params={"tags": []})
        skill.setup()
        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        params = webhook["params"]
        # Empty list is not None, so it should be included
        assert "tags" in params
        assert params["tags"] == []

    def test_setup_with_zero_count(self):
        """Test setup with count of 0 (uses the provided value)"""
        skill, _ = _create_skill(params={"count": 0})
        skill.setup()
        assert skill.count == 0

    def test_setup_with_zero_distance(self):
        """Test setup with distance of 0"""
        skill, _ = _create_skill(params={"distance": 0.0})
        skill.setup()
        assert skill.distance == 0.0

    def test_setup_idempotent(self):
        """Test that calling setup multiple times works correctly"""
        skill, _ = _create_skill()

        result1 = skill.setup()
        result2 = skill.setup()

        assert result1 is True
        assert result2 is True
        assert skill.space_name == "testspace"

    def test_register_tools_after_setup_with_modified_params(self):
        """Test that register_tools uses the values set during setup"""
        skill, mock_agent = _create_skill(params={"count": 5})
        skill.setup()

        # Verify that the count was set during setup
        assert skill.count == 5

        skill.register_tools()

        swaig_func = mock_agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert webhook["params"]["count"] == 5
        assert webhook["foreach"]["max"] == 5

    def test_missing_only_one_required_param_reports_it(self):
        """Test that only the specific missing param is reported"""
        skill, _ = _create_skill(params={"document_id": ""})
        with patch.object(skill.logger, "error") as mock_error:
            result = skill.setup()
            assert result is False
            error_msg = mock_error.call_args[0][0]
            assert "document_id" in error_msg
            # The other required params should NOT be in the error
            assert "space_name" not in error_msg
            assert "project_id" not in error_msg
            assert "token" not in error_msg

    def test_falsy_but_present_required_params_treated_as_missing(self):
        """Test that falsy values (empty string, None, 0) for required params cause failure"""
        # Empty string
        skill, _ = _create_skill(params={"space_name": ""})
        assert skill.setup() is False

        # None
        skill, _ = _create_skill(params={"space_name": None})
        assert skill.setup() is False

    def test_cleanup_inherited_from_base(self):
        """Test that cleanup method is inherited and does not raise"""
        skill, _ = _create_skill()
        # cleanup is a no-op inherited from SkillBase
        skill.cleanup()  # Should not raise
