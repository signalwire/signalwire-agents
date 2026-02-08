"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the SWMLTransfer skill module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from signalwire_agents.skills.swml_transfer.skill import SWMLTransferSkill
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None):
    """
    Helper to create a SWMLTransferSkill with a mocked agent.
    Provides sensible defaults for all required parameters.
    """
    default_params = {
        "transfers": {
            "/sales/i": {
                "url": "https://example.com/sales-agent",
                "message": "Transferring you to sales.",
                "return_message": "Sales transfer complete.",
                "post_process": True,
                "final": True,
            },
            "/support/i": {
                "address": "+15551234567",
                "message": "Connecting you to support.",
                "return_message": "Support call complete.",
                "post_process": False,
                "final": False,
            },
        },
    }
    if params is not None:
        default_params.update(params)

    mock_agent = Mock()
    mock_agent.define_tool = Mock()
    mock_agent.register_swaig_function = Mock()
    skill = SWMLTransferSkill(agent=mock_agent, params=default_params)
    return skill


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestSWMLTransferSkillClassAttributes:
    """Verify class-level constants and metadata."""

    def test_skill_name(self):
        assert SWMLTransferSkill.SKILL_NAME == "swml_transfer"

    def test_skill_description(self):
        assert SWMLTransferSkill.SKILL_DESCRIPTION == "Transfer calls between agents based on pattern matching"

    def test_skill_version(self):
        assert SWMLTransferSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        assert SWMLTransferSkill.REQUIRED_PACKAGES == []

    def test_required_env_vars(self):
        assert SWMLTransferSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        assert SWMLTransferSkill.SUPPORTS_MULTIPLE_INSTANCES is True


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestSWMLTransferSkillInit:
    """Tests for __init__ (inherited from SkillBase)."""

    def test_agent_is_stored(self):
        mock_agent = Mock()
        skill = SWMLTransferSkill(agent=mock_agent, params={"transfers": {}})
        assert skill.agent is mock_agent

    def test_params_stored(self):
        params = {"transfers": {"/sales/": {"url": "http://x"}}}
        skill = SWMLTransferSkill(agent=Mock(), params=params)
        assert "/sales/" in skill.params["transfers"]

    def test_params_default_to_empty_dict(self):
        skill = SWMLTransferSkill(agent=Mock())
        assert skill.params == {}

    def test_logger_created(self):
        skill = SWMLTransferSkill(agent=Mock())
        assert skill.logger is not None
        assert skill.logger.name == "skill.swml_transfer"

    def test_swaig_fields_extracted_from_params(self):
        params = {"swaig_fields": {"meta_data": {"x": 1}}, "transfers": {}}
        skill = SWMLTransferSkill(agent=Mock(), params=params)
        assert skill.swaig_fields == {"meta_data": {"x": 1}}
        assert "swaig_fields" not in skill.params


# ---------------------------------------------------------------------------
# get_instance_key
# ---------------------------------------------------------------------------

class TestGetInstanceKey:
    """Tests for get_instance_key method."""

    def test_default_instance_key(self):
        skill = _make_skill()
        skill.setup()
        assert skill.get_instance_key() == "swml_transfer_transfer_call"

    def test_custom_tool_name_instance_key(self):
        skill = _make_skill({"tool_name": "route_call"})
        skill.setup()
        assert skill.get_instance_key() == "swml_transfer_route_call"

    def test_instance_key_before_setup_uses_params(self):
        """get_instance_key reads from self.params, so it works before setup."""
        skill = _make_skill({"tool_name": "early_key"})
        assert skill.get_instance_key() == "swml_transfer_early_key"


# ---------------------------------------------------------------------------
# setup() behaviour
# ---------------------------------------------------------------------------

class TestSetup:
    """Tests for setup() validation and configuration."""

    def test_setup_returns_true_with_valid_params(self):
        skill = _make_skill()
        assert skill.setup() is True

    def test_setup_stores_tool_name_default(self):
        skill = _make_skill()
        skill.setup()
        assert skill.tool_name == "transfer_call"

    def test_setup_stores_custom_tool_name(self):
        skill = _make_skill({"tool_name": "my_transfer"})
        skill.setup()
        assert skill.tool_name == "my_transfer"

    def test_setup_stores_description_default(self):
        skill = _make_skill()
        skill.setup()
        assert skill.description == "Transfer call based on pattern matching"

    def test_setup_stores_custom_description(self):
        skill = _make_skill({"description": "Route the call"})
        skill.setup()
        assert skill.description == "Route the call"

    def test_setup_stores_parameter_name_default(self):
        skill = _make_skill()
        skill.setup()
        assert skill.parameter_name == "transfer_type"

    def test_setup_stores_custom_parameter_name(self):
        skill = _make_skill({"parameter_name": "dest"})
        skill.setup()
        assert skill.parameter_name == "dest"

    def test_setup_stores_parameter_description_default(self):
        skill = _make_skill()
        skill.setup()
        assert skill.parameter_description == "The type of transfer to perform"

    def test_setup_stores_default_message(self):
        skill = _make_skill()
        skill.setup()
        assert skill.default_message == "Please specify a valid transfer type."

    def test_setup_stores_custom_default_message(self):
        skill = _make_skill({"default_message": "Unknown transfer."})
        skill.setup()
        assert skill.default_message == "Unknown transfer."

    def test_setup_stores_default_post_process(self):
        skill = _make_skill()
        skill.setup()
        assert skill.default_post_process is False

    def test_setup_stores_custom_default_post_process(self):
        skill = _make_skill({"default_post_process": True})
        skill.setup()
        assert skill.default_post_process is True

    def test_setup_stores_required_fields_default(self):
        skill = _make_skill()
        skill.setup()
        assert skill.required_fields == {}

    def test_setup_stores_custom_required_fields(self):
        fields = {"caller_name": "The caller's name"}
        skill = _make_skill({"required_fields": fields})
        skill.setup()
        assert skill.required_fields == fields

    def test_setup_sets_defaults_on_transfer_configs(self):
        """Verify setup fills in defaults for optional config fields."""
        transfers = {
            "/billing/": {"url": "https://example.com/billing"}
        }
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        config = skill.transfers["/billing/"]
        assert config["message"] == "Transferring you now..."
        assert config["return_message"] == "The transfer is complete. How else can I help you?"
        assert config["post_process"] is True
        assert config["final"] is True

    # --- Failure cases ---

    def test_setup_fails_missing_transfers(self):
        skill = _make_skill({"transfers": None})
        assert skill.setup() is False

    def test_setup_fails_empty_transfers(self):
        """Empty dict is falsy for 'not self.params.get(param)'."""
        skill = _make_skill({"transfers": {}})
        assert skill.setup() is False

    def test_setup_fails_transfers_not_dict(self):
        skill = _make_skill({"transfers": "not_a_dict"})
        assert skill.setup() is False

    def test_setup_fails_transfer_config_not_dict(self):
        skill = _make_skill({"transfers": {"/bad/": "string_config"}})
        assert skill.setup() is False

    def test_setup_fails_transfer_missing_url_and_address(self):
        skill = _make_skill({"transfers": {"/nowhere/": {"message": "hi"}}})
        assert skill.setup() is False

    def test_setup_fails_transfer_has_both_url_and_address(self):
        skill = _make_skill({
            "transfers": {
                "/both/": {
                    "url": "https://example.com/agent",
                    "address": "+15551234567"
                }
            }
        })
        assert skill.setup() is False


# ---------------------------------------------------------------------------
# register_tools()
# ---------------------------------------------------------------------------

class TestRegisterTools:
    """Tests for register_tools() method."""

    def test_register_calls_register_swaig_function(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        skill.agent.register_swaig_function.assert_called_once()

    def test_registered_function_name(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        assert call_args["function"] == "transfer_call"

    def test_registered_function_custom_name(self):
        skill = _make_skill({"tool_name": "route_it"})
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        assert call_args["function"] == "route_it"

    def test_registered_function_description(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        assert call_args["description"] == "Transfer call based on pattern matching"

    def test_registered_function_has_transfer_type_param(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        params = call_args["parameters"]
        assert "transfer_type" in params["properties"]
        assert "transfer_type" in params.get("required", [])

    def test_registered_function_custom_parameter_name(self):
        skill = _make_skill({"parameter_name": "dest_type"})
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        params = call_args["parameters"]
        assert "dest_type" in params["properties"]

    def test_registered_function_has_expressions(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        data_map = call_args["data_map"]
        # 2 patterns + 1 fallback = 3 expressions
        assert len(data_map["expressions"]) == 3

    def test_registered_function_fallback_expression(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        expressions = call_args["data_map"]["expressions"]
        fallback = expressions[-1]
        assert fallback["pattern"] == r"/.*/"
        assert fallback["output"]["response"] == "Please specify a valid transfer type."

    def test_url_transfer_uses_swml_transfer_action(self):
        """A 'url' config should produce an expression with SWML transfer action."""
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        expressions = call_args["data_map"]["expressions"]
        # Find the sales expression (url-based)
        sales_expr = [e for e in expressions if e["pattern"] == "/sales/i"][0]
        actions = sales_expr["output"]["action"]
        # Should have a SWML action with transfer key
        assert any("SWML" in a for a in actions)
        swml_action = [a for a in actions if "SWML" in a][0]
        assert swml_action["transfer"] == "true"
        # Check the dest in the SWML
        main_section = swml_action["SWML"]["sections"]["main"]
        assert any("transfer" in step for step in main_section)

    def test_address_transfer_uses_connect_action(self):
        """An 'address' config should produce an expression with connect action."""
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        expressions = call_args["data_map"]["expressions"]
        # Find the support expression (address-based)
        support_expr = [e for e in expressions if e["pattern"] == "/support/i"][0]
        actions = support_expr["output"]["action"]
        swml_action = [a for a in actions if "SWML" in a][0]
        main_section = swml_action["SWML"]["sections"]["main"]
        connect_step = [s for s in main_section if "connect" in s][0]
        assert connect_step["connect"]["to"] == "+15551234567"

    def test_address_transfer_with_from_addr(self):
        """An address config with from_addr should include 'from' in the connect action."""
        transfers = {
            "/vip/": {
                "address": "+15559876543",
                "from_addr": "+15550001111"
            }
        }
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        expressions = call_args["data_map"]["expressions"]
        vip_expr = [e for e in expressions if e["pattern"] == "/vip/"][0]
        actions = vip_expr["output"]["action"]
        swml_action = [a for a in actions if "SWML" in a][0]
        main_section = swml_action["SWML"]["sections"]["main"]
        connect_step = [s for s in main_section if "connect" in s][0]
        assert connect_step["connect"]["from"] == "+15550001111"

    def test_address_transfer_non_final(self):
        """An address config with final=False uses 'false' in SWML transfer key."""
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        expressions = call_args["data_map"]["expressions"]
        support_expr = [e for e in expressions if e["pattern"] == "/support/i"][0]
        actions = support_expr["output"]["action"]
        swml_action = [a for a in actions if "SWML" in a][0]
        assert swml_action["transfer"] == "false"

    def test_required_fields_added_as_parameters(self):
        skill = _make_skill({"required_fields": {"caller_name": "Name of caller"}})
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        params = call_args["parameters"]
        assert "caller_name" in params["properties"]
        assert "caller_name" in params.get("required", [])

    def test_required_fields_saved_in_global_data(self):
        """When required_fields is set, expressions should contain set_global_data actions."""
        skill = _make_skill({"required_fields": {"caller_name": "Name of caller"}})
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        expressions = call_args["data_map"]["expressions"]
        # Check a non-fallback expression
        first_expr = expressions[0]
        actions = first_expr["output"]["action"]
        global_data_actions = [a for a in actions if "set_global_data" in a]
        assert len(global_data_actions) >= 1
        gd = global_data_actions[0]["set_global_data"]
        assert "call_data" in gd
        assert gd["call_data"]["caller_name"] == "${args.caller_name}"

    def test_required_fields_in_fallback_expression(self):
        """The fallback expression should also save required fields."""
        skill = _make_skill({"required_fields": {"caller_name": "Name of caller"}})
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        expressions = call_args["data_map"]["expressions"]
        fallback = expressions[-1]
        actions = fallback["output"]["action"]
        global_data_actions = [a for a in actions if "set_global_data" in a]
        assert len(global_data_actions) >= 1

    def test_register_tools_fallback_without_register_swaig_function(self):
        """When agent lacks register_swaig_function, error is logged."""
        skill = _make_skill()
        skill.setup()
        # Remove register_swaig_function
        delattr(skill.agent, 'register_swaig_function')
        # Should not raise, just logs error
        skill.register_tools()


# ---------------------------------------------------------------------------
# get_hints()
# ---------------------------------------------------------------------------

class TestGetHints:
    """Tests for get_hints() method."""

    def test_get_hints_returns_list(self):
        skill = _make_skill()
        skill.setup()
        hints = skill.get_hints()
        assert isinstance(hints, list)

    def test_get_hints_contains_common_words(self):
        skill = _make_skill()
        skill.setup()
        hints = skill.get_hints()
        assert "transfer" in hints
        assert "connect" in hints
        assert "speak to" in hints
        assert "talk to" in hints

    def test_get_hints_extracts_pattern_names(self):
        skill = _make_skill()
        skill.setup()
        hints = skill.get_hints()
        assert "sales" in hints
        assert "support" in hints

    def test_get_hints_handles_pipe_separated_patterns(self):
        transfers = {
            "/billing|accounts/i": {"url": "https://example.com/billing"}
        }
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        hints = skill.get_hints()
        assert "billing" in hints
        assert "accounts" in hints

    def test_get_hints_skips_catch_all_patterns(self):
        """Patterns starting with '.' (like '.*') should be skipped."""
        transfers = {
            "/.*/": {"url": "https://example.com/fallback"}
        }
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        hints = skill.get_hints()
        # Should NOT include the catch-all, but still has common words
        for h in hints:
            assert h != ".*"

    def test_get_hints_strips_regex_delimiters(self):
        """Leading/trailing slashes should be stripped from patterns."""
        transfers = {
            "/billing/": {"url": "https://example.com/billing"}
        }
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        hints = skill.get_hints()
        assert "billing" in hints

    def test_get_hints_strips_flags(self):
        """Trailing flags like /i should be stripped."""
        transfers = {
            "/technical/i": {"url": "https://example.com/tech"}
        }
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        hints = skill.get_hints()
        assert "technical" in hints


# ---------------------------------------------------------------------------
# get_prompt_sections()
# ---------------------------------------------------------------------------

class TestGetPromptSections:
    """Tests for get_prompt_sections() method."""

    def test_returns_list(self):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        assert isinstance(sections, list)

    def test_returns_two_sections(self):
        """Should have 'Transferring' and 'Transfer Instructions' sections."""
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        assert len(sections) == 2

    def test_transferring_section_title(self):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        assert sections[0]["title"] == "Transferring"

    def test_transferring_section_body_includes_tool_name(self):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        assert "transfer_call" in sections[0]["body"]

    def test_transferring_section_bullets(self):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        bullets = sections[0]["bullets"]
        assert len(bullets) >= 1
        # Check that bullet descriptions reference destinations
        combined = " ".join(bullets)
        assert "sales" in combined.lower() or "example.com" in combined

    def test_transfer_instructions_section_title(self):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        assert sections[1]["title"] == "Transfer Instructions"

    def test_transfer_instructions_bullets_mention_tool_name(self):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        bullets = sections[1]["bullets"]
        combined = " ".join(bullets)
        assert "transfer_call" in combined

    def test_transfer_instructions_bullets_mention_param_name(self):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        bullets = sections[1]["bullets"]
        combined = " ".join(bullets)
        assert "transfer_type" in combined

    def test_required_fields_appear_in_instructions(self):
        skill = _make_skill({"required_fields": {"caller_name": "The name of the caller"}})
        skill.setup()
        sections = skill.get_prompt_sections()
        bullets = sections[1]["bullets"]
        combined = " ".join(bullets)
        assert "caller_name" in combined
        assert "The name of the caller" in combined
        assert "call_data" in combined

    def test_empty_transfers_returns_empty(self):
        """If setup fails (empty transfers), prompt sections are not reachable.
        But if transfers was somehow set to empty dict post-setup, returns []."""
        skill = _make_skill()
        skill.setup()
        skill.transfers = {}
        sections = skill.get_prompt_sections()
        assert sections == []

    def test_catch_all_pattern_skipped_in_bullets(self):
        """Patterns starting with '.' should not appear in transfer bullets."""
        transfers = {
            "/.*/": {"url": "https://example.com/fallback"},
            "/sales/": {"url": "https://example.com/sales"},
        }
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        sections = skill.get_prompt_sections()
        bullets = sections[0]["bullets"]
        # Only the sales pattern should generate a bullet
        assert len(bullets) == 1
        assert "sales" in bullets[0]

    def test_address_destination_in_bullets(self):
        """Address-based transfers should show the address in bullets."""
        transfers = {
            "/helpdesk/": {"address": "+15559990000"}
        }
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        sections = skill.get_prompt_sections()
        bullets = sections[0]["bullets"]
        combined = " ".join(bullets)
        assert "+15559990000" in combined


# ---------------------------------------------------------------------------
# get_parameter_schema()
# ---------------------------------------------------------------------------

class TestGetParameterSchema:
    """Tests for get_parameter_schema() classmethod."""

    def test_returns_dict(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert isinstance(schema, dict)

    def test_includes_transfers_key(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "transfers" in schema

    def test_transfers_required(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert schema["transfers"]["required"] is True

    def test_includes_description_key(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "description" in schema
        assert schema["description"]["required"] is False

    def test_includes_parameter_name_key(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "parameter_name" in schema
        assert schema["parameter_name"]["default"] == "transfer_type"

    def test_includes_parameter_description_key(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "parameter_description" in schema

    def test_includes_default_message_key(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "default_message" in schema
        assert schema["default_message"]["default"] == "Please specify a valid transfer type."

    def test_includes_default_post_process_key(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "default_post_process" in schema
        assert schema["default_post_process"]["default"] is False

    def test_includes_required_fields_key(self):
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "required_fields" in schema
        assert schema["required_fields"]["type"] == "object"

    def test_inherits_swaig_fields(self):
        """Should include base class swaig_fields parameter."""
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "swaig_fields" in schema

    def test_inherits_tool_name_for_multi_instance(self):
        """Since SUPPORTS_MULTIPLE_INSTANCES is True, tool_name should be present."""
        schema = SWMLTransferSkill.get_parameter_schema()
        assert "tool_name" in schema


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case and integration tests."""

    def test_single_url_transfer(self):
        """Skill with only one URL-based transfer."""
        transfers = {"/only/": {"url": "https://example.com/only"}}
        skill = _make_skill({"transfers": transfers})
        assert skill.setup() is True
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        # 1 pattern + 1 fallback
        assert len(call_args["data_map"]["expressions"]) == 2

    def test_single_address_transfer(self):
        """Skill with only one address-based transfer."""
        transfers = {"/only/": {"address": "+15551112222"}}
        skill = _make_skill({"transfers": transfers})
        assert skill.setup() is True
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        assert len(call_args["data_map"]["expressions"]) == 2

    def test_many_transfers(self):
        """Skill with many transfer patterns."""
        transfers = {}
        for i in range(10):
            transfers[f"/dept{i}/"] = {"url": f"https://example.com/dept{i}"}
        skill = _make_skill({"transfers": transfers})
        assert skill.setup() is True
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        # 10 patterns + 1 fallback
        assert len(call_args["data_map"]["expressions"]) == 11

    def test_transfer_with_no_optional_fields(self):
        """Transfer config with only 'url' should get all defaults filled by setup."""
        transfers = {"/minimal/": {"url": "https://example.com/minimal"}}
        skill = _make_skill({"transfers": transfers})
        skill.setup()
        config = skill.transfers["/minimal/"]
        assert "message" in config
        assert "return_message" in config
        assert "post_process" in config
        assert "final" in config

    def test_expression_string_references_parameter_name(self):
        """Each expression's 'string' field should reference the parameter_name."""
        skill = _make_skill({"parameter_name": "my_param"})
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        expressions = call_args["data_map"]["expressions"]
        for expr in expressions:
            assert expr["string"] == "${args.my_param}"

    def test_multiple_required_fields(self):
        """Multiple required fields should all appear as parameters and in global data."""
        fields = {
            "caller_name": "Name of the caller",
            "account_number": "Account number",
        }
        skill = _make_skill({"required_fields": fields})
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        params = call_args["parameters"]
        assert "caller_name" in params["properties"]
        assert "account_number" in params["properties"]
        assert "caller_name" in params["required"]
        assert "account_number" in params["required"]
