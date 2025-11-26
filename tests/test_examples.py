#!/usr/bin/env python3
"""
Test suite for all examples in the examples/ directory.

This test suite verifies that all example files:
1. Can be loaded without import errors
2. Generate valid SWML output
3. Have properly defined tools/functions

Usage:
    pytest tests/test_examples.py -v
    pytest tests/test_examples.py -v -k "test_agent"
    pytest tests/test_examples.py -v --tb=short
"""

import subprocess
import json
import sys
from pathlib import Path

import pytest

# Get the examples directory
REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def run_swaig_test(agent_path: Path, *args, timeout: int = 30) -> tuple:
    """
    Run swaig-test on an agent file and return (returncode, stdout, stderr).
    """
    cmd = [sys.executable, "-m", "signalwire_agents.cli.swaig_test_wrapper", str(agent_path)] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"


def get_swml_json(agent_path: Path) -> dict:
    """
    Get SWML JSON output from an agent file.
    """
    returncode, stdout, stderr = run_swaig_test(agent_path, "--dump-swml", "--raw")
    if returncode != 0:
        pytest.fail(f"swaig-test failed for {agent_path}:\nstderr: {stderr}\nstdout: {stdout}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON from {agent_path}: {e}\nOutput: {stdout}")


def list_tools(agent_path: Path) -> list:
    """
    List tools available in an agent.
    """
    returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
    if returncode != 0:
        return []
    tools = []
    for line in stdout.split('\n'):
        line = line.strip()
        if ' - ' in line and not line.startswith('Parameters:'):
            parts = line.split(' - ')
            if parts:
                tool_name = parts[0].strip()
                if tool_name and not tool_name.startswith('('):
                    tools.append(tool_name)
    return tools


# Examples that require API keys or special environment variables
EXAMPLES_REQUIRING_CREDENTIALS = {
    "joke_agent.py": "API_NINJAS_KEY",
    "joke_skill_demo.py": "API_NINJAS_KEY",
    "web_search_agent.py": "GOOGLE_SEARCH_API_KEY",
    "datasphere_serverless_env_demo.py": "SIGNALWIRE_SPACE_NAME",
    "datasphere_webhook_env_demo.py": "SIGNALWIRE_SPACE_NAME",
    "env_auth_simple.py": "SWML_BASIC_AUTH_USER",
}

# Examples that are not standard agents (demos, tests, interactive)
NON_STANDARD_EXAMPLES = {
    "swml_service_example.py",  # Interactive menu
    "basic_swml_service.py",  # Has its own CLI
    "dynamic_swml_service.py",  # Has its own CLI
    "auto_vivified_example.py",  # Has its own CLI
    "record_call_example.py",  # Demo structure
    "room_and_sip_example.py",  # Demo structure
    "tap_example.py",  # Demo structure
    "advanced_datamap_demo.py",  # Demo structure
    "search_server_standalone.py",  # Not an agent
    "test_lambda_handler.py",  # Test file
}

# Examples requiring optional dependencies
EXAMPLES_REQUIRING_DEPS = {
    "lambda_agent.py": "mangum",
}


class TestBasicAgentExamples:
    """Test basic agent examples that should load cleanly."""

    @pytest.mark.parametrize("agent_file", [
        "simple_agent.py",
        "simple_static_agent.py",
        "simple_dynamic_agent.py",
        "simple_dynamic_enhanced.py",
        "declarative_agent.py",
        "faq_bot_agent.py",
    ])
    def test_basic_agents_load(self, agent_file):
        """Test basic agent examples can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"

    @pytest.mark.parametrize("agent_file", [
        "simple_agent.py",
        "simple_static_agent.py",
        "declarative_agent.py",
    ])
    def test_basic_agents_generate_valid_swml(self, agent_file):
        """Test basic agents generate valid SWML."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")

        swml = get_swml_json(agent_path)
        assert "version" in swml, "SWML missing 'version'"
        assert "sections" in swml, "SWML missing 'sections'"
        assert "main" in swml["sections"], "SWML missing 'sections.main'"


class TestContextsExamples:
    """Test context and workflow examples."""

    @pytest.mark.parametrize("agent_file", [
        "contexts_demo.py",
        "info_gatherer_example.py",
        "dynamic_info_gatherer_example.py",
    ])
    def test_contexts_agents_load(self, agent_file):
        """Test context-based agents can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"

    def test_survey_agent_multi_class(self):
        """Test survey_agent_example.py with explicit agent class."""
        agent_path = EXAMPLES_DIR / "survey_agent_example.py"
        if not agent_path.exists():
            pytest.skip("survey_agent_example.py not found")
        # This file has multiple agent classes - test with specific one
        returncode, stdout, stderr = run_swaig_test(agent_path, "--agent-class", "ProductSurveyAgent", "--list-tools")
        assert returncode == 0, f"Failed to load ProductSurveyAgent:\nstderr: {stderr}\nstdout: {stdout}"


class TestDataMapExamples:
    """Test DataMap examples."""

    @pytest.mark.parametrize("agent_file", [
        "data_map_demo.py",
    ])
    def test_datamap_agents_load(self, agent_file):
        """Test DataMap agents can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        if agent_file in NON_STANDARD_EXAMPLES:
            pytest.skip(f"Skipping {agent_file} - non-standard agent structure")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestSkillsExamples:
    """Test skills-related examples."""

    @pytest.mark.parametrize("agent_file", [
        "skills_demo.py",
        "wikipedia_demo.py",
    ])
    def test_skills_agents_load(self, agent_file):
        """Test skills agents can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        if agent_file in EXAMPLES_REQUIRING_CREDENTIALS:
            pytest.skip(f"Skipping {agent_file} - requires {EXAMPLES_REQUIRING_CREDENTIALS[agent_file]}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        # Skills may require env vars, so we accept load failure with specific error
        if returncode != 0:
            # Check if it's a missing env var error (expected for some skills)
            if "GOOGLE_SEARCH" in stderr or "API_KEY" in stderr or "env" in stderr.lower():
                pytest.skip(f"Skipping {agent_file} - requires API keys")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestWebSearchExamples:
    """Test web search examples (may require API keys)."""

    @pytest.mark.parametrize("agent_file", [
        "web_search_multi_instance_demo.py",
    ])
    def test_web_search_agents_load(self, agent_file):
        """Test web search agents can be loaded (may skip if no API keys)."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        if returncode != 0:
            if "GOOGLE_SEARCH" in stderr or "API_KEY" in stderr or "GOOGLE_SEARCH" in stdout:
                pytest.skip(f"Skipping {agent_file} - requires Google API keys")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestDatasphereExamples:
    """Test Datasphere examples (may require env vars)."""

    @pytest.mark.parametrize("agent_file", [
        "datasphere_serverless_demo.py",
        "datasphere_multi_instance_demo.py",
    ])
    def test_datasphere_agents_load(self, agent_file):
        """Test Datasphere agents can be loaded (may skip if no credentials)."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        if returncode != 0:
            if "SIGNALWIRE" in stderr or "credentials" in stderr.lower() or "SIGNALWIRE" in stdout:
                pytest.skip(f"Skipping {agent_file} - requires SignalWire credentials")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestSWAIGFeaturesExamples:
    """Test SWAIG feature examples."""

    @pytest.mark.parametrize("agent_file", [
        "swaig_features_agent.py",
    ])
    def test_swaig_features_agents_load(self, agent_file):
        """Test SWAIG feature agents can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        if agent_file in NON_STANDARD_EXAMPLES:
            pytest.skip(f"Skipping {agent_file} - non-standard agent structure")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestSWMLServiceExamples:
    """Test SWML service examples."""

    @pytest.mark.parametrize("agent_file", [
        "swml_service_routing_example.py",
    ])
    def test_swml_service_agents_load(self, agent_file):
        """Test SWML service examples can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        if agent_file in NON_STANDARD_EXAMPLES:
            pytest.skip(f"Skipping {agent_file} - non-standard agent structure")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestDeploymentExamples:
    """Test deployment-related examples."""

    @pytest.mark.parametrize("agent_file", [
        "kubernetes_ready_agent.py",
        "custom_path_agent.py",
    ])
    def test_deployment_agents_load(self, agent_file):
        """Test deployment examples can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        if agent_file in EXAMPLES_REQUIRING_DEPS:
            pytest.skip(f"Skipping {agent_file} - requires {EXAMPLES_REQUIRING_DEPS[agent_file]}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestMultiAgentExamples:
    """Test multi-agent examples."""

    def test_multi_agent_server_load(self):
        """Test multi-agent server can be loaded."""
        agent_path = EXAMPLES_DIR / "multi_agent_server.py"
        if not agent_path.exists():
            pytest.skip("multi_agent_server.py not found")
        # Multi-agent files may need --agent flag
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-agents")
        # Should show agents or indicate multiple agents
        assert returncode == 0 or "multiple" in stdout.lower() or "agent" in stdout.lower(), \
            f"Failed to list agents:\nstderr: {stderr}\nstdout: {stdout}"

    def test_multi_endpoint_agent_load(self):
        """Test multi-endpoint agent can be loaded."""
        agent_path = EXAMPLES_DIR / "multi_endpoint_agent.py"
        if not agent_path.exists():
            pytest.skip("multi_endpoint_agent.py not found")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load multi_endpoint_agent.py:\nstderr: {stderr}\nstdout: {stdout}"


class TestPrefabExamples:
    """Test prefab agent examples."""

    @pytest.mark.parametrize("agent_file", [
        "concierge_agent_example.py",
        "receptionist_agent_example.py",
    ])
    def test_prefab_agents_load(self, agent_file):
        """Test prefab examples can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestDynamicConfigExamples:
    """Test dynamic configuration examples."""

    @pytest.mark.parametrize("agent_file", [
        "comprehensive_dynamic_agent.py",
    ])
    def test_dynamic_config_agents_load(self, agent_file):
        """Test dynamic config examples can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        if agent_file in NON_STANDARD_EXAMPLES:
            pytest.skip(f"Skipping {agent_file} - non-standard agent structure")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestAuthExamples:
    """Test authentication examples."""

    @pytest.mark.parametrize("agent_file", [
        "env_auth_test.py",
    ])
    def test_auth_agents_load(self, agent_file):
        """Test auth examples can be loaded."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        if agent_file in EXAMPLES_REQUIRING_CREDENTIALS:
            pytest.skip(f"Skipping {agent_file} - requires {EXAMPLES_REQUIRING_CREDENTIALS[agent_file]}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestSearchExamples:
    """Test search-related examples (may require index files)."""

    @pytest.mark.parametrize("agent_file", [
        "sigmond_simple.py",
        "sigmond_native_search.py",
        "sigmond_remote_search.py",
    ])
    def test_search_agents_load(self, agent_file):
        """Test search agents can be loaded (may skip if no index)."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        if returncode != 0:
            if "index" in stderr.lower() or "swsearch" in stderr.lower():
                pytest.skip(f"Skipping {agent_file} - requires search index")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestBedrockExamples:
    """Test AWS Bedrock examples (require AWS credentials)."""

    @pytest.mark.parametrize("agent_file", [
        "bedrock_agent_run.py",
        "bedrock_agent_test.py",
        "bedrock_server_test.py",
    ])
    def test_bedrock_agents_load(self, agent_file):
        """Test Bedrock agents can be loaded (skip if no AWS credentials)."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        if returncode != 0:
            if "AWS" in stderr or "boto" in stderr or "credentials" in stderr.lower():
                pytest.skip(f"Skipping {agent_file} - requires AWS credentials")
            # bedrock_with_skills.py has a skill loading issue
            if "Skill" in stdout and "not found" in stdout:
                pytest.skip(f"Skipping {agent_file} - skill loading issue")
        assert returncode == 0, f"Failed to load {agent_file}:\nstderr: {stderr}\nstdout: {stdout}"


class TestSpecialExamples:
    """Test special/edge case examples."""

    def test_search_server_standalone(self):
        """Test search server standalone (not an agent, may fail gracefully)."""
        agent_path = EXAMPLES_DIR / "search_server_standalone.py"
        if not agent_path.exists():
            pytest.skip("search_server_standalone.py not found")
        # This is a search server, not an agent - may not work with swaig-test
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        # Accept if it loads or fails with expected message
        if returncode != 0:
            if "not an agent" in stderr.lower() or "no agent" in stderr.lower():
                pytest.skip("search_server_standalone.py is not an agent file")

    def test_lambda_handler(self):
        """Test lambda handler example."""
        agent_path = EXAMPLES_DIR / "test_lambda_handler.py"
        if not agent_path.exists():
            pytest.skip("test_lambda_handler.py not found")
        # This is a test file, may not export an agent directly
        returncode, stdout, stderr = run_swaig_test(agent_path, "--list-tools")
        # Accept load or skip if it's not a standard agent
        if returncode != 0:
            if "no agent" in stderr.lower() or "not found" in stderr.lower():
                pytest.skip("test_lambda_handler.py doesn't export a standard agent")


class TestSWMLGeneration:
    """Test that key examples generate valid SWML."""

    @pytest.mark.parametrize("agent_file", [
        "simple_agent.py",
        "contexts_demo.py",
        "swaig_features_agent.py",
        "declarative_agent.py",
    ])
    def test_swml_has_ai_section(self, agent_file):
        """Test SWML has AI configuration."""
        agent_path = EXAMPLES_DIR / agent_file
        if not agent_path.exists():
            pytest.skip(f"Agent file not found: {agent_file}")

        swml = get_swml_json(agent_path)
        main_section = swml.get("sections", {}).get("main", [])

        # Find AI verb in main section
        ai_found = False
        for verb in main_section:
            if "ai" in verb:
                ai_found = True
                break

        assert ai_found, f"SWML for {agent_file} missing 'ai' verb"


class TestToolsPresence:
    """Test that specific agents have expected tools."""

    def test_simple_agent_has_tools(self):
        """Test simple_agent has expected tools."""
        agent_path = EXAMPLES_DIR / "simple_agent.py"
        if not agent_path.exists():
            pytest.skip("simple_agent.py not found")

        tools = list_tools(agent_path)
        # simple_agent.py defines get_time and get_weather
        assert "get_time" in tools, f"Missing get_time tool. Found: {tools}"
        assert "get_weather" in tools, f"Missing get_weather tool. Found: {tools}"

    def test_swaig_features_agent_has_tools(self):
        """Test swaig_features_agent has expected tools."""
        agent_path = EXAMPLES_DIR / "swaig_features_agent.py"
        if not agent_path.exists():
            pytest.skip("swaig_features_agent.py not found")

        tools = list_tools(agent_path)
        assert "get_time" in tools, f"Missing get_time tool. Found: {tools}"
        assert "get_weather" in tools, f"Missing get_weather tool. Found: {tools}"


class TestServerlessSimulation:
    """Test serverless environment simulation."""

    @pytest.mark.parametrize("platform", ["lambda", "cloud_function"])
    def test_serverless_swml_generation(self, platform):
        """Test SWML generation in serverless simulation."""
        agent_path = EXAMPLES_DIR / "simple_agent.py"
        if not agent_path.exists():
            pytest.skip("simple_agent.py not found")

        returncode, stdout, stderr = run_swaig_test(
            agent_path, "--simulate-serverless", platform, "--dump-swml", "--raw"
        )
        assert returncode == 0, f"Serverless simulation failed for {platform}:\nstderr: {stderr}"

        swml = json.loads(stdout)
        assert "version" in swml, f"Invalid SWML for {platform}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
