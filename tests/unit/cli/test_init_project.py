"""
Unit tests for CLI init_project module.

Tests cover:
- Print/prompt utility functions
- Token masking
- Password generation
- Environment credential retrieval
- Template generation functions
- ProjectGenerator class
- run_quick configuration builder
- main() CLI entry point
"""

import pytest
import sys
import os
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

from signalwire_agents.cli.init_project import (
    Colors,
    print_step,
    print_success,
    print_warning,
    print_error,
    prompt,
    prompt_yes_no,
    prompt_select,
    prompt_multiselect,
    mask_token,
    get_env_credentials,
    generate_password,
    get_agent_template,
    get_app_template,
    get_test_template,
    get_readme_template,
    get_web_index_template,
    ProjectGenerator,
    run_quick,
    main,
    CLOUD_PLATFORMS,
    DEFAULT_REGIONS,
    TEMPLATE_GITIGNORE,
    TEMPLATE_REQUIREMENTS,
)


# =============================================================================
# Colors Class Tests
# =============================================================================

class TestColors:
    """Tests for the ANSI color constants."""

    def test_colors_has_expected_attributes(self):
        assert Colors.RED == '\033[0;31m'
        assert Colors.GREEN == '\033[0;32m'
        assert Colors.YELLOW == '\033[1;33m'
        assert Colors.BLUE == '\033[0;34m'
        assert Colors.CYAN == '\033[0;36m'
        assert Colors.BOLD == '\033[1m'
        assert Colors.DIM == '\033[2m'
        assert Colors.NC == '\033[0m'


# =============================================================================
# Print Utility Tests
# =============================================================================

class TestPrintUtilities:
    """Tests for print_step, print_success, print_warning, print_error."""

    def test_print_step(self, capsys):
        print_step("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out
        assert Colors.BLUE in captured.out

    def test_print_success(self, capsys):
        print_success("done")
        captured = capsys.readouterr()
        assert "done" in captured.out
        assert Colors.GREEN in captured.out

    def test_print_warning(self, capsys):
        print_warning("careful")
        captured = capsys.readouterr()
        assert "careful" in captured.out
        assert Colors.YELLOW in captured.out

    def test_print_error(self, capsys):
        print_error("oops")
        captured = capsys.readouterr()
        assert "oops" in captured.out
        assert Colors.RED in captured.out


# =============================================================================
# Prompt Function Tests
# =============================================================================

class TestPromptFunctions:
    """Tests for interactive prompt functions."""

    @patch('builtins.input', return_value='')
    def test_prompt_returns_default_when_empty(self, mock_input):
        result = prompt("Name", "default_val")
        assert result == "default_val"
        mock_input.assert_called_once()

    @patch('builtins.input', return_value='custom')
    def test_prompt_returns_user_input(self, mock_input):
        result = prompt("Name", "default_val")
        assert result == "custom"

    @patch('builtins.input', return_value='  spaced  ')
    def test_prompt_strips_whitespace(self, mock_input):
        result = prompt("Name", "default_val")
        assert result == "spaced"

    @patch('builtins.input', return_value='hello')
    def test_prompt_without_default(self, mock_input):
        result = prompt("Name")
        assert result == "hello"
        # Should not include bracket notation
        mock_input.assert_called_once_with("Name: ")

    @patch('builtins.input', return_value='')
    def test_prompt_yes_no_default_true(self, mock_input):
        result = prompt_yes_no("Continue?", default=True)
        assert result is True

    @patch('builtins.input', return_value='')
    def test_prompt_yes_no_default_false(self, mock_input):
        result = prompt_yes_no("Continue?", default=False)
        assert result is False

    @patch('builtins.input', return_value='y')
    def test_prompt_yes_no_explicit_yes(self, mock_input):
        result = prompt_yes_no("Continue?", default=False)
        assert result is True

    @patch('builtins.input', return_value='yes')
    def test_prompt_yes_no_explicit_yes_full(self, mock_input):
        result = prompt_yes_no("Continue?", default=False)
        assert result is True

    @patch('builtins.input', return_value='n')
    def test_prompt_yes_no_explicit_no(self, mock_input):
        result = prompt_yes_no("Continue?", default=True)
        assert result is False

    @patch('builtins.input', return_value='')
    def test_prompt_select_returns_default(self, mock_input):
        result = prompt_select("Pick:", ["A", "B", "C"], default=2)
        assert result == 2

    @patch('builtins.input', return_value='3')
    def test_prompt_select_returns_chosen(self, mock_input):
        result = prompt_select("Pick:", ["A", "B", "C"], default=1)
        assert result == 3

    @patch('builtins.input', side_effect=['bad', '0', '4', '2'])
    def test_prompt_select_rejects_invalid_then_accepts(self, mock_input):
        result = prompt_select("Pick:", ["A", "B", "C"], default=1)
        assert result == 2
        assert mock_input.call_count == 4

    @patch('builtins.input', side_effect=['1', ''])
    def test_prompt_multiselect_toggle_and_accept(self, mock_input):
        result = prompt_multiselect("Features:", ["A", "B"], [False, True])
        assert result == [True, True]

    @patch('builtins.input', side_effect=[''])
    def test_prompt_multiselect_accept_defaults(self, mock_input):
        result = prompt_multiselect("Features:", ["A", "B"], [True, False])
        assert result == [True, False]


# =============================================================================
# Mask Token Tests
# =============================================================================

class TestMaskToken:
    """Tests for the mask_token helper."""

    def test_mask_short_token(self):
        result = mask_token("abc")
        assert result == "***"

    def test_mask_exactly_10_chars(self):
        result = mask_token("abcdefghij")
        assert result == "**********"

    def test_mask_long_token(self):
        result = mask_token("abcdefghijklmnop")
        assert result == "abcd...nop"


# =============================================================================
# Get Env Credentials Tests
# =============================================================================

class TestGetEnvCredentials:
    """Tests for get_env_credentials."""

    @patch.dict(os.environ, {
        'SIGNALWIRE_SPACE_NAME': 'myspace',
        'SIGNALWIRE_PROJECT_ID': 'proj123',
        'SIGNALWIRE_TOKEN': 'tok456',
    })
    def test_returns_env_values(self):
        creds = get_env_credentials()
        assert creds['space'] == 'myspace'
        assert creds['project'] == 'proj123'
        assert creds['token'] == 'tok456'

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_empty_when_unset(self):
        # Remove any env vars that might be set
        for key in ['SIGNALWIRE_SPACE_NAME', 'SIGNALWIRE_PROJECT_ID', 'SIGNALWIRE_TOKEN']:
            os.environ.pop(key, None)
        creds = get_env_credentials()
        assert creds['space'] == ''
        assert creds['project'] == ''
        assert creds['token'] == ''


# =============================================================================
# Generate Password Tests
# =============================================================================

class TestGeneratePassword:
    """Tests for generate_password."""

    def test_default_length(self):
        pw = generate_password()
        assert len(pw) == 32

    def test_custom_length(self):
        pw = generate_password(16)
        assert len(pw) == 16

    def test_passwords_are_unique(self):
        pw1 = generate_password()
        pw2 = generate_password()
        assert pw1 != pw2


# =============================================================================
# Template Generation Tests
# =============================================================================

class TestGetAgentTemplate:
    """Tests for get_agent_template."""

    def test_basic_agent_template(self):
        features = {'example_tool': True, 'debug_webhooks': False, 'basic_auth': False}
        result = get_agent_template('basic', features)
        assert 'class MainAgent(AgentBase):' in result
        assert 'from signalwire_agents import AgentBase' in result
        assert 'SwaigFunctionResult' in result
        assert 'get_info' in result

    def test_agent_template_without_tool(self):
        features = {'example_tool': False, 'debug_webhooks': False, 'basic_auth': False}
        result = get_agent_template('basic', features)
        assert 'class MainAgent(AgentBase):' in result
        assert 'get_info' not in result
        assert 'SwaigFunctionResult' not in result

    def test_agent_template_with_all_features(self):
        features = {'example_tool': True, 'debug_webhooks': True, 'basic_auth': True}
        result = get_agent_template('full', features)
        assert 'import os' in result
        assert '_configure_debug_webhooks' in result
        assert 'SWML_BASIC_AUTH_USER' in result
        assert 'get_info' in result

    def test_agent_template_with_debug_only(self):
        features = {'example_tool': False, 'debug_webhooks': True, 'basic_auth': False}
        result = get_agent_template('basic', features)
        assert 'import os' in result
        assert '_configure_debug_webhooks' in result
        assert 'on_summary' in result


class TestGetAppTemplate:
    """Tests for get_app_template."""

    def test_basic_app_template(self):
        features = {'debug_webhooks': False, 'web_ui': False}
        result = get_app_template(features)
        assert 'from signalwire_agents import AgentServer' in result
        assert 'def main():' in result
        assert 'server.run()' in result

    def test_app_template_with_debug(self):
        features = {'debug_webhooks': True, 'web_ui': False}
        result = get_app_template(features)
        assert 'print_debug_data' in result
        assert 'print_post_prompt_data' in result
        assert '/debug' in result
        assert '/post_prompt' in result

    def test_app_template_with_web_ui(self):
        features = {'debug_webhooks': False, 'web_ui': True}
        result = get_app_template(features)
        assert 'serve_static_files' in result


class TestGetTestTemplate:
    """Tests for get_test_template."""

    def test_test_template_with_tool(self):
        result = get_test_template(has_tool=True)
        assert 'test_get_info_function' in result
        assert 'TestDirectImport' in result
        assert 'test_agent_has_tools' in result

    def test_test_template_without_tool(self):
        result = get_test_template(has_tool=False)
        assert 'TestDirectImport' not in result
        assert 'test_agent_has_tools' not in result


class TestGetReadmeTemplate:
    """Tests for get_readme_template."""

    def test_basic_readme(self):
        features = {'debug_webhooks': False, 'web_ui': False}
        result = get_readme_template("test-project", features)
        assert '# test-project' in result
        assert '/swml' in result

    def test_readme_with_debug(self):
        features = {'debug_webhooks': True, 'web_ui': False}
        result = get_readme_template("test-project", features)
        assert '/debug' in result
        assert '/post_prompt' in result


class TestGetWebIndexTemplate:
    """Tests for get_web_index_template."""

    def test_web_template_has_html(self):
        result = get_web_index_template()
        assert '<!DOCTYPE html>' in result
        assert 'SignalWire Agent' in result


# =============================================================================
# ProjectGenerator Tests
# =============================================================================

class TestProjectGenerator:
    """Tests for the ProjectGenerator class."""

    def _make_config(self, platform='local', **overrides):
        config = {
            'project_name': 'test-agent',
            'project_dir': '/tmp/test-agent',
            'platform': platform,
            'agent_type': 'basic',
            'features': {
                'debug_webhooks': False,
                'post_prompt': False,
                'web_ui': False,
                'example_tool': True,
                'tests': True,
                'basic_auth': False,
            },
            'credentials': {'space': '', 'project': '', 'token': ''},
            'create_venv': False,
            'cloud_config': {},
        }
        config.update(overrides)
        return config

    def test_constructor(self):
        config = self._make_config()
        gen = ProjectGenerator(config)
        assert gen.project_name == 'test-agent'
        assert gen.platform == 'local'
        assert gen.project_dir == Path('/tmp/test-agent')

    @patch.object(ProjectGenerator, '_generate_local', return_value=True)
    def test_generate_dispatches_to_local(self, mock_gen):
        config = self._make_config(platform='local')
        gen = ProjectGenerator(config)
        result = gen.generate()
        assert result is True
        mock_gen.assert_called_once()

    @patch.object(ProjectGenerator, '_generate_aws', return_value=True)
    def test_generate_dispatches_to_aws(self, mock_gen):
        config = self._make_config(platform='aws')
        gen = ProjectGenerator(config)
        result = gen.generate()
        assert result is True
        mock_gen.assert_called_once()

    @patch.object(ProjectGenerator, '_generate_gcp', return_value=True)
    def test_generate_dispatches_to_gcp(self, mock_gen):
        config = self._make_config(platform='gcp')
        gen = ProjectGenerator(config)
        result = gen.generate()
        assert result is True
        mock_gen.assert_called_once()

    @patch.object(ProjectGenerator, '_generate_azure', return_value=True)
    def test_generate_dispatches_to_azure(self, mock_gen):
        config = self._make_config(platform='azure')
        gen = ProjectGenerator(config)
        result = gen.generate()
        assert result is True
        mock_gen.assert_called_once()

    def test_generate_unknown_platform(self):
        config = self._make_config(platform='unknown_cloud')
        gen = ProjectGenerator(config)
        result = gen.generate()
        assert result is False

    def test_generate_catches_exception(self):
        config = self._make_config(platform='local')
        gen = ProjectGenerator(config)
        with patch.object(gen, '_generate_local', side_effect=PermissionError("denied")):
            result = gen.generate()
            assert result is False

    def test_get_template_vars(self):
        config = self._make_config(
            platform='aws',
            cloud_config={'region': 'us-west-2'},
        )
        gen = ProjectGenerator(config)
        tvars = gen._get_template_vars()
        assert tvars['agent_name'] == 'test-agent'
        assert tvars['agent_name_slug'] == 'test-agent'
        assert tvars['agent_class'] == 'TestAgentAgent'
        assert tvars['function_name'] == 'test-agent'
        assert tvars['region'] == 'us-west-2'
        assert tvars['auth_user'] == 'admin'
        assert len(tvars['auth_password']) == 16

    def test_get_template_vars_default_region(self):
        config = self._make_config(platform='aws', cloud_config={})
        gen = ProjectGenerator(config)
        tvars = gen._get_template_vars()
        assert tvars['region'] == 'us-east-1'

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.write_text')
    def test_generate_local_creates_structure(self, mock_write, mock_mkdir):
        config = self._make_config()
        gen = ProjectGenerator(config)
        result = gen._generate_local()
        assert result is True
        # Verify mkdir was called (for project dir and subdirs)
        assert mock_mkdir.call_count >= 1
        # Verify files were written
        assert mock_write.call_count >= 5


# =============================================================================
# run_quick Tests
# =============================================================================

class TestRunQuick:
    """Tests for run_quick configuration builder."""

    def test_basic_quick_mode(self):
        args = argparse.Namespace(
            platform='local', region=None, type='basic', no_venv=False
        )
        config = run_quick('myagent', args)
        assert config['project_name'] == 'myagent'
        assert config['platform'] == 'local'
        assert config['agent_type'] == 'basic'
        assert config['features']['example_tool'] is True
        assert config['features']['debug_webhooks'] is False
        assert config['create_venv'] is True

    def test_full_type_quick_mode(self):
        args = argparse.Namespace(
            platform='local', region=None, type='full', no_venv=True
        )
        config = run_quick('myagent', args)
        assert config['agent_type'] == 'full'
        assert config['features']['debug_webhooks'] is True
        assert config['features']['web_ui'] is True
        assert config['features']['basic_auth'] is True
        assert config['create_venv'] is False

    def test_aws_platform_quick_mode(self):
        args = argparse.Namespace(
            platform='aws', region='us-west-2', type='basic', no_venv=False
        )
        config = run_quick('myagent', args)
        assert config['platform'] == 'aws'
        assert config['cloud_config']['region'] == 'us-west-2'
        # Cloud platforms don't create venvs
        assert config['create_venv'] is False
        # Cloud platforms have simplified features
        assert config['features']['tests'] is False
        assert config['features']['basic_auth'] is True

    def test_aws_default_region(self):
        args = argparse.Namespace(
            platform='aws', region=None, type='basic', no_venv=False
        )
        config = run_quick('myagent', args)
        assert config['cloud_config']['region'] == 'us-east-1'

    def test_azure_includes_resource_group(self):
        args = argparse.Namespace(
            platform='azure', region='eastus', type='basic', no_venv=False
        )
        config = run_quick('myagent', args)
        assert config['cloud_config']['resource_group'] == 'myagent-rg'

    def test_project_dir_is_absolute(self):
        args = argparse.Namespace(
            platform='local', region=None, type='basic', no_venv=False
        )
        config = run_quick('myagent', args)
        assert os.path.isabs(config['project_dir'])


# =============================================================================
# main() Entry Point Tests
# =============================================================================

class TestMain:
    """Tests for the main() CLI entry point."""

    @patch('signalwire_agents.cli.init_project.ProjectGenerator')
    @patch('sys.argv', ['sw-agent-init', 'testproject', '--type', 'basic', '--no-venv'])
    def test_main_quick_mode(self, mock_gen_class):
        mock_gen = Mock()
        mock_gen.generate.return_value = True
        mock_gen_class.return_value = mock_gen

        main()

        mock_gen_class.assert_called_once()
        config = mock_gen_class.call_args[0][0]
        assert config['project_name'] == 'testproject'
        assert config['agent_type'] == 'basic'
        mock_gen.generate.assert_called_once()

    @patch('signalwire_agents.cli.init_project.ProjectGenerator')
    @patch('sys.argv', ['sw-agent-init', 'testproject', '--type', 'full', '--no-venv'])
    def test_main_quick_mode_full(self, mock_gen_class):
        mock_gen = Mock()
        mock_gen.generate.return_value = True
        mock_gen_class.return_value = mock_gen

        main()

        config = mock_gen_class.call_args[0][0]
        assert config['agent_type'] == 'full'
        assert config['features']['debug_webhooks'] is True

    @patch('signalwire_agents.cli.init_project.ProjectGenerator')
    @patch('sys.argv', ['sw-agent-init', 'testproject', '-p', 'aws', '--no-venv'])
    def test_main_aws_platform(self, mock_gen_class):
        mock_gen = Mock()
        mock_gen.generate.return_value = True
        mock_gen_class.return_value = mock_gen

        main()

        config = mock_gen_class.call_args[0][0]
        assert config['platform'] == 'aws'

    @patch('signalwire_agents.cli.init_project.ProjectGenerator')
    @patch('sys.argv', ['sw-agent-init', 'testproject', '--no-venv', '--dir', '/tmp/custom'])
    def test_main_custom_dir(self, mock_gen_class):
        mock_gen = Mock()
        mock_gen.generate.return_value = True
        mock_gen_class.return_value = mock_gen

        main()

        config = mock_gen_class.call_args[0][0]
        assert '/tmp/custom' in config['project_dir']

    @patch('signalwire_agents.cli.init_project.ProjectGenerator')
    @patch('sys.argv', ['sw-agent-init', 'testproject', '--no-venv'])
    def test_main_generate_failure_exits(self, mock_gen_class):
        mock_gen = Mock()
        mock_gen.generate.return_value = False
        mock_gen_class.return_value = mock_gen

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_cloud_platforms_has_expected_keys(self):
        assert 'local' in CLOUD_PLATFORMS
        assert 'aws' in CLOUD_PLATFORMS
        assert 'gcp' in CLOUD_PLATFORMS
        assert 'azure' in CLOUD_PLATFORMS

    def test_default_regions(self):
        assert DEFAULT_REGIONS['aws'] == 'us-east-1'
        assert DEFAULT_REGIONS['gcp'] == 'us-central1'
        assert DEFAULT_REGIONS['azure'] == 'eastus'
