"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for CLI dokku deployment module.

Tests cover:
- Print/prompt utility functions
- Password generation
- DokkuProjectGenerator class (name derivation, file generation, modes)
- CLI command handlers (init, deploy, logs, config, scale)
- Argument parsing and command dispatch via main()
- Error handling and edge cases
"""

import pytest
import sys
import json
import argparse
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from io import StringIO

from signalwire_agents.cli.dokku import (
    Colors,
    print_step,
    print_success,
    print_warning,
    print_error,
    print_header,
    prompt,
    prompt_yes_no,
    generate_password,
    DokkuProjectGenerator,
    cmd_init,
    cmd_deploy,
    cmd_logs,
    cmd_config,
    cmd_scale,
    _get_app_name,
    main,
    PROCFILE_TEMPLATE,
    RUNTIME_TEMPLATE,
    REQUIREMENTS_TEMPLATE,
    CHECKS_TEMPLATE,
    GITIGNORE_TEMPLATE,
    ENV_EXAMPLE_TEMPLATE,
    APP_TEMPLATE,
    APP_TEMPLATE_WITH_WEB,
    APP_JSON_TEMPLATE,
    DEPLOY_SCRIPT_TEMPLATE,
    README_SIMPLE_TEMPLATE,
    DEPLOY_WORKFLOW_TEMPLATE,
    PREVIEW_WORKFLOW_TEMPLATE,
    DOKKU_CONFIG_TEMPLATE,
    SERVICES_TEMPLATE,
    README_CICD_TEMPLATE,
    WEB_INDEX_TEMPLATE,
)


# =============================================================================
# Colors Class Tests
# =============================================================================

class TestColors:
    """Tests for the ANSI color constants."""

    def test_colors_has_red(self):
        assert Colors.RED == '\033[0;31m'

    def test_colors_has_green(self):
        assert Colors.GREEN == '\033[0;32m'

    def test_colors_has_yellow(self):
        assert Colors.YELLOW == '\033[1;33m'

    def test_colors_has_blue(self):
        assert Colors.BLUE == '\033[0;34m'

    def test_colors_has_cyan(self):
        assert Colors.CYAN == '\033[0;36m'

    def test_colors_has_magenta(self):
        assert Colors.MAGENTA == '\033[0;35m'

    def test_colors_has_bold(self):
        assert Colors.BOLD == '\033[1m'

    def test_colors_has_dim(self):
        assert Colors.DIM == '\033[2m'

    def test_colors_has_nc(self):
        assert Colors.NC == '\033[0m'


# =============================================================================
# Print Function Tests
# =============================================================================

class TestPrintFunctions:
    """Tests for colored print utility functions."""

    def test_print_step(self, capsys):
        print_step("Installing packages")
        captured = capsys.readouterr()
        assert "==>" in captured.out
        assert "Installing packages" in captured.out
        assert Colors.BLUE in captured.out

    def test_print_success(self, capsys):
        print_success("Done!")
        captured = capsys.readouterr()
        assert "Done!" in captured.out
        assert Colors.GREEN in captured.out

    def test_print_warning(self, capsys):
        print_warning("Watch out")
        captured = capsys.readouterr()
        assert "Watch out" in captured.out
        assert Colors.YELLOW in captured.out

    def test_print_error(self, capsys):
        print_error("Something failed")
        captured = capsys.readouterr()
        assert "Something failed" in captured.out
        assert Colors.RED in captured.out

    def test_print_header(self, capsys):
        print_header("My Header")
        captured = capsys.readouterr()
        assert "My Header" in captured.out
        assert Colors.BOLD in captured.out
        assert Colors.CYAN in captured.out


# =============================================================================
# Prompt Function Tests
# =============================================================================

class TestPrompt:
    """Tests for interactive prompt functions."""

    @patch('builtins.input', return_value='myvalue')
    def test_prompt_returns_user_input(self, mock_input):
        result = prompt("Enter name")
        assert result == 'myvalue'
        mock_input.assert_called_once_with("Enter name: ")

    @patch('builtins.input', return_value='')
    def test_prompt_returns_default_on_empty(self, mock_input):
        result = prompt("Enter name", "default-val")
        assert result == 'default-val'
        mock_input.assert_called_once_with("Enter name [default-val]: ")

    @patch('builtins.input', return_value='custom')
    def test_prompt_returns_user_input_over_default(self, mock_input):
        result = prompt("Enter name", "default-val")
        assert result == 'custom'

    @patch('builtins.input', return_value='  spaced  ')
    def test_prompt_strips_whitespace(self, mock_input):
        result = prompt("Enter name")
        assert result == 'spaced'

    @patch('builtins.input', return_value='  ')
    def test_prompt_empty_after_strip_returns_default(self, mock_input):
        result = prompt("Question", "fallback")
        assert result == 'fallback'


class TestPromptYesNo:
    """Tests for the yes/no prompt function."""

    @patch('builtins.input', return_value='')
    def test_default_true_on_empty(self, mock_input):
        result = prompt_yes_no("Continue?", default=True)
        assert result is True
        assert "Y/n" in mock_input.call_args[0][0]

    @patch('builtins.input', return_value='')
    def test_default_false_on_empty(self, mock_input):
        result = prompt_yes_no("Continue?", default=False)
        assert result is False
        assert "y/N" in mock_input.call_args[0][0]

    @patch('builtins.input', return_value='y')
    def test_accepts_y(self, mock_input):
        assert prompt_yes_no("OK?", default=False) is True

    @patch('builtins.input', return_value='yes')
    def test_accepts_yes(self, mock_input):
        assert prompt_yes_no("OK?", default=False) is True

    @patch('builtins.input', return_value='Y')
    def test_accepts_uppercase_y(self, mock_input):
        assert prompt_yes_no("OK?", default=False) is True

    @patch('builtins.input', return_value='n')
    def test_rejects_n(self, mock_input):
        assert prompt_yes_no("OK?", default=True) is False

    @patch('builtins.input', return_value='no')
    def test_rejects_no(self, mock_input):
        assert prompt_yes_no("OK?", default=True) is False

    @patch('builtins.input', return_value='maybe')
    def test_non_yes_returns_false(self, mock_input):
        assert prompt_yes_no("OK?", default=True) is False


# =============================================================================
# Password Generation Tests
# =============================================================================

class TestGeneratePassword:
    """Tests for the password generation function."""

    def test_default_length(self):
        pw = generate_password()
        assert len(pw) == 32

    def test_custom_length(self):
        pw = generate_password(length=16)
        assert len(pw) == 16

    def test_uniqueness(self):
        pw1 = generate_password()
        pw2 = generate_password()
        assert pw1 != pw2

    def test_contains_only_url_safe_chars(self):
        pw = generate_password(64)
        # token_urlsafe uses A-Z, a-z, 0-9, -, _
        for ch in pw:
            assert ch.isalnum() or ch in ('-', '_'), f"Unexpected char: {ch}"


# =============================================================================
# DokkuProjectGenerator Tests
# =============================================================================

class TestDokkuProjectGeneratorInit:
    """Tests for DokkuProjectGenerator initialization and name derivation."""

    def test_basic_name_derivation(self):
        gen = DokkuProjectGenerator("my-agent", {})
        assert gen.app_name == "my-agent"
        assert gen.agent_slug == "my-agent"
        assert gen.agent_class == "MyAgentAgent"

    def test_underscore_name_derivation(self):
        gen = DokkuProjectGenerator("my_cool_agent", {})
        assert gen.agent_slug == "my-cool-agent"
        assert gen.agent_class == "MyCoolAgentAgent"

    def test_space_in_name(self):
        gen = DokkuProjectGenerator("My Agent", {})
        assert gen.agent_slug == "my-agent"
        assert gen.agent_class == "MyAgentAgent"

    def test_single_word_name(self):
        gen = DokkuProjectGenerator("bot", {})
        assert gen.agent_slug == "bot"
        assert gen.agent_class == "BotAgent"

    def test_default_project_dir(self):
        gen = DokkuProjectGenerator("myapp", {})
        assert gen.project_dir == Path("./myapp")

    def test_custom_project_dir(self):
        gen = DokkuProjectGenerator("myapp", {'project_dir': '/tmp/custom'})
        assert str(gen.project_dir) == "/tmp/custom"


class TestDokkuProjectGeneratorGenerate:
    """Tests for the generate() method and file writing."""

    @patch.object(DokkuProjectGenerator, '_write_cicd_files')
    @patch.object(DokkuProjectGenerator, '_write_simple_files')
    @patch.object(DokkuProjectGenerator, '_write_core_files')
    @patch('signalwire_agents.cli.dokku.print_success')
    def test_generate_simple_mode(self, mock_ps, mock_core, mock_simple, mock_cicd, tmp_path):
        gen = DokkuProjectGenerator("testapp", {'project_dir': str(tmp_path / 'out')})
        result = gen.generate()
        assert result is True
        mock_core.assert_called_once()
        mock_simple.assert_called_once()
        mock_cicd.assert_not_called()

    @patch.object(DokkuProjectGenerator, '_write_cicd_files')
    @patch.object(DokkuProjectGenerator, '_write_simple_files')
    @patch.object(DokkuProjectGenerator, '_write_core_files')
    @patch('signalwire_agents.cli.dokku.print_success')
    def test_generate_cicd_mode(self, mock_ps, mock_core, mock_simple, mock_cicd, tmp_path):
        gen = DokkuProjectGenerator("testapp", {
            'project_dir': str(tmp_path / 'out'),
            'cicd': True
        })
        result = gen.generate()
        assert result is True
        mock_core.assert_called_once()
        mock_cicd.assert_called_once()
        mock_simple.assert_not_called()

    @patch.object(DokkuProjectGenerator, '_write_core_files', side_effect=OSError("disk full"))
    @patch('signalwire_agents.cli.dokku.print_error')
    @patch('signalwire_agents.cli.dokku.print_success')
    def test_generate_handles_exception(self, mock_ps, mock_pe, mock_core, tmp_path):
        gen = DokkuProjectGenerator("testapp", {'project_dir': str(tmp_path / 'out')})
        result = gen.generate()
        assert result is False
        mock_pe.assert_called_once()
        assert "Failed to generate project" in mock_pe.call_args[0][0]


class TestDokkuProjectGeneratorWriteFile:
    """Tests for the _write_file helper."""

    def test_write_file_creates_file(self, tmp_path):
        gen = DokkuProjectGenerator("testapp", {'project_dir': str(tmp_path)})
        gen._write_file('hello.txt', 'Hello World')
        assert (tmp_path / 'hello.txt').exists()
        assert (tmp_path / 'hello.txt').read_text() == 'Hello World'

    def test_write_file_creates_nested_dirs(self, tmp_path):
        gen = DokkuProjectGenerator("testapp", {'project_dir': str(tmp_path)})
        gen._write_file('a/b/c.txt', 'nested')
        assert (tmp_path / 'a' / 'b' / 'c.txt').exists()

    def test_write_file_executable(self, tmp_path):
        gen = DokkuProjectGenerator("testapp", {'project_dir': str(tmp_path)})
        gen._write_file('script.sh', '#!/bin/bash', executable=True)
        mode = (tmp_path / 'script.sh').stat().st_mode
        assert mode & 0o755 == 0o755


class TestDokkuProjectGeneratorCoreFIles:
    """Tests that _write_core_files creates all expected files."""

    def test_core_files_without_web(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {'project_dir': str(tmp_path)})
        gen._write_core_files()
        assert (tmp_path / 'Procfile').exists()
        assert (tmp_path / 'runtime.txt').exists()
        assert (tmp_path / 'requirements.txt').exists()
        assert (tmp_path / 'CHECKS').exists()
        assert (tmp_path / '.gitignore').exists()
        assert (tmp_path / '.env.example').exists()
        assert (tmp_path / 'app.json').exists()
        assert (tmp_path / 'app.py').exists()
        # Standard template used (not web)
        content = (tmp_path / 'app.py').read_text()
        assert 'AgentBase' in content
        assert 'AgentServer' not in content

    def test_core_files_with_web(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {
            'project_dir': str(tmp_path),
            'web': True
        })
        gen._write_core_files()
        content = (tmp_path / 'app.py').read_text()
        assert 'AgentServer' in content
        assert (tmp_path / 'web' / 'index.html').exists()

    def test_procfile_content(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {'project_dir': str(tmp_path)})
        gen._write_core_files()
        content = (tmp_path / 'Procfile').read_text()
        assert 'gunicorn' in content
        assert 'uvicorn' in content

    def test_runtime_content(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {'project_dir': str(tmp_path)})
        gen._write_core_files()
        content = (tmp_path / 'runtime.txt').read_text()
        assert 'python-3.11' in content

    def test_env_example_contains_app_name(self, tmp_path):
        gen = DokkuProjectGenerator("my-cool-app", {'project_dir': str(tmp_path)})
        gen._write_core_files()
        content = (tmp_path / '.env.example').read_text()
        assert 'my-cool-app' in content

    def test_app_json_contains_app_name(self, tmp_path):
        gen = DokkuProjectGenerator("testbot", {'project_dir': str(tmp_path)})
        gen._write_core_files()
        content = (tmp_path / 'app.json').read_text()
        data = json.loads(content)
        assert data['name'] == 'testbot'

    def test_app_py_uses_correct_class_name(self, tmp_path):
        gen = DokkuProjectGenerator("my-agent", {'project_dir': str(tmp_path)})
        gen._write_core_files()
        content = (tmp_path / 'app.py').read_text()
        assert 'class MyAgentAgent' in content
        assert 'name="my-agent"' in content


class TestDokkuProjectGeneratorSimpleFiles:
    """Tests for _write_simple_files."""

    def test_simple_files_created(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {
            'project_dir': str(tmp_path),
            'dokku_host': 'dokku.example.com',
            'route': 'swaig'
        })
        gen._write_simple_files()
        assert (tmp_path / 'deploy.sh').exists()
        assert (tmp_path / 'README.md').exists()

    def test_deploy_script_is_executable(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {
            'project_dir': str(tmp_path),
            'dokku_host': 'dokku.example.com',
            'route': 'swaig'
        })
        gen._write_simple_files()
        mode = (tmp_path / 'deploy.sh').stat().st_mode
        assert mode & 0o755 == 0o755

    def test_deploy_script_contains_host(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {
            'project_dir': str(tmp_path),
            'dokku_host': 'dokku.myhost.com',
            'route': 'swaig'
        })
        gen._write_simple_files()
        content = (tmp_path / 'deploy.sh').read_text()
        assert 'dokku.myhost.com' in content

    def test_readme_contains_app_name(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {
            'project_dir': str(tmp_path),
            'dokku_host': 'dokku.example.com',
            'route': 'swaig'
        })
        gen._write_simple_files()
        content = (tmp_path / 'README.md').read_text()
        assert 'myapp' in content

    def test_default_dokku_host(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {
            'project_dir': str(tmp_path),
        })
        gen._write_simple_files()
        content = (tmp_path / 'deploy.sh').read_text()
        assert 'dokku.yourdomain.com' in content


class TestDokkuProjectGeneratorCicdFiles:
    """Tests for _write_cicd_files."""

    def test_cicd_files_created(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {'project_dir': str(tmp_path)})
        gen._write_cicd_files()
        assert (tmp_path / '.github' / 'workflows' / 'deploy.yml').exists()
        assert (tmp_path / '.github' / 'workflows' / 'preview.yml').exists()
        assert (tmp_path / '.dokku' / 'config.yml').exists()
        assert (tmp_path / '.dokku' / 'services.yml').exists()
        assert (tmp_path / 'README.md').exists()

    def test_deploy_workflow_content(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {'project_dir': str(tmp_path)})
        gen._write_cicd_files()
        content = (tmp_path / '.github' / 'workflows' / 'deploy.yml').read_text()
        assert 'Deploy' in content
        assert 'dokku-deploy-system' in content

    def test_preview_workflow_content(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {'project_dir': str(tmp_path)})
        gen._write_cicd_files()
        content = (tmp_path / '.github' / 'workflows' / 'preview.yml').read_text()
        assert 'Preview' in content
        assert 'pull_request' in content

    def test_config_yml_content(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {'project_dir': str(tmp_path)})
        gen._write_cicd_files()
        content = (tmp_path / '.dokku' / 'config.yml').read_text()
        assert 'resources:' in content
        assert 'healthcheck:' in content

    def test_services_yml_content(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {'project_dir': str(tmp_path)})
        gen._write_cicd_files()
        content = (tmp_path / '.dokku' / 'services.yml').read_text()
        assert 'postgres:' in content
        assert 'redis:' in content

    def test_cicd_readme_contains_app_name(self, tmp_path):
        gen = DokkuProjectGenerator("superbot", {'project_dir': str(tmp_path)})
        gen._write_cicd_files()
        content = (tmp_path / 'README.md').read_text()
        assert 'superbot' in content


class TestDokkuProjectGeneratorWebFiles:
    """Tests for _write_web_files."""

    def test_web_dir_created(self, tmp_path):
        gen = DokkuProjectGenerator("myapp", {
            'project_dir': str(tmp_path),
            'route': 'swaig'
        })
        gen._write_web_files()
        assert (tmp_path / 'web').is_dir()
        assert (tmp_path / 'web' / 'index.html').exists()

    def test_web_index_html_contains_agent_name(self, tmp_path):
        gen = DokkuProjectGenerator("Cool Bot", {
            'project_dir': str(tmp_path),
            'route': 'swaig'
        })
        gen._write_web_files()
        content = (tmp_path / 'web' / 'index.html').read_text()
        assert 'Cool Bot' in content


# =============================================================================
# Full Generate Integration Tests (using tmp_path)
# =============================================================================

class TestDokkuProjectGeneratorFullGenerate:
    """Integration-level tests for full project generation with tmp_path."""

    def test_full_simple_generate(self, tmp_path):
        out = tmp_path / "proj"
        gen = DokkuProjectGenerator("test-agent", {
            'project_dir': str(out),
            'dokku_host': 'dokku.test.com',
            'route': 'swaig',
        })
        result = gen.generate()
        assert result is True
        # Core files
        assert (out / 'Procfile').exists()
        assert (out / 'app.py').exists()
        assert (out / 'requirements.txt').exists()
        # Simple files
        assert (out / 'deploy.sh').exists()
        assert (out / 'README.md').exists()
        # No cicd files
        assert not (out / '.github').exists()
        assert not (out / '.dokku').exists()

    def test_full_cicd_generate(self, tmp_path):
        out = tmp_path / "proj"
        gen = DokkuProjectGenerator("test-agent", {
            'project_dir': str(out),
            'cicd': True,
        })
        result = gen.generate()
        assert result is True
        assert (out / '.github' / 'workflows' / 'deploy.yml').exists()
        assert (out / '.dokku' / 'config.yml').exists()
        # No simple deploy.sh
        assert not (out / 'deploy.sh').exists()

    def test_full_generate_with_web(self, tmp_path):
        out = tmp_path / "proj"
        gen = DokkuProjectGenerator("web-agent", {
            'project_dir': str(out),
            'web': True,
        })
        result = gen.generate()
        assert result is True
        assert (out / 'web' / 'index.html').exists()
        content = (out / 'app.py').read_text()
        assert 'AgentServer' in content


# =============================================================================
# cmd_init Tests
# =============================================================================

class TestCmdInit:
    """Tests for the cmd_init CLI command handler."""

    def _make_args(self, name='testapp', cicd=False, web=False, host=None,
                   dir_val=None, force=False):
        args = argparse.Namespace()
        args.name = name
        args.cicd = cicd
        args.web = web
        args.host = host
        args.dir = dir_val
        args.force = force
        return args

    @patch.object(DokkuProjectGenerator, 'generate', return_value=True)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_simple_with_host(self, mock_path_cls, mock_gen):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args(host='dokku.example.com')
        result = cmd_init(args)
        assert result == 0
        mock_gen.assert_called_once()

    @patch.object(DokkuProjectGenerator, 'generate', return_value=True)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_cicd_mode(self, mock_path_cls, mock_gen):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args(cicd=True, host='dokku.example.com')
        result = cmd_init(args)
        assert result == 0

    @patch.object(DokkuProjectGenerator, 'generate', return_value=False)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_generation_failure_returns_1(self, mock_path_cls, mock_gen):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args(host='dokku.example.com')
        result = cmd_init(args)
        assert result == 1

    @patch('signalwire_agents.cli.dokku.shutil')
    @patch.object(DokkuProjectGenerator, 'generate', return_value=True)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_force_overwrites_existing_dir(self, mock_path_cls, mock_gen, mock_shutil):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args(host='dokku.example.com', force=True)
        result = cmd_init(args)
        assert result == 0
        mock_shutil.rmtree.assert_called_once_with(mock_path_instance)

    @patch('signalwire_agents.cli.dokku.prompt_yes_no', return_value=False)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_existing_dir_no_force_aborts(self, mock_path_cls, mock_prompt):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args(host='dokku.example.com')
        result = cmd_init(args)
        assert result == 1

    @patch('signalwire_agents.cli.dokku.prompt_yes_no', side_effect=[False, True])
    @patch('signalwire_agents.cli.dokku.prompt', return_value='dokku.example.com')
    @patch.object(DokkuProjectGenerator, 'generate', return_value=True)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_interactive_mode_simple(self, mock_path_cls, mock_gen,
                                          mock_prompt, mock_yes_no):
        """When no --host and no --cicd, enters interactive mode."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args()  # no host, no cicd
        result = cmd_init(args)
        assert result == 0
        # Should have prompted for cicd (False) and then web
        assert mock_yes_no.call_count == 2

    @patch('signalwire_agents.cli.dokku.prompt_yes_no', side_effect=[True, True])
    @patch.object(DokkuProjectGenerator, 'generate', return_value=True)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_interactive_cicd_mode(self, mock_path_cls, mock_gen, mock_yes_no):
        """When user chooses cicd in interactive mode."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args()  # no host, no cicd
        result = cmd_init(args)
        assert result == 0

    @patch.object(DokkuProjectGenerator, 'generate', return_value=True)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_with_web_flag(self, mock_path_cls, mock_gen):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args(host='dokku.example.com', web=True)
        result = cmd_init(args)
        assert result == 0

    @patch.object(DokkuProjectGenerator, 'generate', return_value=True)
    @patch('signalwire_agents.cli.dokku.Path')
    def test_init_custom_dir(self, mock_path_cls, mock_gen):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args(host='dokku.example.com', dir_val='/tmp/custom')
        result = cmd_init(args)
        assert result == 0


# =============================================================================
# cmd_deploy Tests
# =============================================================================

class TestCmdDeploy:
    """Tests for the cmd_deploy CLI command handler."""

    def _make_args(self, app=None, host=None):
        args = argparse.Namespace()
        args.app = app
        args.host = host
        return args

    @patch('signalwire_agents.cli.dokku.Path')
    def test_deploy_no_procfile_returns_error(self, mock_path_cls):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        args = self._make_args()
        result = cmd_deploy(args)
        assert result == 1

    @patch('signalwire_agents.cli.dokku.subprocess')
    @patch('signalwire_agents.cli.dokku.Path')
    def test_deploy_with_app_name_and_host(self, mock_path_cls, mock_subprocess):
        # Procfile exists, .git exists
        path_instances = {}

        def path_side_effect(p):
            if p not in path_instances:
                m = MagicMock()
                path_instances[p] = m
                # Procfile exists
                if p == 'Procfile':
                    m.exists.return_value = True
                elif p == 'app.json':
                    m.exists.return_value = False
                elif p == '.git':
                    m.exists.return_value = True
                else:
                    m.exists.return_value = False
            return path_instances[p]

        mock_path_cls.side_effect = path_side_effect

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        args = self._make_args(app='myapp', host='dokku.example.com')
        result = cmd_deploy(args)
        assert result == 0

    @patch('signalwire_agents.cli.dokku.subprocess')
    @patch('signalwire_agents.cli.dokku.Path')
    def test_deploy_git_push_failure(self, mock_path_cls, mock_subprocess):
        path_instances = {}

        def path_side_effect(p):
            if p not in path_instances:
                m = MagicMock()
                path_instances[p] = m
                if p == 'Procfile':
                    m.exists.return_value = True
                elif p == 'app.json':
                    m.exists.return_value = False
                elif p == '.git':
                    m.exists.return_value = True
                else:
                    m.exists.return_value = False
            return path_instances[p]

        mock_path_cls.side_effect = path_side_effect

        # Make the git push fail (last subprocess.run call)
        results = [
            MagicMock(returncode=0),  # apps:create
            MagicMock(returncode=0),  # remote remove
            MagicMock(returncode=0),  # remote add
            MagicMock(returncode=1),  # git push fails
        ]
        mock_subprocess.run.side_effect = results

        args = self._make_args(app='myapp', host='dokku.example.com')
        result = cmd_deploy(args)
        assert result == 1

    @patch('signalwire_agents.cli.dokku.subprocess')
    @patch('signalwire_agents.cli.dokku.Path')
    def test_deploy_initializes_git_if_needed(self, mock_path_cls, mock_subprocess):
        path_instances = {}

        def path_side_effect(p):
            if p not in path_instances:
                m = MagicMock()
                path_instances[p] = m
                if p == 'Procfile':
                    m.exists.return_value = True
                elif p == 'app.json':
                    m.exists.return_value = False
                elif p == '.git':
                    m.exists.return_value = False  # No git
                else:
                    m.exists.return_value = False
            return path_instances[p]

        mock_path_cls.side_effect = path_side_effect

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        args = self._make_args(app='myapp', host='dokku.example.com')
        result = cmd_deploy(args)
        assert result == 0

        # Check that git init was called
        calls = mock_subprocess.run.call_args_list
        git_init_calls = [c for c in calls if c[0][0] == ['git', 'init']]
        assert len(git_init_calls) == 1

    @patch('signalwire_agents.cli.dokku.prompt', side_effect=['myapp', 'dokku.example.com'])
    @patch('signalwire_agents.cli.dokku.subprocess')
    @patch('signalwire_agents.cli.dokku.Path')
    def test_deploy_prompts_for_missing_info(self, mock_path_cls, mock_subprocess, mock_prompt):
        path_instances = {}

        def path_side_effect(p):
            if p not in path_instances:
                m = MagicMock()
                path_instances[p] = m
                if p == 'Procfile':
                    m.exists.return_value = True
                elif p == 'app.json':
                    m.exists.return_value = False
                elif p == '.git':
                    m.exists.return_value = True
                else:
                    m.exists.return_value = False
            return path_instances[p]

        mock_path_cls.side_effect = path_side_effect
        mock_subprocess.run.return_value = MagicMock(returncode=0)

        args = self._make_args()  # no app, no host
        result = cmd_deploy(args)
        assert result == 0
        assert mock_prompt.call_count == 2

    @patch('signalwire_agents.cli.dokku.subprocess')
    @patch('builtins.open', mock_open(read_data='{"name": "from-json"}'))
    @patch('signalwire_agents.cli.dokku.Path')
    def test_deploy_reads_app_name_from_app_json(self, mock_path_cls, mock_subprocess):
        path_instances = {}

        def path_side_effect(p):
            if p not in path_instances:
                m = MagicMock()
                path_instances[p] = m
                if p == 'Procfile':
                    m.exists.return_value = True
                elif p == 'app.json':
                    m.exists.return_value = True
                elif p == '.git':
                    m.exists.return_value = True
                else:
                    m.exists.return_value = False
            return path_instances[p]

        mock_path_cls.side_effect = path_side_effect
        mock_subprocess.run.return_value = MagicMock(returncode=0)

        args = self._make_args(host='dokku.example.com')  # no app, but app.json exists
        result = cmd_deploy(args)
        assert result == 0


# =============================================================================
# cmd_logs Tests
# =============================================================================

class TestCmdLogs:
    """Tests for the cmd_logs CLI command handler."""

    def _make_args(self, app=None, host=None, tail=False, num=None):
        args = argparse.Namespace()
        args.app = app
        args.host = host
        args.tail = tail
        args.num = num
        return args

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_logs_basic(self, mock_subprocess):
        args = self._make_args(app='myapp', host='dokku.example.com')
        result = cmd_logs(args)
        assert result == 0
        mock_subprocess.run.assert_called_once()
        cmd = mock_subprocess.run.call_args[0][0]
        assert cmd == ['ssh', 'dokku@dokku.example.com', 'logs', 'myapp']

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_logs_with_tail(self, mock_subprocess):
        args = self._make_args(app='myapp', host='dokku.example.com', tail=True)
        result = cmd_logs(args)
        cmd = mock_subprocess.run.call_args[0][0]
        assert '-t' in cmd

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_logs_with_num(self, mock_subprocess):
        args = self._make_args(app='myapp', host='dokku.example.com', num=50)
        result = cmd_logs(args)
        cmd = mock_subprocess.run.call_args[0][0]
        assert '--num' in cmd
        assert '50' in cmd

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_logs_with_tail_and_num(self, mock_subprocess):
        args = self._make_args(app='myapp', host='dokku.example.com', tail=True, num=100)
        result = cmd_logs(args)
        cmd = mock_subprocess.run.call_args[0][0]
        assert '-t' in cmd
        assert '--num' in cmd
        assert '100' in cmd

    @patch('signalwire_agents.cli.dokku._get_app_name', return_value='fromjson')
    @patch('signalwire_agents.cli.dokku.prompt', return_value='dokku.example.com')
    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_logs_prompts_for_missing_info(self, mock_subprocess, mock_prompt, mock_get_name):
        args = self._make_args()  # no app, no host
        result = cmd_logs(args)
        assert result == 0
        mock_get_name.assert_called_once()
        mock_prompt.assert_called_once()


# =============================================================================
# cmd_config Tests
# =============================================================================

class TestCmdConfig:
    """Tests for the cmd_config CLI command handler."""

    def _make_args(self, action='show', vars_list=None, app=None, host=None):
        args = argparse.Namespace()
        args.config_action = action
        args.vars = vars_list or []
        args.app = app
        args.host = host
        return args

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_config_show(self, mock_subprocess):
        args = self._make_args(action='show', app='myapp', host='dokku.example.com')
        result = cmd_config(args)
        assert result == 0
        cmd = mock_subprocess.run.call_args[0][0]
        assert 'config:show' in cmd
        assert 'myapp' in cmd

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_config_set(self, mock_subprocess):
        args = self._make_args(
            action='set',
            vars_list=['KEY=value', 'OTHER=thing'],
            app='myapp',
            host='dokku.example.com'
        )
        result = cmd_config(args)
        assert result == 0
        cmd = mock_subprocess.run.call_args[0][0]
        assert 'config:set' in cmd
        assert 'KEY=value' in cmd
        assert 'OTHER=thing' in cmd

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_config_unset(self, mock_subprocess):
        args = self._make_args(
            action='unset',
            vars_list=['KEY'],
            app='myapp',
            host='dokku.example.com'
        )
        result = cmd_config(args)
        assert result == 0
        cmd = mock_subprocess.run.call_args[0][0]
        assert 'config:unset' in cmd
        assert 'KEY' in cmd

    def test_config_set_no_vars_returns_error(self):
        args = self._make_args(action='set', app='myapp', host='dokku.example.com')
        result = cmd_config(args)
        assert result == 1

    def test_config_unset_no_vars_returns_error(self):
        args = self._make_args(action='unset', app='myapp', host='dokku.example.com')
        result = cmd_config(args)
        assert result == 1

    @patch('signalwire_agents.cli.dokku._get_app_name', return_value='fromjson')
    @patch('signalwire_agents.cli.dokku.prompt', return_value='dokku.example.com')
    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_config_prompts_for_missing_info(self, mock_subprocess, mock_prompt, mock_get_name):
        args = self._make_args(action='show')  # no app, no host
        result = cmd_config(args)
        assert result == 0
        mock_get_name.assert_called_once()
        mock_prompt.assert_called_once()


# =============================================================================
# cmd_scale Tests
# =============================================================================

class TestCmdScale:
    """Tests for the cmd_scale CLI command handler."""

    def _make_args(self, scale_args=None, app=None, host=None):
        args = argparse.Namespace()
        args.scale_args = scale_args or []
        args.app = app
        args.host = host
        return args

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_scale_show_current(self, mock_subprocess):
        args = self._make_args(app='myapp', host='dokku.example.com')
        result = cmd_scale(args)
        assert result == 0
        cmd = mock_subprocess.run.call_args[0][0]
        assert 'ps:scale' in cmd
        assert 'myapp' in cmd
        # No extra args for showing
        assert len(cmd) == 4  # ssh, dokku@host, ps:scale, myapp

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_scale_set(self, mock_subprocess):
        args = self._make_args(
            scale_args=['web=2'],
            app='myapp',
            host='dokku.example.com'
        )
        result = cmd_scale(args)
        assert result == 0
        cmd = mock_subprocess.run.call_args[0][0]
        assert 'ps:scale' in cmd
        assert 'web=2' in cmd

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_scale_set_multiple(self, mock_subprocess):
        args = self._make_args(
            scale_args=['web=2', 'worker=3'],
            app='myapp',
            host='dokku.example.com'
        )
        result = cmd_scale(args)
        assert result == 0
        cmd = mock_subprocess.run.call_args[0][0]
        assert 'web=2' in cmd
        assert 'worker=3' in cmd

    @patch('signalwire_agents.cli.dokku._get_app_name', return_value='fromjson')
    @patch('signalwire_agents.cli.dokku.prompt', return_value='dokku.example.com')
    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_scale_prompts_for_missing_info(self, mock_subprocess, mock_prompt, mock_get_name):
        args = self._make_args()  # no app, no host
        result = cmd_scale(args)
        assert result == 0
        mock_get_name.assert_called_once()
        mock_prompt.assert_called_once()


# =============================================================================
# _get_app_name Tests
# =============================================================================

class TestGetAppName:
    """Tests for the _get_app_name helper function."""

    @patch('builtins.open', mock_open(read_data='{"name": "json-app"}'))
    @patch('signalwire_agents.cli.dokku.Path')
    def test_reads_from_app_json(self, mock_path_cls):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        result = _get_app_name()
        assert result == 'json-app'

    @patch('signalwire_agents.cli.dokku.prompt', return_value='prompted-app')
    @patch('signalwire_agents.cli.dokku.Path')
    def test_prompts_when_no_app_json(self, mock_path_cls, mock_prompt):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.return_value = mock_path_instance

        result = _get_app_name()
        assert result == 'prompted-app'

    @patch('signalwire_agents.cli.dokku.prompt', return_value='fallback')
    @patch('builtins.open', side_effect=json.JSONDecodeError("err", "doc", 0))
    @patch('signalwire_agents.cli.dokku.Path')
    def test_prompts_on_invalid_json(self, mock_path_cls, mock_open_fn, mock_prompt):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        result = _get_app_name()
        assert result == 'fallback'

    @patch('builtins.open', mock_open(read_data='{}'))
    @patch('signalwire_agents.cli.dokku.Path')
    def test_returns_empty_string_when_name_missing(self, mock_path_cls):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        result = _get_app_name()
        assert result == ''


# =============================================================================
# main() and Argument Parsing Tests
# =============================================================================

class TestMain:
    """Tests for the main() entry point and argument parsing."""

    @patch('signalwire_agents.cli.dokku.cmd_init', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'init', 'myapp', '--host', 'dokku.test.com'])
    def test_main_init_command(self, mock_cmd_init):
        result = main()
        assert result == 0
        mock_cmd_init.assert_called_once()
        args = mock_cmd_init.call_args[0][0]
        assert args.name == 'myapp'
        assert args.host == 'dokku.test.com'
        assert args.command == 'init'

    @patch('signalwire_agents.cli.dokku.cmd_init', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'init', 'myapp', '--cicd'])
    def test_main_init_cicd(self, mock_cmd_init):
        result = main()
        assert result == 0
        args = mock_cmd_init.call_args[0][0]
        assert args.cicd is True

    @patch('signalwire_agents.cli.dokku.cmd_init', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'init', 'myapp', '--web'])
    def test_main_init_web(self, mock_cmd_init):
        result = main()
        assert result == 0
        args = mock_cmd_init.call_args[0][0]
        assert args.web is True

    @patch('signalwire_agents.cli.dokku.cmd_init', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'init', 'myapp', '--force'])
    def test_main_init_force(self, mock_cmd_init):
        result = main()
        assert result == 0
        args = mock_cmd_init.call_args[0][0]
        assert args.force is True

    @patch('signalwire_agents.cli.dokku.cmd_init', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'init', 'myapp', '-f'])
    def test_main_init_force_short(self, mock_cmd_init):
        result = main()
        assert result == 0
        args = mock_cmd_init.call_args[0][0]
        assert args.force is True

    @patch('signalwire_agents.cli.dokku.cmd_init', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'init', 'myapp', '--dir', '/tmp/out'])
    def test_main_init_custom_dir(self, mock_cmd_init):
        result = main()
        assert result == 0
        args = mock_cmd_init.call_args[0][0]
        assert args.dir == '/tmp/out'

    @patch('signalwire_agents.cli.dokku.cmd_deploy', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'deploy', '--app', 'myapp', '--host', 'dokku.test.com'])
    def test_main_deploy_command(self, mock_cmd_deploy):
        result = main()
        assert result == 0
        mock_cmd_deploy.assert_called_once()
        args = mock_cmd_deploy.call_args[0][0]
        assert args.app == 'myapp'
        assert args.host == 'dokku.test.com'

    @patch('signalwire_agents.cli.dokku.cmd_deploy', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'deploy', '-a', 'myapp', '-H', 'dokku.test.com'])
    def test_main_deploy_short_flags(self, mock_cmd_deploy):
        result = main()
        assert result == 0
        args = mock_cmd_deploy.call_args[0][0]
        assert args.app == 'myapp'
        assert args.host == 'dokku.test.com'

    @patch('signalwire_agents.cli.dokku.cmd_logs', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'logs', '-a', 'myapp', '-H', 'h', '-t', '-n', '20'])
    def test_main_logs_command(self, mock_cmd_logs):
        result = main()
        assert result == 0
        mock_cmd_logs.assert_called_once()
        args = mock_cmd_logs.call_args[0][0]
        assert args.app == 'myapp'
        assert args.tail is True
        assert args.num == 20

    @patch('signalwire_agents.cli.dokku.cmd_config', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'config', 'set', 'KEY=val', '-a', 'myapp', '-H', 'h'])
    def test_main_config_set_command(self, mock_cmd_config):
        result = main()
        assert result == 0
        mock_cmd_config.assert_called_once()
        args = mock_cmd_config.call_args[0][0]
        assert args.config_action == 'set'
        assert args.vars == ['KEY=val']

    @patch('signalwire_agents.cli.dokku.cmd_config', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'config', 'show', '-a', 'myapp', '-H', 'h'])
    def test_main_config_show_command(self, mock_cmd_config):
        result = main()
        assert result == 0
        args = mock_cmd_config.call_args[0][0]
        assert args.config_action == 'show'

    @patch('signalwire_agents.cli.dokku.cmd_config', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'config', 'unset', 'KEY', '-a', 'myapp', '-H', 'h'])
    def test_main_config_unset_command(self, mock_cmd_config):
        result = main()
        assert result == 0
        args = mock_cmd_config.call_args[0][0]
        assert args.config_action == 'unset'
        assert args.vars == ['KEY']

    @patch('signalwire_agents.cli.dokku.cmd_scale', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'scale', 'web=2', '-a', 'myapp', '-H', 'h'])
    def test_main_scale_command(self, mock_cmd_scale):
        result = main()
        assert result == 0
        mock_cmd_scale.assert_called_once()
        args = mock_cmd_scale.call_args[0][0]
        assert args.scale_args == ['web=2']

    @patch('sys.argv', ['sw-agent-dokku'])
    def test_main_no_command_returns_1(self):
        result = main()
        assert result == 1


# =============================================================================
# Template Content Tests
# =============================================================================

class TestTemplates:
    """Tests to verify template content is correctly defined."""

    def test_procfile_template_has_gunicorn(self):
        assert 'gunicorn' in PROCFILE_TEMPLATE
        assert 'UvicornWorker' in PROCFILE_TEMPLATE

    def test_runtime_template_has_python(self):
        assert 'python-3.11' in RUNTIME_TEMPLATE

    def test_requirements_template_has_deps(self):
        assert 'signalwire-agents' in REQUIREMENTS_TEMPLATE
        assert 'gunicorn' in REQUIREMENTS_TEMPLATE
        assert 'uvicorn' in REQUIREMENTS_TEMPLATE

    def test_checks_template_has_health(self):
        assert '/health' in CHECKS_TEMPLATE

    def test_gitignore_template_excludes_env(self):
        assert '.env' in GITIGNORE_TEMPLATE
        assert '__pycache__' in GITIGNORE_TEMPLATE

    def test_env_example_template_has_placeholders(self):
        assert '{app_name}' in ENV_EXAMPLE_TEMPLATE
        assert 'SIGNALWIRE_SPACE_NAME' in ENV_EXAMPLE_TEMPLATE

    def test_app_template_has_class_placeholder(self):
        assert '{agent_class}' in APP_TEMPLATE
        assert '{agent_name}' in APP_TEMPLATE
        assert '{agent_slug}' in APP_TEMPLATE

    def test_app_template_with_web_has_server(self):
        assert 'AgentServer' in APP_TEMPLATE_WITH_WEB
        assert 'setup_swml_handler' in APP_TEMPLATE_WITH_WEB

    def test_app_json_template_is_valid_json_after_format(self):
        content = APP_JSON_TEMPLATE.format(app_name='test')
        data = json.loads(content)
        assert data['name'] == 'test'

    def test_deploy_workflow_template_mentions_dokku(self):
        assert 'dokku-deploy-system' in DEPLOY_WORKFLOW_TEMPLATE

    def test_preview_workflow_template_mentions_pull_request(self):
        assert 'pull_request' in PREVIEW_WORKFLOW_TEMPLATE

    def test_dokku_config_template_has_resources(self):
        assert 'resources:' in DOKKU_CONFIG_TEMPLATE
        assert 'memory:' in DOKKU_CONFIG_TEMPLATE

    def test_services_template_has_postgres(self):
        assert 'postgres:' in SERVICES_TEMPLATE

    def test_web_index_template_has_html(self):
        assert '<!DOCTYPE html>' in WEB_INDEX_TEMPLATE
        assert '{agent_name}' in WEB_INDEX_TEMPLATE

    def test_deploy_script_template_has_bash(self):
        assert '#!/bin/bash' in DEPLOY_SCRIPT_TEMPLATE
        assert '{app_name}' in DEPLOY_SCRIPT_TEMPLATE

    def test_readme_simple_template_has_deploy(self):
        assert 'deploy' in README_SIMPLE_TEMPLATE.lower()

    def test_readme_cicd_template_has_github(self):
        assert 'GitHub' in README_CICD_TEMPLATE


# =============================================================================
# Edge Case and Integration Tests
# =============================================================================

class TestEdgeCases:
    """Tests for various edge cases and special scenarios."""

    def test_generate_password_length_zero(self):
        pw = generate_password(length=0)
        assert pw == ''

    def test_generate_password_length_one(self):
        pw = generate_password(length=1)
        assert len(pw) == 1

    def test_agent_class_name_mixed_delimiters(self):
        gen = DokkuProjectGenerator("my-cool_app name", {})
        # "my-cool_app name" -> slug: "my-cool-app-name"
        # class: split on - and _ and space -> My Cool App Name + Agent
        assert gen.agent_slug == "my-cool-app-name"
        assert gen.agent_class == "MyCoolAppNameAgent"

    def test_agent_class_name_all_caps(self):
        gen = DokkuProjectGenerator("ABC", {})
        assert gen.agent_slug == "abc"
        assert gen.agent_class == "AbcAgent"

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_deploy_remote_url_format(self, mock_subprocess):
        """Verify the dokku remote URL is correctly formed."""
        with patch('signalwire_agents.cli.dokku.Path') as mock_path_cls:
            path_instances = {}

            def path_side_effect(p):
                if p not in path_instances:
                    m = MagicMock()
                    path_instances[p] = m
                    if p == 'Procfile':
                        m.exists.return_value = True
                    elif p == 'app.json':
                        m.exists.return_value = False
                    elif p == '.git':
                        m.exists.return_value = True
                    else:
                        m.exists.return_value = False
                return path_instances[p]

            mock_path_cls.side_effect = path_side_effect
            mock_subprocess.run.return_value = MagicMock(returncode=0)

            args = argparse.Namespace(app='myapp', host='dokku.host.com')
            cmd_deploy(args)

            # Find the remote add call
            for c in mock_subprocess.run.call_args_list:
                call_args = c[0][0]
                if 'remote' in call_args and 'add' in call_args:
                    assert 'dokku@dokku.host.com:myapp' in call_args
                    break

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_logs_command_structure(self, mock_subprocess):
        """Verify the SSH log command is correctly formed."""
        args = argparse.Namespace(
            app='testapp', host='my.dokku.host', tail=True, num=200
        )
        cmd_logs(args)
        cmd = mock_subprocess.run.call_args[0][0]
        assert cmd[0] == 'ssh'
        assert cmd[1] == 'dokku@my.dokku.host'
        assert cmd[2] == 'logs'
        assert cmd[3] == 'testapp'
        assert '-t' in cmd
        assert '--num' in cmd
        assert '200' in cmd

    def test_config_set_empty_vars_list(self):
        """Empty vars list (not None, but []) should still fail."""
        args = argparse.Namespace(
            config_action='set', vars=[], app='myapp', host='dokku.example.com'
        )
        result = cmd_config(args)
        assert result == 1

    def test_config_unset_empty_vars_list(self):
        """Empty vars list (not None, but []) should still fail."""
        args = argparse.Namespace(
            config_action='unset', vars=[], app='myapp', host='dokku.example.com'
        )
        result = cmd_config(args)
        assert result == 1

    @patch('signalwire_agents.cli.dokku.subprocess')
    def test_scale_show_no_extra_args(self, mock_subprocess):
        """When scale_args is empty, just show current scale without extra args."""
        args = argparse.Namespace(
            scale_args=[], app='myapp', host='dokku.example.com'
        )
        cmd_scale(args)
        cmd = mock_subprocess.run.call_args[0][0]
        # Should be exactly: ssh dokku@host ps:scale myapp
        assert cmd == ['ssh', 'dokku@dokku.example.com', 'ps:scale', 'myapp']

    def test_generate_creates_project_dir_if_missing(self, tmp_path):
        """generate() should create the project directory with parents."""
        deep_dir = tmp_path / "a" / "b" / "c"
        gen = DokkuProjectGenerator("myapp", {
            'project_dir': str(deep_dir),
            'dokku_host': 'dokku.example.com',
        })
        result = gen.generate()
        assert result is True
        assert deep_dir.exists()

    @patch('signalwire_agents.cli.dokku.cmd_init', return_value=0)
    @patch('sys.argv', ['sw-agent-dokku', 'init', 'my-app', '--cicd', '--web',
                        '--host', 'h', '--dir', '/tmp/d', '-f'])
    def test_main_all_init_flags(self, mock_cmd_init):
        """All init flags can be passed together."""
        result = main()
        assert result == 0
        args = mock_cmd_init.call_args[0][0]
        assert args.name == 'my-app'
        assert args.cicd is True
        assert args.web is True
        assert args.host == 'h'
        assert args.dir == '/tmp/d'
        assert args.force is True
