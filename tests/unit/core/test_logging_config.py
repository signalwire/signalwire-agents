"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for logging_config module (structlog-based)
"""

import pytest
import logging
import json
import os
import sys
from unittest.mock import patch
from io import StringIO

import structlog

from signalwire_agents.core.logging_config import (
    get_logger,
    get_execution_mode,
    configure_logging,
    reset_logging_configuration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_logging(monkeypatch):
    """Reset logging state before every test."""
    reset_logging_configuration()
    # Also clear any handlers left from previous tests
    sw_logger = logging.getLogger("signalwire_agents")
    sw_logger.handlers.clear()
    sw_logger.setLevel(logging.NOTSET)
    sw_logger.propagate = True
    yield
    reset_logging_configuration()
    sw_logger.handlers.clear()
    sw_logger.setLevel(logging.NOTSET)
    sw_logger.propagate = True


# ===========================================================================
# Execution mode detection (unchanged logic)
# ===========================================================================


class TestGetExecutionMode:
    """Test execution mode detection"""

    def test_cgi_mode_detection(self):
        with patch.dict(os.environ, {'GATEWAY_INTERFACE': 'CGI/1.1'}, clear=False):
            assert get_execution_mode() == 'cgi'

    def test_lambda_mode_detection(self):
        with patch.dict(os.environ, {'AWS_LAMBDA_FUNCTION_NAME': 'test-function'}, clear=False):
            assert get_execution_mode() == 'lambda'

    def test_lambda_mode_detection_with_task_root(self):
        with patch.dict(os.environ, {'LAMBDA_TASK_ROOT': '/var/task'}, clear=False):
            assert get_execution_mode() == 'lambda'

    def test_google_cloud_function_detection(self):
        with patch.dict(os.environ, {'FUNCTION_TARGET': 'my_function'}, clear=False):
            assert get_execution_mode() == 'google_cloud_function'

    def test_azure_function_detection(self):
        with patch.dict(os.environ, {'AZURE_FUNCTIONS_ENVIRONMENT': 'Production'}, clear=False):
            assert get_execution_mode() == 'azure_function'

    def test_server_mode_default(self):
        env_vars_to_clear = [
            'GATEWAY_INTERFACE', 'AWS_LAMBDA_FUNCTION_NAME',
            'LAMBDA_TASK_ROOT', 'FUNCTION_TARGET', 'K_SERVICE',
            'GOOGLE_CLOUD_PROJECT', 'AZURE_FUNCTIONS_ENVIRONMENT',
            'FUNCTIONS_WORKER_RUNTIME', 'AzureWebJobsStorage',
        ]
        with patch.dict(os.environ, {}, clear=False):
            for var in env_vars_to_clear:
                os.environ.pop(var, None)
            assert get_execution_mode() == 'server'


# ===========================================================================
# get_logger
# ===========================================================================


class TestGetLogger:
    """Test get_logger function"""

    def test_returns_structlog_bound_logger(self):
        logger = get_logger("test_logger")
        # structlog BoundLoggers should have bind()
        assert hasattr(logger, 'bind')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')

    def test_different_names_create_different_loggers(self):
        logger1 = get_logger("logger_a")
        logger2 = get_logger("logger_b")
        assert logger1 is not logger2

    def test_triggers_configure_logging(self):
        with patch('signalwire_agents.core.logging_config.configure_logging') as mock_conf:
            get_logger("test")
            mock_conf.assert_called_once()


# ===========================================================================
# configure_logging
# ===========================================================================


class TestConfigureLogging:
    """Test logging configuration"""

    def test_idempotent(self):
        """configure_logging should only run once until reset."""
        configure_logging()
        # Grab the handler count after first call
        sw = logging.getLogger("signalwire_agents")
        handler_count = len(sw.handlers)
        # Second call should be a no-op (no extra handlers)
        configure_logging()
        assert len(sw.handlers) == handler_count

    def test_default_mode(self):
        """Default (no env var) should attach a handler to signalwire_agents."""
        configure_logging()
        sw = logging.getLogger("signalwire_agents")
        assert len(sw.handlers) == 1
        assert sw.propagate is False

    def test_off_mode(self):
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'off'}):
            configure_logging()
            sw = logging.getLogger("signalwire_agents")
            # Off mode sets level above CRITICAL and no handlers
            assert sw.level > logging.CRITICAL
            assert len(sw.handlers) == 0

    def test_stderr_mode(self):
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'stderr'}):
            configure_logging()
            sw = logging.getLogger("signalwire_agents")
            assert len(sw.handlers) == 1
            assert sw.handlers[0].stream is sys.stderr

    def test_auto_mode_cgi(self):
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'auto', 'GATEWAY_INTERFACE': 'CGI/1.1'}):
            configure_logging()
            sw = logging.getLogger("signalwire_agents")
            # CGI → off mode
            assert sw.level > logging.CRITICAL

    def test_auto_mode_server(self):
        env = {'SIGNALWIRE_LOG_MODE': 'auto'}
        removals = ['GATEWAY_INTERFACE', 'AWS_LAMBDA_FUNCTION_NAME', 'LAMBDA_TASK_ROOT']
        with patch.dict(os.environ, env, clear=False):
            for v in removals:
                os.environ.pop(v, None)
            configure_logging()
            sw = logging.getLogger("signalwire_agents")
            assert len(sw.handlers) == 1

    def test_log_level_env(self):
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_LEVEL': 'debug'}):
            configure_logging()
            sw = logging.getLogger("signalwire_agents")
            assert sw.level == logging.DEBUG

    def test_json_format(self):
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_FORMAT': 'json'}):
            configure_logging()
            sw = logging.getLogger("signalwire_agents")
            assert len(sw.handlers) == 1
            # Handler's formatter should be a ProcessorFormatter
            fmt = sw.handlers[0].formatter
            assert isinstance(fmt, structlog.stdlib.ProcessorFormatter)


# ===========================================================================
# Structured logging behaviour
# ===========================================================================


class TestStructuredLogging:
    """Test that structlog features work end-to-end through stdlib."""

    def test_bind_adds_context(self):
        """bound fields should appear in the log output."""
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_FORMAT': 'json', 'SIGNALWIRE_LOG_LEVEL': 'debug'}):
            configure_logging()

            buf = StringIO()
            sw = logging.getLogger("signalwire_agents")
            sw.handlers[0].stream = buf

            log = get_logger("signalwire_agents.test_bind")
            bound = log.bind(request_id="abc123")
            bound.info("hello")

            output = buf.getvalue().strip()
            data = json.loads(output)
            assert data["request_id"] == "abc123"
            assert data["event"] == "hello"

    def test_nested_bind(self):
        """Multiple bind() calls should stack context."""
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_FORMAT': 'json', 'SIGNALWIRE_LOG_LEVEL': 'debug'}):
            configure_logging()

            buf = StringIO()
            sw = logging.getLogger("signalwire_agents")
            sw.handlers[0].stream = buf

            log = get_logger("signalwire_agents.test_nested")
            log2 = log.bind(a="1").bind(b="2")
            log2.info("nested")

            output = buf.getvalue().strip()
            data = json.loads(output)
            assert data["a"] == "1"
            assert data["b"] == "2"
            assert data["event"] == "nested"

    def test_exc_info_produces_traceback(self):
        """exc_info should produce a real traceback, not 'exc_info=True' string."""
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_FORMAT': 'json', 'SIGNALWIRE_LOG_LEVEL': 'debug'}):
            configure_logging()

            buf = StringIO()
            sw = logging.getLogger("signalwire_agents")
            sw.handlers[0].stream = buf

            log = get_logger("signalwire_agents.test_exc")

            try:
                raise ValueError("boom")
            except ValueError:
                log.error("caught error", exc_info=True)

            output = buf.getvalue().strip()
            data = json.loads(output)
            assert data["event"] == "caught error"
            # The exception field should contain the traceback string
            assert "ValueError: boom" in data.get("exception", "")


# ===========================================================================
# Off mode: no FD leak
# ===========================================================================


class TestOffModeNoFdLeak:
    """Off mode should not open /dev/null or any file."""

    def test_no_file_handles_opened(self):
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'off'}):
            with patch('builtins.open') as mock_open:
                configure_logging()
                mock_open.assert_not_called()


# ===========================================================================
# No root logger hijacking
# ===========================================================================


class TestNoRootLoggerHijacking:
    """Verify root logger handlers are not modified."""

    def test_root_logger_unchanged(self):
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        original_level = root.level

        configure_logging()

        assert root.handlers == original_handlers
        assert root.level == original_level


# ===========================================================================
# JSON mode
# ===========================================================================


class TestJsonMode:
    """Verify JSON mode produces parseable output with structured fields."""

    def test_json_output(self):
        with patch.dict(os.environ, {'SIGNALWIRE_LOG_FORMAT': 'json', 'SIGNALWIRE_LOG_LEVEL': 'debug'}):
            configure_logging()

            buf = StringIO()
            sw = logging.getLogger("signalwire_agents")
            # Replace handler's stream with our buffer
            sw.handlers[0].stream = buf

            log = get_logger("signalwire_agents.test_json")
            bound = log.bind(user="alice")
            bound.info("login")

            output = buf.getvalue().strip()
            assert output, "Expected JSON output"
            data = json.loads(output)
            assert data["user"] == "alice"
            assert data["event"] == "login"


# ===========================================================================
# Color detection
# ===========================================================================


class TestColorDetection:
    """Verify color auto-detection logic."""

    def test_no_colors_when_not_tty(self):
        from signalwire_agents.core.logging_config import _detect_colors
        with patch.object(sys, 'stdout', new_callable=StringIO):
            assert _detect_colors() is False

    def test_no_colors_when_dump_swml(self):
        from signalwire_agents.core.logging_config import _detect_colors
        original_argv = sys.argv[:]
        try:
            sys.argv.append('--dump-swml')
            assert _detect_colors() is False
        finally:
            sys.argv[:] = original_argv

    def test_no_colors_when_raw(self):
        from signalwire_agents.core.logging_config import _detect_colors
        original_argv = sys.argv[:]
        try:
            sys.argv.append('--raw')
            assert _detect_colors() is False
        finally:
            sys.argv[:] = original_argv


# ===========================================================================
# reset_logging_configuration
# ===========================================================================


class TestResetLogging:
    """Test that reset allows reconfiguration."""

    def test_reset_allows_reconfigure(self):
        configure_logging()
        sw = logging.getLogger("signalwire_agents")
        assert len(sw.handlers) >= 1

        reset_logging_configuration()
        sw.handlers.clear()
        sw.setLevel(logging.NOTSET)

        with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'off'}):
            configure_logging()
            assert sw.level > logging.CRITICAL
