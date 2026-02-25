"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Central logging configuration for SignalWire Agents SDK

This module provides a single point of control for all logging across the SDK.
All components should use get_logger() instead of direct logging module usage
or print() statements.

Uses structlog with stdlib integration so that:
- All SDK loggers go through structlog processors (bind, JSON, colors)
- pytest caplog still works (structlog writes through stdlib)
- Third-party loggers are unaffected (handler on signalwire_agents only)
"""

import logging
import os
import re
import sys

import structlog

_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def strip_control_chars(logger, method_name, event_dict):
    """Strip control characters from log event values to prevent log injection."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = _CONTROL_CHAR_RE.sub('', value)
    return event_dict

# Global flag to ensure configuration only happens once
_logging_configured = False


def get_execution_mode():
    """
    Determine the execution mode based on environment variables

    Returns:
        str: 'server', 'cgi', 'lambda', 'google_cloud_function', 'azure_function', or 'unknown'
    """
    # Check for CGI environment
    if os.getenv('GATEWAY_INTERFACE'):
        return 'cgi'

    # Check for AWS Lambda environment
    if os.getenv('AWS_LAMBDA_FUNCTION_NAME') or os.getenv('LAMBDA_TASK_ROOT'):
        return 'lambda'

    # Check for Google Cloud Functions environment
    if (os.getenv('FUNCTION_TARGET') or
        os.getenv('K_SERVICE') or
        os.getenv('GOOGLE_CLOUD_PROJECT')):
        return 'google_cloud_function'

    # Check for Azure Functions environment
    if (os.getenv('AZURE_FUNCTIONS_ENVIRONMENT') or
        os.getenv('FUNCTIONS_WORKER_RUNTIME') or
        os.getenv('AzureWebJobsStorage')):
        return 'azure_function'

    # Default to server mode
    return 'server'


def reset_logging_configuration():
    """
    Reset the logging configuration flag to allow reconfiguration

    This is useful when environment variables change after initial configuration.
    """
    global _logging_configured
    _logging_configured = False
    structlog.reset_defaults()


def _detect_colors():
    """Auto-detect whether the output stream supports colors."""
    stream = sys.stderr if os.getenv('SIGNALWIRE_LOG_MODE', '').lower() == 'stderr' else sys.stdout
    if not hasattr(stream, 'isatty'):
        return False
    if not stream.isatty():
        return False
    if '--raw' in sys.argv or '--dump-swml' in sys.argv:
        return False
    return True


def configure_logging():
    """
    Configure logging system once, globally, based on environment variables

    Environment Variables:
        SIGNALWIRE_LOG_MODE: off, stderr, default, auto
        SIGNALWIRE_LOG_LEVEL: debug, info, warning, error, critical
        SIGNALWIRE_LOG_FORMAT: console, json (default: console)
    """
    global _logging_configured

    if _logging_configured:
        return

    # Get configuration from environment
    log_mode = os.getenv('SIGNALWIRE_LOG_MODE', '').lower()
    log_level = os.getenv('SIGNALWIRE_LOG_LEVEL', 'info').lower()
    log_format = os.getenv('SIGNALWIRE_LOG_FORMAT', 'console').lower()

    # Determine log mode if auto or not specified
    if not log_mode or log_mode == 'auto':
        execution_mode = get_execution_mode()
        if execution_mode == 'cgi':
            log_mode = 'off'
        else:
            log_mode = 'default'

    # Configure based on mode
    if log_mode == 'off':
        _configure_off_mode()
    elif log_mode == 'stderr':
        _configure_stderr_mode(log_level, log_format)
    else:  # default mode
        _configure_default_mode(log_level, log_format)

    _logging_configured = True


def _get_structlog_processors():
    """Processor chain for structlog.get_logger() callers."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        strip_control_chars,
    ]


def _drop_internal_keys(logger, method_name, event_dict):
    """Remove structlog/ProcessorFormatter internal keys from the event dict."""
    event_dict.pop("_record", None)
    event_dict.pop("_from_structlog", None)
    return event_dict


def _get_formatter_processors():
    """Processor chain for ProcessorFormatter (stdlib LogRecords).

    Does NOT include filter_by_level because:
    1. stdlib already filters by level before records reach the handler
    2. filter_by_level requires a Logger object which may be None in this context
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        strip_control_chars,
        _drop_internal_keys,
    ]


def _configure_structlog(level_num, log_format, stream):
    """Configure structlog and attach a handler to the signalwire_agents logger.

    Args:
        level_num: numeric logging level (e.g. logging.INFO)
        log_format: 'console' or 'json'
        stream: output stream (sys.stdout or sys.stderr)
    """
    # Choose final renderer
    if log_format == 'json':
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=_detect_colors())

    # Configure structlog itself (for structlog.get_logger() callers)
    structlog.configure(
        processors=_get_structlog_processors() + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Build a shared handler with ProcessorFormatter
    handler = logging.StreamHandler(stream)
    handler.setLevel(level_num)
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=_get_formatter_processors() + [renderer],
    )
    handler.setFormatter(formatter)

    # Attach handler to the signalwire_agents namespace (not root)
    sw_logger = logging.getLogger("signalwire_agents")
    sw_logger.handlers.clear()
    sw_logger.setLevel(level_num)
    sw_logger.propagate = False  # Don't bubble up to root
    sw_logger.addHandler(handler)

    # Also attach to known SDK short-name loggers
    for name in _get_sdk_logger_names():
        lgr = logging.getLogger(name)
        lgr.handlers.clear()
        lgr.setLevel(level_num)
        lgr.propagate = False
        lgr.addHandler(handler)


def _get_sdk_logger_names():
    """Known SDK logger names that don't use the signalwire_agents. prefix.

    These are used by the 11 files that call get_logger() with short names.
    They need to be handled alongside the signalwire_agents namespace logger.
    """
    return [
        "swml_service", "agent_base", "AgentServer", "skill_registry",
        "skill_manager", "security_config", "config_loader", "auth_handler",
        "web_service", "search_service", "bedrock_agent",
    ]


def _configure_off_mode():
    """Suppress all logging output without leaking file descriptors."""
    off_level = logging.CRITICAL + 10

    # Silence the signalwire_agents namespace
    sw_logger = logging.getLogger("signalwire_agents")
    sw_logger.handlers.clear()
    sw_logger.setLevel(off_level)
    sw_logger.propagate = False

    # Silence known SDK short-name loggers
    for name in _get_sdk_logger_names():
        lgr = logging.getLogger(name)
        lgr.handlers.clear()
        lgr.setLevel(off_level)
        lgr.propagate = False

    # Configure structlog with a filtering bound logger that suppresses everything
    structlog.configure(
        processors=[
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
        cache_logger_on_first_use=True,
    )


def _configure_stderr_mode(log_level, log_format='console'):
    """Configure logging to stderr."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    _configure_structlog(numeric_level, log_format, sys.stderr)


def _configure_default_mode(log_level, log_format='console'):
    """Configure standard logging to stdout."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    _configure_structlog(numeric_level, log_format, sys.stdout)


def get_logger(name):
    """
    Get a logger instance for the specified name with structured logging support

    This is the single entry point for all logging in the SDK.
    All modules should use this instead of direct logging module usage.

    Args:
        name: Logger name, typically __name__

    Returns:
        A structlog BoundLogger that supports .bind(), .info(), .debug(), etc.
    """
    # Ensure logging is configured
    configure_logging()

    return structlog.get_logger(name)
