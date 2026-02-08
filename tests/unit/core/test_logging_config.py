"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for logging_config module
"""

import pytest
import logging
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from signalwire_agents.core.logging_config import (
    get_logger, 
    get_execution_mode, 
    configure_logging,
    StructuredLoggerWrapper,
    _logging_configured
)


class TestGetExecutionMode:
    """Test execution mode detection"""
    
    def test_cgi_mode_detection(self):
        """Test CGI mode detection"""
        with patch.dict(os.environ, {'GATEWAY_INTERFACE': 'CGI/1.1'}, clear=False):
            assert get_execution_mode() == 'cgi'
    
    def test_lambda_mode_detection(self):
        """Test Lambda mode detection"""
        with patch.dict(os.environ, {'AWS_LAMBDA_FUNCTION_NAME': 'test-function'}, clear=False):
            assert get_execution_mode() == 'lambda'
    
    def test_lambda_mode_detection_with_task_root(self):
        """Test Lambda mode detection with LAMBDA_TASK_ROOT"""
        with patch.dict(os.environ, {'LAMBDA_TASK_ROOT': '/var/task'}, clear=False):
            assert get_execution_mode() == 'lambda'
    
    def test_server_mode_default(self):
        """Test server mode as default"""
        # Clear all serverless environment variables
        env_vars_to_clear = [
            'GATEWAY_INTERFACE', 'AWS_LAMBDA_FUNCTION_NAME', 
            'LAMBDA_TASK_ROOT'
        ]
        
        with patch.dict(os.environ, {}, clear=False):
            for var in env_vars_to_clear:
                if var in os.environ:
                    del os.environ[var]
            assert get_execution_mode() == 'server'


class TestStructuredLoggerWrapper:
    """Test StructuredLoggerWrapper functionality"""
    
    def test_initialization(self):
        """Test wrapper initialization"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        assert wrapper._logger is mock_logger
    
    def test_basic_logging_methods(self):
        """Test basic logging methods"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        # Test each logging level
        wrapper.debug("debug message")
        mock_logger.debug.assert_called_with("debug message")
        
        wrapper.info("info message")
        mock_logger.info.assert_called_with("info message")
        
        wrapper.warning("warning message")
        mock_logger.warning.assert_called_with("warning message")
        
        wrapper.error("error message")
        mock_logger.error.assert_called_with("error message")
        
        wrapper.critical("critical message")
        mock_logger.critical.assert_called_with("critical message")
    
    def test_structured_logging_with_kwargs(self):
        """Test structured logging with keyword arguments"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        wrapper.info("test message", user_id="123", action="login")
        
        # Should format the message with structured data using parentheses format
        expected_call = "test message (user_id=123, action=login)"
        mock_logger.info.assert_called_with(expected_call)
    
    def test_structured_logging_with_complex_values(self):
        """Test structured logging with complex values"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        wrapper.info("test", data={"key": "value"}, count=42, enabled=True)
        
        # Should handle different data types
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "test (" in call_args
        assert "data=" in call_args
        assert "count=42" in call_args
        assert "enabled=True" in call_args
    
    def test_attribute_delegation(self):
        """Test that other attributes are delegated to underlying logger"""
        mock_logger = Mock()
        mock_logger.level = logging.INFO
        mock_logger.handlers = []
        
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        assert wrapper.level == logging.INFO
        assert wrapper.handlers == []
    
    def test_method_delegation(self):
        """Test that other methods are delegated to underlying logger"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        wrapper.setLevel(logging.DEBUG)
        mock_logger.setLevel.assert_called_with(logging.DEBUG)
        
        wrapper.addHandler(Mock())
        mock_logger.addHandler.assert_called()
    
    def test_warn_alias(self):
        """Test that warn is an alias for warning"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        wrapper.warn("warning message")
        mock_logger.warning.assert_called_with("warning message")


class TestConfigureLogging:
    """Test logging configuration"""
    
    def test_configure_logging_basic(self):
        """Test basic logging configuration"""
        with patch('signalwire_agents.core.logging_config._logging_configured', False):
            with patch('signalwire_agents.core.logging_config._configure_default_mode') as mock_default:
                configure_logging()
                mock_default.assert_called_once()
    
    def test_configure_logging_idempotent(self):
        """Test that configure_logging is idempotent"""
        with patch('signalwire_agents.core.logging_config._logging_configured', True):
            with patch('logging.basicConfig') as mock_basic_config:
                configure_logging()
                mock_basic_config.assert_not_called()
    
    def test_configure_logging_with_env_vars(self):
        """Test logging configuration with environment variables"""
        with patch('signalwire_agents.core.logging_config._logging_configured', False):
            with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'stderr', 'SIGNALWIRE_LOG_LEVEL': 'debug'}):
                with patch('signalwire_agents.core.logging_config._configure_stderr_mode') as mock_stderr:
                    configure_logging()
                    mock_stderr.assert_called_once_with('debug')
    
    def test_configure_logging_off_mode(self):
        """Test logging configuration in off mode"""
        with patch('signalwire_agents.core.logging_config._logging_configured', False):
            with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'off'}):
                with patch('logging.basicConfig') as mock_basic_config:
                    with patch('builtins.open', create=True) as mock_open:
                        configure_logging()
                        mock_basic_config.assert_called()
                        # Should configure with devnull
                        call_kwargs = mock_basic_config.call_args[1]
                        assert call_kwargs['level'] == logging.CRITICAL


class TestGetLogger:
    """Test get_logger function"""
    
    def test_get_logger_basic(self):
        """Test basic logger creation"""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, StructuredLoggerWrapper)
        assert logger.name == "test_logger"
    
    def test_get_logger_different_names(self):
        """Test that different names create different loggers"""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        
        assert logger1 is not logger2
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
    
    def test_get_logger_configures_logging(self):
        """Test that get_logger triggers logging configuration"""
        with patch('signalwire_agents.core.logging_config._logging_configured', False):
            with patch('signalwire_agents.core.logging_config.configure_logging') as mock_configure:
                get_logger("test")
                mock_configure.assert_called_once()


class TestLoggingIntegration:
    """Test logging integration scenarios"""
    
    def test_logger_output_format(self):
        """Test that logger produces expected output format"""
        # Create a string buffer to capture output
        log_buffer = StringIO()
        
        # Create a logger with a stream handler
        logger = logging.getLogger("test_integration")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Add our test handler
        handler = logging.StreamHandler(log_buffer)
        handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        logger.addHandler(handler)
        
        # Create wrapper and test
        wrapper = StructuredLoggerWrapper(logger)
        wrapper.info("test message", key1="value1", key2=42)
        
        # Check output
        output = log_buffer.getvalue()
        assert "INFO:test_integration:" in output
        assert "test message" in output
        assert "key1=value1" in output
        assert "key2=42" in output
    
    def test_logger_with_exception(self):
        """Test logger behavior with exceptions"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        try:
            raise ValueError("test error")
        except ValueError:
            wrapper.error("An error occurred", exc_info=True)
        
        # Should pass exc_info through as part of the formatted message
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[0][0]
        assert "exc_info=True" in call_args
    
    def test_logger_performance(self):
        """Test logger performance with many calls"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        # Should handle many rapid calls without issues
        for i in range(100):
            wrapper.info(f"message {i}", iteration=i)
        
        assert mock_logger.info.call_count == 100


class TestLoggingErrorHandling:
    """Test error handling in logging"""
    
    def test_wrapper_with_none_logger(self):
        """Test wrapper behavior with None logger"""
        with pytest.raises(AttributeError):
            wrapper = StructuredLoggerWrapper(None)
            wrapper.info("test")
    
    def test_wrapper_with_invalid_logger(self):
        """Test wrapper behavior with invalid logger"""
        invalid_logger = "not a logger"
        wrapper = StructuredLoggerWrapper(invalid_logger)
        
        with pytest.raises(AttributeError):
            wrapper.info("test")
    
    def test_structured_logging_with_none_values(self):
        """Test structured logging with None values"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        wrapper.info("test", value=None, other="test")
        
        # Should handle None values gracefully
        call_args = mock_logger.info.call_args[0][0]
        assert "value=None" in call_args
        assert "other=test" in call_args
    
    def test_structured_logging_with_special_characters(self):
        """Test structured logging with special characters"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        wrapper.info("test", msg="hello=world", path="/test/path")
        
        # Should handle special characters in values
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "msg=" in call_args
        assert "path=" in call_args
    
    def test_format_structured_message_no_kwargs(self):
        """Test message formatting with no kwargs"""
        mock_logger = Mock()
        wrapper = StructuredLoggerWrapper(mock_logger)
        
        wrapper.info("simple message")
        mock_logger.info.assert_called_with("simple message")


class TestLoggingModeConfiguration:
    """Test different logging mode configurations"""
    
    def test_auto_mode_cgi_environment(self):
        """Test auto mode detection in CGI environment"""
        with patch('signalwire_agents.core.logging_config._logging_configured', False):
            with patch.dict(os.environ, {'GATEWAY_INTERFACE': 'CGI/1.1', 'SIGNALWIRE_LOG_MODE': 'auto'}):
                with patch('signalwire_agents.core.logging_config._configure_off_mode') as mock_off:
                    configure_logging()
                    mock_off.assert_called_once()
    
    def test_auto_mode_server_environment(self):
        """Test auto mode detection in server environment"""
        with patch('signalwire_agents.core.logging_config._logging_configured', False):
            with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'auto'}, clear=False):
                # Clear serverless env vars
                for var in ['GATEWAY_INTERFACE', 'AWS_LAMBDA_FUNCTION_NAME', 'LAMBDA_TASK_ROOT']:
                    if var in os.environ:
                        del os.environ[var]
                
                with patch('signalwire_agents.core.logging_config._configure_default_mode') as mock_default:
                    configure_logging()
                    mock_default.assert_called_once()
    
    def test_default_mode_configuration(self):
        """Test default mode configuration"""
        with patch('signalwire_agents.core.logging_config._logging_configured', False):
            with patch.dict(os.environ, {'SIGNALWIRE_LOG_MODE': 'default', 'SIGNALWIRE_LOG_LEVEL': 'warning'}):
                with patch('signalwire_agents.core.logging_config._configure_default_mode') as mock_default:
                    configure_logging()
                    mock_default.assert_called_once_with('warning') 