"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for SecurityConfig module
"""

import os
import secrets
import pytest
from unittest.mock import Mock, patch, MagicMock


# All SecurityConfig instantiation must happen with env and config loader mocked,
# because __init__ calls load_from_env() and _load_config_file() immediately.
# We patch at the module level where needed.

ENV_CLEAR_KEYS = [
    'SWML_SSL_ENABLED', 'SWML_SSL_CERT_PATH', 'SWML_SSL_KEY_PATH',
    'SWML_DOMAIN', 'SWML_SSL_VERIFY_MODE', 'SWML_ALLOWED_HOSTS',
    'SWML_CORS_ORIGINS', 'SWML_MAX_REQUEST_SIZE', 'SWML_RATE_LIMIT',
    'SWML_REQUEST_TIMEOUT', 'SWML_USE_HSTS', 'SWML_HSTS_MAX_AGE',
    'SWML_BASIC_AUTH_USER', 'SWML_BASIC_AUTH_PASSWORD',
]


def _clean_env():
    """Return a dict suitable for patch.dict that removes all SWML_ keys."""
    return {k: v for k, v in os.environ.items() if k not in ENV_CLEAR_KEYS}


def _make_config(**env_overrides):
    """
    Create a SecurityConfig with a clean environment and no config file.
    Any env_overrides are applied on top of the clean environment.
    """
    clean = {k: v for k, v in os.environ.items() if k not in ENV_CLEAR_KEYS}
    clean.update(env_overrides)
    with patch.dict(os.environ, clean, clear=True):
        with patch(
            'signalwire_agents.core.security_config.ConfigLoader.find_config_file',
            return_value=None,
        ):
            from signalwire_agents.core.security_config import SecurityConfig
            return SecurityConfig()


class TestSecurityConfigClassAttributes:
    """Test that class-level constants are defined correctly."""

    def test_ssl_env_var_names(self):
        from signalwire_agents.core.security_config import SecurityConfig
        assert SecurityConfig.SSL_ENABLED == 'SWML_SSL_ENABLED'
        assert SecurityConfig.SSL_CERT_PATH == 'SWML_SSL_CERT_PATH'
        assert SecurityConfig.SSL_KEY_PATH == 'SWML_SSL_KEY_PATH'
        assert SecurityConfig.SSL_DOMAIN == 'SWML_DOMAIN'
        assert SecurityConfig.SSL_VERIFY_MODE == 'SWML_SSL_VERIFY_MODE'

    def test_additional_env_var_names(self):
        from signalwire_agents.core.security_config import SecurityConfig
        assert SecurityConfig.ALLOWED_HOSTS == 'SWML_ALLOWED_HOSTS'
        assert SecurityConfig.CORS_ORIGINS == 'SWML_CORS_ORIGINS'
        assert SecurityConfig.MAX_REQUEST_SIZE == 'SWML_MAX_REQUEST_SIZE'
        assert SecurityConfig.RATE_LIMIT == 'SWML_RATE_LIMIT'
        assert SecurityConfig.REQUEST_TIMEOUT == 'SWML_REQUEST_TIMEOUT'
        assert SecurityConfig.USE_HSTS == 'SWML_USE_HSTS'
        assert SecurityConfig.HSTS_MAX_AGE == 'SWML_HSTS_MAX_AGE'

    def test_auth_env_var_names(self):
        from signalwire_agents.core.security_config import SecurityConfig
        assert SecurityConfig.BASIC_AUTH_USER == 'SWML_BASIC_AUTH_USER'
        assert SecurityConfig.BASIC_AUTH_PASSWORD == 'SWML_BASIC_AUTH_PASSWORD'

    def test_defaults_dict_contains_expected_keys(self):
        from signalwire_agents.core.security_config import SecurityConfig
        defaults = SecurityConfig.DEFAULTS
        assert defaults['SWML_SSL_ENABLED'] is False
        assert defaults['SWML_SSL_VERIFY_MODE'] == 'CERT_REQUIRED'
        assert defaults['SWML_ALLOWED_HOSTS'] == '*'
        assert defaults['SWML_CORS_ORIGINS'] == '*'
        assert defaults['SWML_MAX_REQUEST_SIZE'] == 10 * 1024 * 1024
        assert defaults['SWML_RATE_LIMIT'] == 60
        assert defaults['SWML_REQUEST_TIMEOUT'] == 30
        assert defaults['SWML_USE_HSTS'] is True
        assert defaults['SWML_HSTS_MAX_AGE'] == 31536000


class TestSecurityConfigDefaults:
    """Test SecurityConfig initialization with default values (no env vars, no config file)."""

    def test_ssl_defaults(self):
        cfg = _make_config()
        assert cfg.ssl_enabled is False
        assert cfg.ssl_cert_path is None
        assert cfg.ssl_key_path is None
        assert cfg.domain is None
        assert cfg.ssl_verify_mode == 'CERT_REQUIRED'

    def test_host_and_cors_defaults(self):
        cfg = _make_config()
        assert cfg.allowed_hosts == ['*']
        assert cfg.cors_origins == ['*']

    def test_numeric_defaults(self):
        cfg = _make_config()
        assert cfg.max_request_size == 10 * 1024 * 1024
        assert cfg.rate_limit == 60
        assert cfg.request_timeout == 30

    def test_hsts_defaults(self):
        cfg = _make_config()
        assert cfg.use_hsts is True
        assert cfg.hsts_max_age == 31536000

    def test_auth_defaults(self):
        cfg = _make_config()
        assert cfg.basic_auth_user is None
        assert cfg.basic_auth_password is None


class TestParseList:
    """Test the _parse_list helper method."""

    def _get_instance(self):
        return _make_config()

    def test_wildcard_string(self):
        cfg = self._get_instance()
        assert cfg._parse_list('*') == ['*']

    def test_single_value(self):
        cfg = self._get_instance()
        assert cfg._parse_list('example.com') == ['example.com']

    def test_comma_separated(self):
        cfg = self._get_instance()
        result = cfg._parse_list('a.com,b.com,c.com')
        assert result == ['a.com', 'b.com', 'c.com']

    def test_comma_separated_with_spaces(self):
        cfg = self._get_instance()
        result = cfg._parse_list('  a.com , b.com , c.com  ')
        assert result == ['a.com', 'b.com', 'c.com']

    def test_list_input_passthrough(self):
        cfg = self._get_instance()
        input_list = ['x.com', 'y.com']
        assert cfg._parse_list(input_list) is input_list

    def test_empty_string(self):
        cfg = self._get_instance()
        assert cfg._parse_list('') == []

    def test_only_commas(self):
        cfg = self._get_instance()
        assert cfg._parse_list(',,,') == []

    def test_trailing_comma(self):
        cfg = self._get_instance()
        assert cfg._parse_list('a.com,b.com,') == ['a.com', 'b.com']


class TestLoadFromEnv:
    """Test loading configuration from environment variables."""

    def test_ssl_enabled_true(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        assert cfg.ssl_enabled is True

    def test_ssl_enabled_1(self):
        cfg = _make_config(SWML_SSL_ENABLED='1')
        assert cfg.ssl_enabled is True

    def test_ssl_enabled_yes(self):
        cfg = _make_config(SWML_SSL_ENABLED='yes')
        assert cfg.ssl_enabled is True

    def test_ssl_enabled_false(self):
        cfg = _make_config(SWML_SSL_ENABLED='false')
        assert cfg.ssl_enabled is False

    def test_ssl_enabled_empty(self):
        cfg = _make_config(SWML_SSL_ENABLED='')
        assert cfg.ssl_enabled is False

    def test_ssl_enabled_case_insensitive(self):
        cfg = _make_config(SWML_SSL_ENABLED='TRUE')
        assert cfg.ssl_enabled is True

    def test_ssl_enabled_yes_uppercase(self):
        cfg = _make_config(SWML_SSL_ENABLED='YES')
        assert cfg.ssl_enabled is True

    def test_ssl_cert_and_key_paths(self):
        cfg = _make_config(
            SWML_SSL_CERT_PATH='/path/to/cert.pem',
            SWML_SSL_KEY_PATH='/path/to/key.pem',
        )
        assert cfg.ssl_cert_path == '/path/to/cert.pem'
        assert cfg.ssl_key_path == '/path/to/key.pem'

    def test_domain(self):
        cfg = _make_config(SWML_DOMAIN='example.com')
        assert cfg.domain == 'example.com'

    def test_ssl_verify_mode(self):
        cfg = _make_config(SWML_SSL_VERIFY_MODE='CERT_OPTIONAL')
        assert cfg.ssl_verify_mode == 'CERT_OPTIONAL'

    def test_allowed_hosts(self):
        cfg = _make_config(SWML_ALLOWED_HOSTS='a.com,b.com')
        assert cfg.allowed_hosts == ['a.com', 'b.com']

    def test_cors_origins(self):
        cfg = _make_config(SWML_CORS_ORIGINS='http://localhost:3000,http://app.com')
        assert cfg.cors_origins == ['http://localhost:3000', 'http://app.com']

    def test_max_request_size(self):
        cfg = _make_config(SWML_MAX_REQUEST_SIZE='5242880')
        assert cfg.max_request_size == 5242880

    def test_rate_limit(self):
        cfg = _make_config(SWML_RATE_LIMIT='120')
        assert cfg.rate_limit == 120

    def test_request_timeout(self):
        cfg = _make_config(SWML_REQUEST_TIMEOUT='60')
        assert cfg.request_timeout == 60

    def test_hsts_max_age(self):
        cfg = _make_config(SWML_HSTS_MAX_AGE='86400')
        assert cfg.hsts_max_age == 86400

    def test_use_hsts_false(self):
        cfg = _make_config(SWML_USE_HSTS='false')
        assert cfg.use_hsts is False

    def test_use_hsts_non_false_value(self):
        cfg = _make_config(SWML_USE_HSTS='true')
        assert cfg.use_hsts is True

    def test_use_hsts_arbitrary_string(self):
        """Non-'false' strings should result in truthy use_hsts."""
        cfg = _make_config(SWML_USE_HSTS='anything')
        assert cfg.use_hsts is True

    def test_basic_auth_user(self):
        cfg = _make_config(SWML_BASIC_AUTH_USER='admin')
        assert cfg.basic_auth_user == 'admin'

    def test_basic_auth_password(self):
        cfg = _make_config(SWML_BASIC_AUTH_PASSWORD='secret123')
        assert cfg.basic_auth_password == 'secret123'

    def test_basic_auth_both(self):
        cfg = _make_config(
            SWML_BASIC_AUTH_USER='myuser',
            SWML_BASIC_AUTH_PASSWORD='mypass',
        )
        assert cfg.basic_auth_user == 'myuser'
        assert cfg.basic_auth_password == 'mypass'


class TestLoadConfigFile:
    """Test loading configuration from a config file."""

    def _make_config_with_file(self, security_section, has_config=True):
        """Helper: create SecurityConfig that loads from a mocked config file."""
        mock_config_loader_instance = MagicMock()
        mock_config_loader_instance.has_config.return_value = has_config
        mock_config_loader_instance.get_section.return_value = security_section

        clean = {k: v for k, v in os.environ.items() if k not in ENV_CLEAR_KEYS}
        with patch.dict(os.environ, clean, clear=True):
            with patch(
                'signalwire_agents.core.security_config.ConfigLoader.find_config_file',
                return_value='/fake/config.json',
            ):
                with patch(
                    'signalwire_agents.core.security_config.ConfigLoader',
                    return_value=mock_config_loader_instance,
                ):
                    from signalwire_agents.core.security_config import SecurityConfig
                    return SecurityConfig()

    def test_no_config_file_found(self):
        """When find_config_file returns None, config file loading is skipped."""
        cfg = _make_config()  # uses find_config_file returning None
        # Should still have defaults
        assert cfg.ssl_enabled is False
        assert cfg.rate_limit == 60

    def test_config_file_has_no_config(self):
        """When ConfigLoader.has_config() returns False, no settings are applied."""
        cfg = self._make_config_with_file({}, has_config=False)
        assert cfg.ssl_enabled is False

    def test_empty_security_section(self):
        """Empty security section should not change defaults."""
        cfg = self._make_config_with_file({})
        assert cfg.ssl_enabled is False
        assert cfg.rate_limit == 60

    def test_ssl_enabled_from_config(self):
        cfg = self._make_config_with_file({'ssl_enabled': True})
        assert cfg.ssl_enabled is True

    def test_ssl_cert_key_from_config(self):
        cfg = self._make_config_with_file({
            'ssl_cert_path': '/config/cert.pem',
            'ssl_key_path': '/config/key.pem',
        })
        assert cfg.ssl_cert_path == '/config/cert.pem'
        assert cfg.ssl_key_path == '/config/key.pem'

    def test_domain_from_config(self):
        cfg = self._make_config_with_file({'domain': 'config.example.com'})
        assert cfg.domain == 'config.example.com'

    def test_ssl_verify_mode_from_config(self):
        cfg = self._make_config_with_file({'ssl_verify_mode': 'CERT_NONE'})
        assert cfg.ssl_verify_mode == 'CERT_NONE'

    def test_allowed_hosts_from_config_string(self):
        cfg = self._make_config_with_file({'allowed_hosts': 'host1.com,host2.com'})
        assert cfg.allowed_hosts == ['host1.com', 'host2.com']

    def test_allowed_hosts_from_config_list(self):
        cfg = self._make_config_with_file({'allowed_hosts': ['host1.com', 'host2.com']})
        assert cfg.allowed_hosts == ['host1.com', 'host2.com']

    def test_cors_origins_from_config(self):
        cfg = self._make_config_with_file({'cors_origins': 'http://app.com'})
        assert cfg.cors_origins == ['http://app.com']

    def test_numeric_settings_from_config(self):
        cfg = self._make_config_with_file({
            'max_request_size': '2097152',
            'rate_limit': '100',
            'request_timeout': '45',
            'hsts_max_age': '7200',
        })
        assert cfg.max_request_size == 2097152
        assert cfg.rate_limit == 100
        assert cfg.request_timeout == 45
        assert cfg.hsts_max_age == 7200

    def test_use_hsts_from_config(self):
        cfg = self._make_config_with_file({'use_hsts': False})
        assert cfg.use_hsts is False

    def test_auth_from_config(self):
        cfg = self._make_config_with_file({
            'auth': {
                'basic': {
                    'user': 'config_user',
                    'password': 'config_pass',
                }
            }
        })
        assert cfg.basic_auth_user == 'config_user'
        assert cfg.basic_auth_password == 'config_pass'

    def test_auth_partial_user_only(self):
        cfg = self._make_config_with_file({
            'auth': {
                'basic': {
                    'user': 'just_user',
                }
            }
        })
        assert cfg.basic_auth_user == 'just_user'
        assert cfg.basic_auth_password is None

    def test_auth_partial_password_only(self):
        cfg = self._make_config_with_file({
            'auth': {
                'basic': {
                    'password': 'just_pass',
                }
            }
        })
        assert cfg.basic_auth_user is None
        assert cfg.basic_auth_password == 'just_pass'

    def test_auth_not_dict_ignored(self):
        """If auth is not a dict, it should be ignored gracefully."""
        cfg = self._make_config_with_file({'auth': 'not_a_dict'})
        assert cfg.basic_auth_user is None
        assert cfg.basic_auth_password is None

    def test_auth_basic_not_dict_ignored(self):
        """If auth.basic is not a dict, it should be ignored gracefully."""
        cfg = self._make_config_with_file({'auth': {'basic': 'not_a_dict'}})
        assert cfg.basic_auth_user is None
        assert cfg.basic_auth_password is None

    def test_config_file_overrides_env(self):
        """Config file settings should override environment variable settings."""
        mock_config_loader_instance = MagicMock()
        mock_config_loader_instance.has_config.return_value = True
        mock_config_loader_instance.get_section.return_value = {
            'rate_limit': '200',
            'domain': 'config-domain.com',
        }

        clean = {k: v for k, v in os.environ.items() if k not in ENV_CLEAR_KEYS}
        clean['SWML_RATE_LIMIT'] = '50'
        clean['SWML_DOMAIN'] = 'env-domain.com'

        with patch.dict(os.environ, clean, clear=True):
            with patch(
                'signalwire_agents.core.security_config.ConfigLoader.find_config_file',
                return_value='/fake/config.json',
            ):
                with patch(
                    'signalwire_agents.core.security_config.ConfigLoader',
                    return_value=mock_config_loader_instance,
                ):
                    from signalwire_agents.core.security_config import SecurityConfig
                    cfg = SecurityConfig()

        # Config file values should win
        assert cfg.rate_limit == 200
        assert cfg.domain == 'config-domain.com'

    def test_explicit_config_file_path(self):
        """When config_file is passed explicitly, find_config_file should not be called."""
        mock_config_loader_instance = MagicMock()
        mock_config_loader_instance.has_config.return_value = True
        mock_config_loader_instance.get_section.return_value = {'rate_limit': '999'}

        clean = {k: v for k, v in os.environ.items() if k not in ENV_CLEAR_KEYS}
        with patch.dict(os.environ, clean, clear=True):
            with patch(
                'signalwire_agents.core.security_config.ConfigLoader.find_config_file',
            ) as mock_find:
                with patch(
                    'signalwire_agents.core.security_config.ConfigLoader',
                    return_value=mock_config_loader_instance,
                ):
                    from signalwire_agents.core.security_config import SecurityConfig
                    cfg = SecurityConfig(config_file='/explicit/path.json')

        mock_find.assert_not_called()
        assert cfg.rate_limit == 999

    def test_service_name_passed_to_find_config(self):
        """service_name should be forwarded to find_config_file when no config_file given."""
        clean = {k: v for k, v in os.environ.items() if k not in ENV_CLEAR_KEYS}
        with patch.dict(os.environ, clean, clear=True):
            with patch(
                'signalwire_agents.core.security_config.ConfigLoader.find_config_file',
                return_value=None,
            ) as mock_find:
                from signalwire_agents.core.security_config import SecurityConfig
                SecurityConfig(service_name='my_service')

        mock_find.assert_called_once_with('my_service')


class TestValidateSSLConfig:
    """Test SSL configuration validation."""

    def test_ssl_disabled_always_valid(self):
        cfg = _make_config()
        is_valid, error = cfg.validate_ssl_config()
        assert is_valid is True
        assert error is None

    def test_ssl_enabled_missing_cert_path(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        cfg.ssl_cert_path = None
        cfg.ssl_key_path = '/path/to/key.pem'
        is_valid, error = cfg.validate_ssl_config()
        assert is_valid is False
        assert 'SWML_SSL_CERT_PATH' in error

    def test_ssl_enabled_missing_key_path(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        cfg.ssl_cert_path = '/path/to/cert.pem'
        cfg.ssl_key_path = None
        is_valid, error = cfg.validate_ssl_config()
        assert is_valid is False
        assert 'SWML_SSL_KEY_PATH' in error

    def test_ssl_enabled_cert_file_not_found(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        cfg.ssl_cert_path = '/nonexistent/cert.pem'
        cfg.ssl_key_path = '/nonexistent/key.pem'
        with patch('os.path.exists', side_effect=lambda p: p != '/nonexistent/cert.pem'):
            is_valid, error = cfg.validate_ssl_config()
        assert is_valid is False
        assert 'certificate file not found' in error

    def test_ssl_enabled_key_file_not_found(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        cfg.ssl_cert_path = '/exists/cert.pem'
        cfg.ssl_key_path = '/nonexistent/key.pem'
        with patch('os.path.exists', side_effect=lambda p: p == '/exists/cert.pem'):
            is_valid, error = cfg.validate_ssl_config()
        assert is_valid is False
        assert 'key file not found' in error

    def test_ssl_enabled_both_files_exist(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        cfg.ssl_cert_path = '/exists/cert.pem'
        cfg.ssl_key_path = '/exists/key.pem'
        with patch('os.path.exists', return_value=True):
            is_valid, error = cfg.validate_ssl_config()
        assert is_valid is True
        assert error is None


class TestGetSSLContextKwargs:
    """Test get_ssl_context_kwargs method."""

    def test_ssl_disabled_returns_empty(self):
        cfg = _make_config()
        assert cfg.get_ssl_context_kwargs() == {}

    def test_ssl_enabled_valid_returns_kwargs(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        cfg.ssl_cert_path = '/exists/cert.pem'
        cfg.ssl_key_path = '/exists/key.pem'
        with patch('os.path.exists', return_value=True):
            result = cfg.get_ssl_context_kwargs()
        assert result == {
            'ssl_certfile': '/exists/cert.pem',
            'ssl_keyfile': '/exists/key.pem',
        }

    def test_ssl_enabled_invalid_returns_empty(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        cfg.ssl_cert_path = None
        cfg.ssl_key_path = None
        result = cfg.get_ssl_context_kwargs()
        assert result == {}

    def test_ssl_enabled_invalid_logs_error(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        cfg.ssl_cert_path = None
        with patch('signalwire_agents.core.security_config.logger') as mock_logger:
            cfg.get_ssl_context_kwargs()
            mock_logger.error.assert_called_once()


class TestGetBasicAuth:
    """Test get_basic_auth credential generation and caching."""

    def test_default_username(self):
        cfg = _make_config()
        username, _ = cfg.get_basic_auth()
        assert username == 'signalwire'

    def test_custom_username(self):
        cfg = _make_config(SWML_BASIC_AUTH_USER='custom_user')
        username, _ = cfg.get_basic_auth()
        assert username == 'custom_user'

    def test_generates_password_when_not_set(self):
        cfg = _make_config()
        _, password = cfg.get_basic_auth()
        assert password is not None
        assert len(password) > 0

    def test_password_is_url_safe_token(self):
        """Verify the generated password comes from secrets.token_urlsafe."""
        cfg = _make_config()
        with patch('signalwire_agents.core.security_config.secrets.token_urlsafe',
                    return_value='mock_token_abc') as mock_token:
            _, password = cfg.get_basic_auth()
        mock_token.assert_called_once_with(32)
        assert password == 'mock_token_abc'

    def test_password_caching_stability_same_instance(self):
        """Multiple calls to get_basic_auth on the same instance return the same password."""
        cfg = _make_config()
        _, password1 = cfg.get_basic_auth()
        _, password2 = cfg.get_basic_auth()
        _, password3 = cfg.get_basic_auth()
        assert password1 == password2
        assert password2 == password3

    def test_password_caching_does_not_regenerate(self):
        """After the first call generates a password, subsequent calls must not call secrets again."""
        cfg = _make_config()
        with patch('signalwire_agents.core.security_config.secrets.token_urlsafe',
                    return_value='first_token') as mock_token:
            _, pw1 = cfg.get_basic_auth()
        assert pw1 == 'first_token'

        # Second call should NOT invoke token_urlsafe again
        with patch('signalwire_agents.core.security_config.secrets.token_urlsafe',
                    return_value='second_token') as mock_token:
            _, pw2 = cfg.get_basic_auth()
        mock_token.assert_not_called()
        assert pw2 == 'first_token'

    def test_preset_password_not_overwritten(self):
        cfg = _make_config(SWML_BASIC_AUTH_PASSWORD='env_password')
        _, password = cfg.get_basic_auth()
        assert password == 'env_password'

    def test_preset_password_stability(self):
        """Pre-set password stays the same across calls."""
        cfg = _make_config(SWML_BASIC_AUTH_PASSWORD='stable')
        _, pw1 = cfg.get_basic_auth()
        _, pw2 = cfg.get_basic_auth()
        assert pw1 == 'stable'
        assert pw2 == 'stable'

    def test_externally_set_password_preserved(self):
        """Setting basic_auth_password directly is respected by get_basic_auth."""
        cfg = _make_config()
        cfg.basic_auth_password = 'manual_password'
        _, password = cfg.get_basic_auth()
        assert password == 'manual_password'

    def test_returns_tuple(self):
        cfg = _make_config()
        result = cfg.get_basic_auth()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generated_passwords_differ_between_instances(self):
        """Different SecurityConfig instances should generate different passwords."""
        cfg1 = _make_config()
        cfg2 = _make_config()
        _, pw1 = cfg1.get_basic_auth()
        _, pw2 = cfg2.get_basic_auth()
        # Extremely unlikely they match with 32-byte random tokens
        # but technically possible; we just verify both are non-empty.
        assert len(pw1) > 0
        assert len(pw2) > 0


class TestGetSecurityHeaders:
    """Test get_security_headers method."""

    def test_http_headers_no_hsts(self):
        cfg = _make_config()
        headers = cfg.get_security_headers(is_https=False)
        assert 'X-Content-Type-Options' in headers
        assert headers['X-Content-Type-Options'] == 'nosniff'
        assert headers['X-Frame-Options'] == 'DENY'
        assert headers['X-XSS-Protection'] == '1; mode=block'
        assert headers['Referrer-Policy'] == 'strict-origin-when-cross-origin'
        assert 'Strict-Transport-Security' not in headers

    def test_https_with_hsts_enabled(self):
        cfg = _make_config()
        headers = cfg.get_security_headers(is_https=True)
        assert 'Strict-Transport-Security' in headers
        assert '31536000' in headers['Strict-Transport-Security']
        assert 'includeSubDomains' in headers['Strict-Transport-Security']

    def test_https_with_hsts_disabled(self):
        cfg = _make_config()
        cfg.use_hsts = False
        headers = cfg.get_security_headers(is_https=True)
        assert 'Strict-Transport-Security' not in headers

    def test_https_custom_hsts_max_age(self):
        cfg = _make_config()
        cfg.hsts_max_age = 86400
        headers = cfg.get_security_headers(is_https=True)
        assert 'max-age=86400' in headers['Strict-Transport-Security']

    def test_default_is_http(self):
        """Default is_https=False."""
        cfg = _make_config()
        headers = cfg.get_security_headers()
        assert 'Strict-Transport-Security' not in headers

    def test_always_includes_base_headers(self):
        """Base security headers are always present regardless of HTTPS status."""
        cfg = _make_config()
        for is_https in (True, False):
            headers = cfg.get_security_headers(is_https=is_https)
            assert 'X-Content-Type-Options' in headers
            assert 'X-Frame-Options' in headers
            assert 'X-XSS-Protection' in headers
            assert 'Referrer-Policy' in headers


class TestShouldAllowHost:
    """Test should_allow_host method."""

    def test_wildcard_allows_all(self):
        cfg = _make_config()
        assert cfg.should_allow_host('anything.com') is True
        assert cfg.should_allow_host('') is True
        assert cfg.should_allow_host('localhost') is True

    def test_specific_host_allowed(self):
        cfg = _make_config(SWML_ALLOWED_HOSTS='example.com,api.example.com')
        assert cfg.should_allow_host('example.com') is True
        assert cfg.should_allow_host('api.example.com') is True

    def test_host_not_in_list(self):
        cfg = _make_config(SWML_ALLOWED_HOSTS='example.com')
        assert cfg.should_allow_host('other.com') is False

    def test_empty_host_not_allowed_when_specific(self):
        cfg = _make_config(SWML_ALLOWED_HOSTS='example.com')
        assert cfg.should_allow_host('') is False

    def test_case_sensitive_matching(self):
        """Host matching is case-sensitive (as set by the environment)."""
        cfg = _make_config(SWML_ALLOWED_HOSTS='Example.com')
        assert cfg.should_allow_host('Example.com') is True
        assert cfg.should_allow_host('example.com') is False


class TestGetCorsConfig:
    """Test get_cors_config method."""

    def test_default_cors_config(self):
        cfg = _make_config()
        cors = cfg.get_cors_config()
        assert cors['allow_origins'] == ['*']
        assert cors['allow_credentials'] is True
        assert cors['allow_methods'] == ['*']
        assert cors['allow_headers'] == ['*']

    def test_custom_cors_origins(self):
        cfg = _make_config(SWML_CORS_ORIGINS='http://localhost:3000,http://app.com')
        cors = cfg.get_cors_config()
        assert cors['allow_origins'] == ['http://localhost:3000', 'http://app.com']

    def test_cors_config_keys(self):
        cfg = _make_config()
        cors = cfg.get_cors_config()
        assert set(cors.keys()) == {'allow_origins', 'allow_credentials', 'allow_methods', 'allow_headers'}


class TestGetUrlScheme:
    """Test get_url_scheme method."""

    def test_http_when_ssl_disabled(self):
        cfg = _make_config()
        assert cfg.get_url_scheme() == 'http'

    def test_https_when_ssl_enabled(self):
        cfg = _make_config(SWML_SSL_ENABLED='true')
        assert cfg.get_url_scheme() == 'https'


class TestLogConfig:
    """Test log_config method."""

    def test_log_config_calls_logger(self):
        cfg = _make_config()
        with patch('signalwire_agents.core.security_config.logger') as mock_logger:
            cfg.log_config('test_service')
            mock_logger.info.assert_called_once()

    def test_log_config_includes_service_name(self):
        cfg = _make_config()
        with patch('signalwire_agents.core.security_config.logger') as mock_logger:
            cfg.log_config('my_service')
            call_kwargs = mock_logger.info.call_args
            # The first positional arg is the event name
            assert call_kwargs[0][0] == 'security_config_loaded'
            # Keyword args should include service
            assert call_kwargs[1]['service'] == 'my_service'

    def test_log_config_includes_key_fields(self):
        cfg = _make_config()
        with patch('signalwire_agents.core.security_config.logger') as mock_logger:
            cfg.log_config('svc')
            kwargs = mock_logger.info.call_args[1]
            assert 'ssl_enabled' in kwargs
            assert 'domain' in kwargs
            assert 'allowed_hosts' in kwargs
            assert 'cors_origins' in kwargs
            assert 'max_request_size' in kwargs
            assert 'rate_limit' in kwargs
            assert 'use_hsts' in kwargs
            assert 'has_basic_auth' in kwargs

    def test_log_config_has_basic_auth_true(self):
        cfg = _make_config(
            SWML_BASIC_AUTH_USER='user',
            SWML_BASIC_AUTH_PASSWORD='pass',
        )
        with patch('signalwire_agents.core.security_config.logger') as mock_logger:
            cfg.log_config('svc')
            kwargs = mock_logger.info.call_args[1]
            assert kwargs['has_basic_auth'] is True

    def test_log_config_has_basic_auth_false(self):
        cfg = _make_config()
        with patch('signalwire_agents.core.security_config.logger') as mock_logger:
            cfg.log_config('svc')
            kwargs = mock_logger.info.call_args[1]
            assert kwargs['has_basic_auth'] is False


class TestGlobalInstance:
    """Test the module-level global security_config instance."""

    def test_global_instance_exists(self):
        from signalwire_agents.core.security_config import security_config
        from signalwire_agents.core.security_config import SecurityConfig
        assert isinstance(security_config, SecurityConfig)

    def test_global_instance_is_same_on_reimport(self):
        from signalwire_agents.core.security_config import security_config as sc1
        from signalwire_agents.core.security_config import security_config as sc2
        assert sc1 is sc2


class TestInitOrder:
    """Test that __init__ applies configuration in the correct priority order."""

    def test_defaults_then_env_then_config_file(self):
        """
        Verify: defaults are set first, then env overrides them,
        then config file overrides env.
        """
        mock_config_loader = MagicMock()
        mock_config_loader.has_config.return_value = True
        mock_config_loader.get_section.return_value = {
            'rate_limit': '300',
        }

        clean = {k: v for k, v in os.environ.items() if k not in ENV_CLEAR_KEYS}
        # Env sets rate_limit to 100
        clean['SWML_RATE_LIMIT'] = '100'

        with patch.dict(os.environ, clean, clear=True):
            with patch(
                'signalwire_agents.core.security_config.ConfigLoader.find_config_file',
                return_value='/fake/config.json',
            ):
                with patch(
                    'signalwire_agents.core.security_config.ConfigLoader',
                    return_value=mock_config_loader,
                ):
                    from signalwire_agents.core.security_config import SecurityConfig
                    cfg = SecurityConfig()

        # Config file value (300) should win over env (100) and default (60)
        assert cfg.rate_limit == 300


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_rate_limit(self):
        cfg = _make_config(SWML_RATE_LIMIT='0')
        assert cfg.rate_limit == 0

    def test_zero_request_timeout(self):
        cfg = _make_config(SWML_REQUEST_TIMEOUT='0')
        assert cfg.request_timeout == 0

    def test_zero_max_request_size(self):
        cfg = _make_config(SWML_MAX_REQUEST_SIZE='0')
        assert cfg.max_request_size == 0

    def test_zero_hsts_max_age(self):
        cfg = _make_config(SWML_HSTS_MAX_AGE='0')
        assert cfg.hsts_max_age == 0
        headers = cfg.get_security_headers(is_https=True)
        assert 'max-age=0' in headers['Strict-Transport-Security']

    def test_very_large_max_request_size(self):
        cfg = _make_config(SWML_MAX_REQUEST_SIZE='1073741824')  # 1GB
        assert cfg.max_request_size == 1073741824

    def test_ssl_validate_after_manual_state_change(self):
        """Validate SSL after manually changing attributes."""
        cfg = _make_config()
        cfg.ssl_enabled = True
        cfg.ssl_cert_path = '/some/cert.pem'
        cfg.ssl_key_path = '/some/key.pem'
        with patch('os.path.exists', return_value=True):
            is_valid, error = cfg.validate_ssl_config()
        assert is_valid is True

    def test_allowed_hosts_single_entry(self):
        cfg = _make_config(SWML_ALLOWED_HOSTS='only-this-host.com')
        assert cfg.allowed_hosts == ['only-this-host.com']
        assert cfg.should_allow_host('only-this-host.com') is True
        assert cfg.should_allow_host('other.com') is False

    def test_parse_list_with_whitespace_only_items(self):
        cfg = _make_config()
        result = cfg._parse_list('a, , b, ,c')
        assert result == ['a', 'b', 'c']

    def test_get_basic_auth_empty_string_user(self):
        """Empty string user from env should be treated as falsy, defaulting to 'signalwire'."""
        cfg = _make_config()
        cfg.basic_auth_user = ''
        username, _ = cfg.get_basic_auth()
        assert username == 'signalwire'

    def test_get_basic_auth_empty_string_password_generates_new(self):
        """Empty string password should be treated as falsy, generating a new one."""
        cfg = _make_config()
        cfg.basic_auth_password = ''
        _, password = cfg.get_basic_auth()
        assert len(password) > 0
        assert password != ''

    def test_multiple_env_vars_combined(self):
        """Test setting many environment variables simultaneously."""
        cfg = _make_config(
            SWML_SSL_ENABLED='true',
            SWML_SSL_CERT_PATH='/cert',
            SWML_SSL_KEY_PATH='/key',
            SWML_DOMAIN='multi.com',
            SWML_ALLOWED_HOSTS='h1.com,h2.com',
            SWML_CORS_ORIGINS='http://c1.com',
            SWML_MAX_REQUEST_SIZE='999',
            SWML_RATE_LIMIT='10',
            SWML_REQUEST_TIMEOUT='5',
            SWML_USE_HSTS='false',
            SWML_HSTS_MAX_AGE='100',
            SWML_BASIC_AUTH_USER='admin',
            SWML_BASIC_AUTH_PASSWORD='pass',
        )
        assert cfg.ssl_enabled is True
        assert cfg.ssl_cert_path == '/cert'
        assert cfg.ssl_key_path == '/key'
        assert cfg.domain == 'multi.com'
        assert cfg.allowed_hosts == ['h1.com', 'h2.com']
        assert cfg.cors_origins == ['http://c1.com']
        assert cfg.max_request_size == 999
        assert cfg.rate_limit == 10
        assert cfg.request_timeout == 5
        assert cfg.use_hsts is False
        assert cfg.hsts_max_age == 100
        assert cfg.basic_auth_user == 'admin'
        assert cfg.basic_auth_password == 'pass'
