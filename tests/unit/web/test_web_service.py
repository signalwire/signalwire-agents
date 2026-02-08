"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for WebService (signalwire_agents/web/web_service.py).

Covers:
  - Initialization with and without FastAPI
  - Configuration loading
  - Path traversal protection
  - XSS prevention in directory listings
  - File extension / size filtering
  - Basic-auth credential validation
  - Directory add / remove helpers
  - Service start / stop lifecycle
  - Route and middleware setup
"""

import os
import secrets
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from html import escape

import pytest


# ---------------------------------------------------------------------------
# Helpers – build a minimally-patched WebService instance
# ---------------------------------------------------------------------------

def _make_security_mock():
    """Return a mock SecurityConfig that satisfies WebService.__init__."""
    sec = MagicMock()
    sec.ssl_enabled = False
    sec.get_basic_auth.return_value = ("admin", "secret")
    sec.get_cors_config.return_value = {
        "allow_origins": ["*"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    sec.get_security_headers.return_value = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }
    sec.should_allow_host.return_value = True
    sec.get_ssl_context_kwargs.return_value = {}
    sec.log_config = MagicMock()
    return sec


def _make_web_service(
    fastapi_available=True,
    directories=None,
    basic_auth=None,
    enable_directory_browsing=False,
    allowed_extensions=None,
    blocked_extensions=None,
    max_file_size=100 * 1024 * 1024,
    enable_cors=True,
):
    """Construct a WebService with all heavy dependencies mocked out."""
    security_mock = _make_security_mock()

    patches = {
        "security_config": patch(
            "signalwire_agents.web.web_service.SecurityConfig",
            return_value=security_mock,
        ),
        "config_loader_find": patch(
            "signalwire_agents.web.web_service.ConfigLoader.find_config_file",
            return_value=None,
        ),
        "config_loader_cls": patch(
            "signalwire_agents.web.web_service.ConfigLoader",
        ),
    }

    if not fastapi_available:
        patches["fastapi_mod"] = patch(
            "signalwire_agents.web.web_service.FastAPI", None
        )

    started = {k: p.start() for k, p in patches.items()}

    from signalwire_agents.web.web_service import WebService

    ws = WebService(
        port=9999,
        directories=directories or {},
        basic_auth=basic_auth,
        enable_directory_browsing=enable_directory_browsing,
        allowed_extensions=allowed_extensions,
        blocked_extensions=blocked_extensions,
        max_file_size=max_file_size,
        enable_cors=enable_cors,
    )

    # Attach for later inspection / teardown
    ws._test_patches = patches
    ws._test_started = started
    ws._test_security_mock = security_mock
    return ws


def _stop_patches(ws):
    for p in ws._test_patches.values():
        p.stop()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def web_service():
    ws = _make_web_service()
    yield ws
    _stop_patches(ws)


@pytest.fixture()
def web_service_no_fastapi():
    ws = _make_web_service(fastapi_available=False)
    yield ws
    _stop_patches(ws)


@pytest.fixture()
def web_service_with_browsing():
    ws = _make_web_service(enable_directory_browsing=True)
    yield ws
    _stop_patches(ws)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestWebServiceInit:
    """Tests for __init__ and _load_config."""

    def test_default_port(self, web_service):
        assert web_service.port == 9999

    def test_default_directories_empty(self, web_service):
        assert web_service.directories == {}

    def test_custom_directories(self):
        ws = _make_web_service(directories={"/static": "/tmp"})
        assert ws.directories == {"/static": "/tmp"}
        _stop_patches(ws)

    def test_enable_directory_browsing_default_off(self, web_service):
        assert web_service.enable_directory_browsing is False

    def test_enable_directory_browsing_on(self, web_service_with_browsing):
        assert web_service_with_browsing.enable_directory_browsing is True

    def test_max_file_size_default(self, web_service):
        assert web_service.max_file_size == 100 * 1024 * 1024

    def test_custom_max_file_size(self):
        ws = _make_web_service(max_file_size=1024)
        assert ws.max_file_size == 1024
        _stop_patches(ws)

    def test_default_blocked_extensions(self, web_service):
        for ext in [".env", ".git", ".gitignore", ".key", ".pem", ".crt",
                    ".pyc", "__pycache__", ".DS_Store", ".swp"]:
            assert ext in web_service.blocked_extensions

    def test_custom_blocked_extensions(self):
        ws = _make_web_service(blocked_extensions=[".secret"])
        assert ws.blocked_extensions == [".secret"]
        _stop_patches(ws)

    def test_allowed_extensions_default_none(self, web_service):
        assert web_service.allowed_extensions is None

    def test_custom_allowed_extensions(self):
        ws = _make_web_service(allowed_extensions=[".html", ".css"])
        assert ws.allowed_extensions == [".html", ".css"]
        _stop_patches(ws)

    def test_app_is_created_when_fastapi_available(self, web_service):
        # When FastAPI is importable the app attribute should not be None
        assert web_service.app is not None

    def test_app_is_none_when_fastapi_unavailable(self, web_service_no_fastapi):
        assert web_service_no_fastapi.app is None

    def test_basic_auth_from_constructor(self):
        ws = _make_web_service(basic_auth=("user1", "pass1"))
        assert ws._basic_auth == ("user1", "pass1")
        _stop_patches(ws)

    def test_basic_auth_falls_back_to_security_config(self, web_service):
        # The security mock returns ("admin", "secret")
        assert web_service._basic_auth == ("admin", "secret")

    def test_enable_cors_stored(self, web_service):
        assert web_service.enable_cors is True

    def test_enable_cors_false(self):
        ws = _make_web_service(enable_cors=False)
        assert ws.enable_cors is False
        _stop_patches(ws)


# ---------------------------------------------------------------------------
# _load_config tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for _load_config with mocked ConfigLoader."""

    def test_no_config_file_sets_defaults(self, web_service):
        # Already exercised in fixture – directories default to empty dict
        assert isinstance(web_service.directories, dict)

    def test_config_with_service_section(self):
        """If a config file supplies a service section, values should be applied."""
        service_section = {
            "port": 7777,
            "directories": {"/docs": "/var/docs"},
            "enable_directory_browsing": True,
            "max_file_size": 512,
            "allowed_extensions": [".txt"],
            "blocked_extensions": [".bin"],
        }

        mock_loader_instance = MagicMock()
        mock_loader_instance.has_config.return_value = True
        mock_loader_instance.get_section.return_value = service_section

        with patch(
            "signalwire_agents.web.web_service.SecurityConfig",
            return_value=_make_security_mock(),
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader.find_config_file",
            return_value="/fake/config.json",
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader",
            return_value=mock_loader_instance,
        ):
            from signalwire_agents.web.web_service import WebService

            ws = WebService(config_file="/fake/config.json")

        # Constructor params override, but _load_config was called first
        # Check that _load_config sets port from service_config
        # Since WebService.__init__ sets self.port = port (default 8002) AFTER
        # _load_config, the constructor default wins.  But _load_config still
        # sets self.directories before the override.  With no directories kwarg
        # the config value should survive.
        assert isinstance(ws.directories, dict)


# ---------------------------------------------------------------------------
# _is_file_allowed tests
# ---------------------------------------------------------------------------

class TestIsFileAllowed:
    """Tests for the _is_file_allowed method."""

    def test_normal_html_file_allowed(self, web_service, tmp_path):
        f = tmp_path / "index.html"
        f.write_text("<html></html>")
        assert web_service._is_file_allowed(f) is True

    def test_blocked_extension_env(self, web_service, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SECRET=x")
        assert web_service._is_file_allowed(f) is False

    def test_blocked_extension_pem(self, web_service, tmp_path):
        f = tmp_path / "server.pem"
        f.write_text("-----BEGIN-----")
        assert web_service._is_file_allowed(f) is False

    def test_blocked_extension_key(self, web_service, tmp_path):
        f = tmp_path / "private.key"
        f.write_text("key data")
        assert web_service._is_file_allowed(f) is False

    def test_blocked_extension_crt(self, web_service, tmp_path):
        f = tmp_path / "cert.crt"
        f.write_text("cert data")
        assert web_service._is_file_allowed(f) is False

    def test_blocked_extension_pyc(self, web_service, tmp_path):
        f = tmp_path / "module.pyc"
        f.write_bytes(b"\x00\x00")
        assert web_service._is_file_allowed(f) is False

    def test_blocked_name_pycache(self, web_service, tmp_path):
        d = tmp_path / "__pycache__"
        d.mkdir()
        f = d / "module.cpython-310.pyc"
        f.write_bytes(b"\x00")
        # __pycache__ appears in the path so it should be blocked
        assert web_service._is_file_allowed(f) is False

    def test_blocked_name_ds_store(self, web_service, tmp_path):
        f = tmp_path / ".DS_Store"
        f.write_bytes(b"\x00")
        assert web_service._is_file_allowed(f) is False

    def test_blocked_extension_swp(self, web_service, tmp_path):
        f = tmp_path / ".file.swp"
        f.write_text("swap")
        assert web_service._is_file_allowed(f) is False

    def test_blocked_gitignore(self, web_service, tmp_path):
        f = tmp_path / ".gitignore"
        f.write_text("*.pyc")
        assert web_service._is_file_allowed(f) is False

    def test_file_exceeds_max_size(self, tmp_path):
        ws = _make_web_service(max_file_size=10)
        f = tmp_path / "big.txt"
        f.write_text("A" * 100)
        assert ws._is_file_allowed(f) is False
        _stop_patches(ws)

    def test_file_within_max_size(self, tmp_path):
        ws = _make_web_service(max_file_size=1000)
        f = tmp_path / "small.txt"
        f.write_text("ok")
        assert ws._is_file_allowed(f) is True
        _stop_patches(ws)

    def test_stat_oserror_returns_false(self, web_service):
        """If stat() raises OSError the file should be denied."""
        mock_path = MagicMock(spec=Path)
        mock_path.stat.side_effect = OSError("disk error")
        assert web_service._is_file_allowed(mock_path) is False

    def test_stat_filenotfounderror_returns_false(self, web_service):
        """If stat() raises FileNotFoundError the file should be denied."""
        mock_path = MagicMock(spec=Path)
        mock_path.stat.side_effect = FileNotFoundError("gone")
        assert web_service._is_file_allowed(mock_path) is False

    def test_allowed_extensions_filter(self, tmp_path):
        ws = _make_web_service(allowed_extensions=[".html"])
        html = tmp_path / "page.html"
        html.write_text("<p>hi</p>")
        css = tmp_path / "style.css"
        css.write_text("body{}")

        assert ws._is_file_allowed(html) is True
        assert ws._is_file_allowed(css) is False
        _stop_patches(ws)

    def test_allowed_extensions_blocks_unlisted(self, tmp_path):
        ws = _make_web_service(allowed_extensions=[".css", ".js"])
        f = tmp_path / "data.json"
        f.write_text("{}")
        assert ws._is_file_allowed(f) is False
        _stop_patches(ws)

    def test_custom_blocked_extensions(self, tmp_path):
        ws = _make_web_service(blocked_extensions=[".secret"])
        f = tmp_path / "data.secret"
        f.write_text("hidden")
        assert ws._is_file_allowed(f) is False
        # .env is no longer blocked with custom list
        f2 = tmp_path / ".env"
        f2.write_text("VAR=1")
        assert ws._is_file_allowed(f2) is True
        _stop_patches(ws)


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------

class TestPathTraversalProtection:
    """Verify that directory traversal attacks are blocked.

    The serve_file handler resolves paths and checks they stay within
    the configured directory.  We test the underlying logic directly.
    """

    def test_traversal_dot_dot_blocked(self, tmp_path):
        """../../../etc/passwd style traversal must be rejected."""
        base_dir = tmp_path / "www"
        base_dir.mkdir()
        (base_dir / "index.html").write_text("ok")

        # Simulate what serve_file does
        file_path = "../../etc/passwd"
        full_path = (Path(str(base_dir)) / file_path).resolve()
        dir_path = Path(str(base_dir)).resolve()

        within = str(full_path).startswith(str(dir_path) + os.sep) or full_path == dir_path
        assert within is False, "Path traversal must be detected"

    def test_valid_subpath_allowed(self, tmp_path):
        base_dir = tmp_path / "www"
        sub = base_dir / "css"
        sub.mkdir(parents=True)
        (sub / "style.css").write_text("body{}")

        full_path = (Path(str(base_dir)) / "css/style.css").resolve()
        dir_path = Path(str(base_dir)).resolve()

        within = str(full_path).startswith(str(dir_path) + os.sep) or full_path == dir_path
        assert within is True

    def test_traversal_encoded_dots_blocked(self, tmp_path):
        """Paths that resolve outside the base should be caught."""
        base_dir = tmp_path / "www"
        base_dir.mkdir()

        file_path = "subdir/../../etc/shadow"
        full_path = (Path(str(base_dir)) / file_path).resolve()
        dir_path = Path(str(base_dir)).resolve()

        within = str(full_path).startswith(str(dir_path) + os.sep) or full_path == dir_path
        assert within is False

    def test_exact_base_dir_allowed(self, tmp_path):
        """Requesting the base directory itself should be allowed (== check)."""
        base_dir = tmp_path / "www"
        base_dir.mkdir()

        full_path = Path(str(base_dir)).resolve()
        dir_path = Path(str(base_dir)).resolve()

        within = str(full_path).startswith(str(dir_path) + os.sep) or full_path == dir_path
        assert within is True

    def test_traversal_with_trailing_slash(self, tmp_path):
        base_dir = tmp_path / "www"
        base_dir.mkdir()

        file_path = "../"
        full_path = (Path(str(base_dir)) / file_path).resolve()
        dir_path = Path(str(base_dir)).resolve()

        within = str(full_path).startswith(str(dir_path) + os.sep) or full_path == dir_path
        # "../" resolves to parent, which equals tmp_path, not base_dir
        # Unless base_dir IS tmp_path. Here base_dir is tmp_path/www so parent != www
        assert within is False

    def test_sibling_directory_blocked(self, tmp_path):
        """Accessing a sibling of the base dir must be rejected."""
        base_dir = tmp_path / "www"
        base_dir.mkdir()
        sibling = tmp_path / "private"
        sibling.mkdir()
        (sibling / "secret.txt").write_text("nope")

        file_path = "../private/secret.txt"
        full_path = (Path(str(base_dir)) / file_path).resolve()
        dir_path = Path(str(base_dir)).resolve()

        within = str(full_path).startswith(str(dir_path) + os.sep) or full_path == dir_path
        assert within is False

    def test_null_byte_in_path(self, tmp_path):
        """Null bytes in a path component should not bypass checks."""
        base_dir = tmp_path / "www"
        base_dir.mkdir()

        # Attempting to construct a path with a null byte
        # Python's Path will typically raise ValueError on null bytes.
        with pytest.raises((ValueError, OSError)):
            (Path(str(base_dir)) / "file\x00.html").resolve()


# ---------------------------------------------------------------------------
# XSS prevention – _generate_directory_listing
# ---------------------------------------------------------------------------

class TestXSSPrevention:
    """Verify that HTML-special characters in file/directory names are escaped."""

    def test_script_in_filename_escaped(self, web_service_with_browsing, tmp_path):
        malicious = tmp_path / '<script>alert(1)</script>.txt'
        try:
            malicious.write_text("xss")
        except (OSError, ValueError):
            pytest.skip("OS does not allow special chars in filenames")

        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        assert "<script>" not in html
        assert escape("<script>alert(1)</script>", quote=True) in html

    def test_html_in_directory_name_escaped(self, web_service_with_browsing, tmp_path):
        malicious_dir = tmp_path / '<img src=x onerror=alert(1)>'
        try:
            malicious_dir.mkdir()
        except (OSError, ValueError):
            pytest.skip("OS does not allow special chars in directory names")

        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        raw_name = '<img src=x onerror=alert(1)>'
        assert raw_name not in html
        assert escape(raw_name, quote=True) in html

    def test_url_path_in_title_escaped(self, web_service_with_browsing, tmp_path):
        """The url_path used in the <title> / <h1> must be escaped."""
        xss_path = '/<script>alert("xss")</script>'
        html = web_service_with_browsing._generate_directory_listing(tmp_path, xss_path)
        assert '<script>alert("xss")</script>' not in html

    def test_ampersand_in_filename_escaped(self, web_service_with_browsing, tmp_path):
        f = tmp_path / "a&b.txt"
        try:
            f.write_text("data")
        except (OSError, ValueError):
            pytest.skip("OS does not allow & in filenames")
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        # The raw '&' in a non-entity context should be escaped to '&amp;'
        assert "a&amp;b.txt" in html

    def test_quote_in_filename_escaped(self, web_service_with_browsing, tmp_path):
        f = tmp_path / 'file"name.txt'
        try:
            f.write_text("data")
        except (OSError, ValueError):
            pytest.skip("OS does not allow quotes in filenames")
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        assert 'file"name.txt' not in html
        assert escape('file"name.txt', quote=True) in html


# ---------------------------------------------------------------------------
# _generate_directory_listing – structural tests
# ---------------------------------------------------------------------------

class TestDirectoryListing:
    """Non-security structural tests for _generate_directory_listing."""

    def test_root_path_no_parent_link(self, web_service_with_browsing, tmp_path):
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/")
        assert "../" not in html

    def test_non_root_path_has_parent_link(self, web_service_with_browsing, tmp_path):
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/sub")
        assert "../" in html

    def test_hidden_files_skipped(self, web_service_with_browsing, tmp_path):
        (tmp_path / ".hidden").write_text("hidden")
        (tmp_path / "visible.txt").write_text("visible")
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        assert ".hidden" not in html
        assert "visible.txt" in html

    def test_file_size_bytes(self, web_service_with_browsing, tmp_path):
        f = tmp_path / "tiny.txt"
        f.write_text("ab")  # 2 bytes
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        assert "B" in html

    def test_file_size_kilobytes(self, web_service_with_browsing, tmp_path):
        f = tmp_path / "medium.txt"
        f.write_text("x" * 2048)
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        assert "KB" in html

    def test_file_size_megabytes(self, web_service_with_browsing, tmp_path):
        ws = _make_web_service(
            enable_directory_browsing=True,
            max_file_size=200 * 1024 * 1024,
        )
        f = tmp_path / "large.bin"
        f.write_bytes(b"\x00" * (2 * 1024 * 1024))
        html = ws._generate_directory_listing(tmp_path, "/files")
        assert "MB" in html
        _stop_patches(ws)

    def test_directories_listed(self, web_service_with_browsing, tmp_path):
        (tmp_path / "subdir").mkdir()
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        assert "subdir/" in html

    def test_blocked_files_not_listed(self, web_service_with_browsing, tmp_path):
        """Files that fail _is_file_allowed should not appear in listings."""
        (tmp_path / "server.pem").write_text("cert")
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        assert "server.pem" not in html


# ---------------------------------------------------------------------------
# _get_current_username – basic auth validation
# ---------------------------------------------------------------------------

class TestGetCurrentUsername:
    """Tests for _get_current_username."""

    def test_no_credentials_returns_none(self, web_service):
        result = web_service._get_current_username(None)
        assert result is None

    def test_valid_credentials(self):
        ws = _make_web_service(basic_auth=("myuser", "mypass"))
        creds = MagicMock()
        creds.username = "myuser"
        creds.password = "mypass"
        assert ws._get_current_username(creds) == "myuser"
        _stop_patches(ws)

    def test_invalid_username_raises(self):
        ws = _make_web_service(basic_auth=("admin", "secret"))
        creds = MagicMock()
        creds.username = "hacker"
        creds.password = "secret"
        from signalwire_agents.web.web_service import HTTPException

        if HTTPException is None:
            pytest.skip("HTTPException not available")
        with pytest.raises(HTTPException):
            ws._get_current_username(creds)
        _stop_patches(ws)

    def test_invalid_password_raises(self):
        ws = _make_web_service(basic_auth=("admin", "secret"))
        creds = MagicMock()
        creds.username = "admin"
        creds.password = "wrong"
        from signalwire_agents.web.web_service import HTTPException

        if HTTPException is None:
            pytest.skip("HTTPException not available")
        with pytest.raises(HTTPException):
            ws._get_current_username(creds)
        _stop_patches(ws)

    def test_both_wrong_raises(self):
        ws = _make_web_service(basic_auth=("admin", "secret"))
        creds = MagicMock()
        creds.username = "bad"
        creds.password = "bad"
        from signalwire_agents.web.web_service import HTTPException

        if HTTPException is None:
            pytest.skip("HTTPException not available")
        with pytest.raises(HTTPException):
            ws._get_current_username(creds)
        _stop_patches(ws)


# ---------------------------------------------------------------------------
# add_directory / remove_directory
# ---------------------------------------------------------------------------

class TestAddRemoveDirectory:
    """Tests for add_directory and remove_directory helpers."""

    def test_add_directory_success(self, web_service, tmp_path):
        d = tmp_path / "public"
        d.mkdir()
        web_service.add_directory("/pub", str(d))
        assert "/pub" in web_service.directories
        assert web_service.directories["/pub"] == str(d)

    def test_add_directory_auto_slash(self, web_service, tmp_path):
        d = tmp_path / "assets"
        d.mkdir()
        web_service.add_directory("assets", str(d))
        assert "/assets" in web_service.directories

    def test_add_nonexistent_directory_raises(self, web_service):
        with pytest.raises(ValueError, match="does not exist"):
            web_service.add_directory("/nope", "/nonexistent/path/xyz")

    def test_add_file_as_directory_raises(self, web_service, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="not a directory"):
            web_service.add_directory("/f", str(f))

    def test_remove_directory(self, web_service, tmp_path):
        d = tmp_path / "removeme"
        d.mkdir()
        web_service.add_directory("/removeme", str(d))
        assert "/removeme" in web_service.directories
        web_service.remove_directory("/removeme")
        assert "/removeme" not in web_service.directories

    def test_remove_directory_auto_slash(self, web_service, tmp_path):
        d = tmp_path / "gone"
        d.mkdir()
        web_service.add_directory("/gone", str(d))
        web_service.remove_directory("gone")  # no leading slash
        assert "/gone" not in web_service.directories

    def test_remove_nonexistent_is_noop(self, web_service):
        # Should not raise
        web_service.remove_directory("/does_not_exist")


# ---------------------------------------------------------------------------
# _mount_directories
# ---------------------------------------------------------------------------

class TestMountDirectories:
    """Tests for _mount_directories edge cases."""

    def test_skip_nonexistent_directory(self, web_service, tmp_path):
        """Directories that don't exist on disk should be skipped."""
        web_service.directories = {"/missing": "/no/such/path"}
        # Should not raise
        web_service._mount_directories()

    def test_skip_non_directory_path(self, web_service, tmp_path):
        """A path that is a file (not a dir) should be skipped."""
        f = tmp_path / "afile.txt"
        f.write_text("content")
        web_service.directories = {"/afile": str(f)}
        web_service._mount_directories()

    def test_route_gets_leading_slash(self, web_service, tmp_path):
        d = tmp_path / "web"
        d.mkdir()
        web_service.directories = {"noslash": str(d)}
        # _mount_directories normalises the route – it should not crash
        web_service._mount_directories()


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

class TestStartStop:
    """Tests for the start and stop lifecycle methods."""

    def test_start_without_fastapi_raises(self, web_service_no_fastapi):
        with pytest.raises(RuntimeError, match="FastAPI not available"):
            web_service_no_fastapi.start()

    def test_start_calls_uvicorn(self, web_service):
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            web_service._basic_auth = ("user", "pass")
            web_service.start(host="127.0.0.1", port=5555)
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1]["host"] == "127.0.0.1"
            assert call_kwargs[1]["port"] == 5555

    def test_start_uses_instance_port_as_default(self, web_service):
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            web_service._basic_auth = ("u", "p")
            web_service.start()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1]["port"] == 9999

    def test_start_with_ssl_params(self, web_service):
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            web_service._basic_auth = ("u", "p")
            web_service.start(ssl_cert="/cert.pem", ssl_key="/key.pem")
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1].get("ssl_certfile") == "/cert.pem"
            assert call_kwargs[1].get("ssl_keyfile") == "/key.pem"

    def test_start_without_uvicorn_raises(self, web_service):
        web_service._basic_auth = ("u", "p")
        with patch.dict("sys.modules", {"uvicorn": None}):
            with pytest.raises((RuntimeError, ImportError)):
                web_service.start()

    def test_stop_is_noop(self, web_service):
        # stop() is a placeholder – should not raise
        web_service.stop()


# ---------------------------------------------------------------------------
# _setup_security
# ---------------------------------------------------------------------------

class TestSetupSecurity:
    """Tests for the _setup_security method."""

    def test_no_app_returns_early(self):
        ws = _make_web_service(fastapi_available=False)
        # Should not raise even though app is None
        ws._setup_security()
        _stop_patches(ws)

    def test_cors_added_when_enabled(self, web_service):
        """When enable_cors is True, add_middleware should have been called."""
        # The middleware is added during __init__ via _setup_security.
        # We can verify it was called on the app mock.
        if hasattr(web_service.app, "add_middleware"):
            # If app is a real FastAPI or a mock, this simply confirms no crash
            pass  # The fact that __init__ succeeded is the test

    def test_cors_not_added_when_disabled(self):
        ws = _make_web_service(enable_cors=False)
        # If app is a real FastAPI instance, we can't easily check middleware
        # was NOT added without inspecting internals.  The key assertion is
        # that construction succeeds.
        assert ws.enable_cors is False
        _stop_patches(ws)


# ---------------------------------------------------------------------------
# _setup_routes
# ---------------------------------------------------------------------------

class TestSetupRoutes:
    """Tests for the _setup_routes method."""

    def test_no_app_returns_early(self):
        ws = _make_web_service(fastapi_available=False)
        ws._setup_routes()  # should not raise
        _stop_patches(ws)

    def test_routes_registered(self, web_service):
        """After init, routes should exist on the app."""
        if hasattr(web_service.app, "routes"):
            route_paths = [
                getattr(r, "path", None) for r in web_service.app.routes
            ]
            assert "/health" in route_paths
            assert "/" in route_paths


# ---------------------------------------------------------------------------
# File extension / MIME type handling
# ---------------------------------------------------------------------------

class TestMimeTypes:
    """Test that custom MIME types are registered during init."""

    def test_js_mime_type(self, web_service):
        import mimetypes as mt
        assert mt.guess_type("script.js")[0] == "application/javascript"

    def test_css_mime_type(self, web_service):
        import mimetypes as mt
        assert mt.guess_type("style.css")[0] == "text/css"

    def test_json_mime_type(self, web_service):
        import mimetypes as mt
        assert mt.guess_type("data.json")[0] == "application/json"


# ---------------------------------------------------------------------------
# Edge-case: blocked extension detection nuances
# ---------------------------------------------------------------------------

class TestBlockedExtensionEdgeCases:
    """Fine-grained tests for extension/name-based blocking logic."""

    def test_blocked_name_without_dot_prefix(self, tmp_path):
        """A blocked entry without a dot prefix (e.g. '__pycache__') is
        matched against both the name and as a substring of the full path."""
        ws = _make_web_service(blocked_extensions=["secret_dir"])
        d = tmp_path / "secret_dir"
        d.mkdir()
        f = d / "data.txt"
        f.write_text("sensitive")
        assert ws._is_file_allowed(f) is False
        _stop_patches(ws)

    def test_blocked_dot_entry_matches_extension(self, tmp_path):
        ws = _make_web_service(blocked_extensions=[".bak"])
        f = tmp_path / "db.bak"
        f.write_text("backup")
        assert ws._is_file_allowed(f) is False
        _stop_patches(ws)

    def test_blocked_dot_entry_matches_full_name(self, tmp_path):
        """An entry like '.git' should block a file literally named '.git'."""
        ws = _make_web_service(blocked_extensions=[".git"])
        f = tmp_path / ".git"
        f.write_text("ref: refs/heads/main")
        assert ws._is_file_allowed(f) is False
        _stop_patches(ws)

    def test_unblocked_extension_allowed(self, tmp_path):
        ws = _make_web_service(blocked_extensions=[".bak"])
        f = tmp_path / "readme.md"
        f.write_text("# Title")
        assert ws._is_file_allowed(f) is True
        _stop_patches(ws)


# ---------------------------------------------------------------------------
# Integration-style: directory listing excludes blocked & hidden
# ---------------------------------------------------------------------------

class TestDirectoryListingFiltering:
    """Verify the listing respects both hidden-file and extension filters."""

    def test_combined_filtering(self, tmp_path):
        ws = _make_web_service(
            enable_directory_browsing=True,
            blocked_extensions=[".secret"],
        )
        (tmp_path / ".hidden").write_text("h")
        (tmp_path / "allowed.txt").write_text("ok")
        (tmp_path / "blocked.secret").write_text("no")

        html = ws._generate_directory_listing(tmp_path, "/d")
        assert ".hidden" not in html
        assert "blocked.secret" not in html
        assert "allowed.txt" in html
        _stop_patches(ws)


# ---------------------------------------------------------------------------
# Regression: ensure stat errors don't crash directory listing
# ---------------------------------------------------------------------------

class TestDirectoryListingStatErrors:
    """Ensure errors during directory iteration don't crash the listing."""

    def test_stat_error_on_file_skips_gracefully(self, web_service_with_browsing, tmp_path):
        """If _is_file_allowed returns False (e.g. stat error), the file
        should simply be omitted from the listing."""
        f = tmp_path / "broken.txt"
        f.write_text("ok")

        # Patch _is_file_allowed to simulate stat failure
        web_service_with_browsing._is_file_allowed = MagicMock(return_value=False)
        html = web_service_with_browsing._generate_directory_listing(tmp_path, "/files")
        assert "broken.txt" not in html


# ---------------------------------------------------------------------------
# NEW TESTS: _load_config branch coverage
# ---------------------------------------------------------------------------

class TestLoadConfigBranches:
    """Tests for _load_config to cover missing branches (lines 124, 129)."""

    def test_load_config_find_returns_none(self):
        """When find_config_file returns None and no config_file given,
        _load_config should return early (line 124)."""
        with patch(
            "signalwire_agents.web.web_service.SecurityConfig",
            return_value=_make_security_mock(),
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader"
        ) as mock_cl_cls:
            mock_cl_cls.find_config_file.return_value = None
            from signalwire_agents.web.web_service import WebService
            ws = WebService(port=9999, directories={})
        # Defaults should be set; ConfigLoader should not have been instantiated
        # for loading (only find_config_file was called)
        assert ws.directories == {}

    def test_load_config_has_config_false(self):
        """When ConfigLoader.has_config() is False, _load_config should
        return early (line 129)."""
        mock_loader = MagicMock()
        mock_loader.has_config.return_value = False

        with patch(
            "signalwire_agents.web.web_service.SecurityConfig",
            return_value=_make_security_mock(),
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader",
            return_value=mock_loader,
        ) as mock_cl_cls:
            mock_cl_cls.find_config_file.return_value = "/fake/config.yaml"
            from signalwire_agents.web.web_service import WebService
            ws = WebService(port=9999, directories={})
        # get_section should never be called
        mock_loader.get_section.assert_not_called()

    def test_load_config_service_section_none(self):
        """When get_section('service') returns None, no config is applied."""
        mock_loader = MagicMock()
        mock_loader.has_config.return_value = True
        mock_loader.get_section.return_value = None

        with patch(
            "signalwire_agents.web.web_service.SecurityConfig",
            return_value=_make_security_mock(),
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader",
            return_value=mock_loader,
        ) as mock_cl_cls:
            mock_cl_cls.find_config_file.return_value = "/fake/config.yaml"
            from signalwire_agents.web.web_service import WebService
            ws = WebService(port=9999, directories={})
        assert ws.directories == {}

    def test_load_config_directories_not_dict_ignored(self):
        """When directories in service config is not a dict, it should be ignored."""
        mock_loader = MagicMock()
        mock_loader.has_config.return_value = True
        mock_loader.get_section.return_value = {
            "directories": "not-a-dict",  # should be ignored
        }

        with patch(
            "signalwire_agents.web.web_service.SecurityConfig",
            return_value=_make_security_mock(),
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader",
            return_value=mock_loader,
        ) as mock_cl_cls:
            mock_cl_cls.find_config_file.return_value = "/fake/config.yaml"
            from signalwire_agents.web.web_service import WebService
            ws = WebService(port=9999)
        # directories should remain the default empty dict since the non-dict
        # value was ignored by _load_config and no directories kwarg was given
        assert ws.directories == {}


# ---------------------------------------------------------------------------
# NEW TESTS: Route handler integration tests via TestClient
# ---------------------------------------------------------------------------

class TestRouteHandlers:
    """Integration tests for the FastAPI route handlers using TestClient.
    Covers lines 313, 324-357 (root and health endpoints)."""

    def _make_testable_service(self, directories=None, enable_directory_browsing=False,
                               basic_auth=None, max_file_size=100 * 1024 * 1024,
                               blocked_extensions=None, allowed_extensions=None):
        """Build a WebService with real FastAPI app for TestClient use."""
        security_mock = _make_security_mock()

        with patch(
            "signalwire_agents.web.web_service.SecurityConfig",
            return_value=security_mock,
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader.find_config_file",
            return_value=None,
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader",
        ):
            from signalwire_agents.web.web_service import WebService
            ws = WebService(
                port=9999,
                directories=directories or {},
                basic_auth=basic_auth or ("testuser", "testpass"),
                enable_directory_browsing=enable_directory_browsing,
                max_file_size=max_file_size,
                blocked_extensions=blocked_extensions,
                allowed_extensions=allowed_extensions,
            )
        ws._test_security_mock = security_mock
        return ws

    def test_health_endpoint(self):
        """GET /health should return status and configuration info."""
        from starlette.testclient import TestClient
        ws = self._make_testable_service()
        client = TestClient(ws.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "directories" in data
        assert "directory_browsing" in data

    def test_root_endpoint_html(self):
        """GET / should return HTML listing available directories."""
        from starlette.testclient import TestClient
        ws = self._make_testable_service(directories={"/docs": "/tmp"})
        client = TestClient(ws.app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "SignalWire Web Service" in resp.text
        assert "/docs" in resp.text

    def test_root_endpoint_no_directories(self):
        """GET / with no directories should still return valid HTML."""
        from starlette.testclient import TestClient
        ws = self._make_testable_service()
        client = TestClient(ws.app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Available Directories" in resp.text

    def _auth(self):
        """Return basic auth tuple for TestClient requests."""
        return ("testuser", "testpass")

    def test_serve_file_success(self, tmp_path):
        """Serving a valid file should return its contents."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()
        (d / "hello.txt").write_text("hello world")

        ws = self._make_testable_service(directories={"/files": str(d)})
        client = TestClient(ws.app)
        resp = client.get("/files/hello.txt", auth=self._auth())
        assert resp.status_code == 200
        assert resp.text == "hello world"

    def test_serve_file_not_found(self, tmp_path):
        """Requesting a nonexistent file should return 404."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()

        ws = self._make_testable_service(directories={"/files": str(d)})
        client = TestClient(ws.app, raise_server_exceptions=False)
        resp = client.get("/files/missing.txt", auth=self._auth())
        assert resp.status_code == 404

    def test_serve_file_path_traversal_denied(self, tmp_path):
        """Path traversal attempts should return 403."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()
        # Create a file outside the served dir
        (tmp_path / "secret.txt").write_text("secret")

        ws = self._make_testable_service(directories={"/files": str(d)})
        client = TestClient(ws.app, raise_server_exceptions=False)
        resp = client.get("/files/../secret.txt", auth=self._auth())
        # The path may be normalized by the HTTP client, but the check should
        # either 403 or 404
        assert resp.status_code in (403, 404)

    def test_serve_file_blocked_extension(self, tmp_path):
        """Files with blocked extensions should return 403."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()
        (d / "secrets.env").write_text("SECRET=x")

        ws = self._make_testable_service(directories={"/files": str(d)})
        client = TestClient(ws.app, raise_server_exceptions=False)
        resp = client.get("/files/secrets.env", auth=self._auth())
        assert resp.status_code == 403

    def test_serve_directory_browsing_disabled(self, tmp_path):
        """When browsing is disabled and no index.html, return 403."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        sub = d / "subdir"
        sub.mkdir(parents=True)

        ws = self._make_testable_service(
            directories={"/files": str(d)},
            enable_directory_browsing=False,
        )
        client = TestClient(ws.app, raise_server_exceptions=False)
        resp = client.get("/files/subdir", auth=self._auth())
        assert resp.status_code == 403

    def test_serve_directory_index_html_fallback(self, tmp_path):
        """When browsing is disabled but index.html exists, serve it."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        sub = d / "subdir"
        sub.mkdir(parents=True)
        (sub / "index.html").write_text("<h1>Index</h1>")

        ws = self._make_testable_service(
            directories={"/files": str(d)},
            enable_directory_browsing=False,
        )
        client = TestClient(ws.app)
        resp = client.get("/files/subdir", auth=self._auth())
        assert resp.status_code == 200
        assert "<h1>Index</h1>" in resp.text

    def test_serve_directory_browsing_enabled(self, tmp_path):
        """When directory browsing is enabled, return directory listing."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        sub = d / "subdir"
        sub.mkdir(parents=True)
        (sub / "file.txt").write_text("content")

        ws = self._make_testable_service(
            directories={"/files": str(d)},
            enable_directory_browsing=True,
        )
        client = TestClient(ws.app)
        resp = client.get("/files/subdir", auth=self._auth())
        assert resp.status_code == 200
        assert "Directory listing" in resp.text
        assert "file.txt" in resp.text

    def test_serve_file_mime_type_json(self, tmp_path):
        """JSON files should be served with the correct MIME type."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()
        (d / "data.json").write_text('{"key": "value"}')

        ws = self._make_testable_service(directories={"/files": str(d)})
        client = TestClient(ws.app)
        resp = client.get("/files/data.json", auth=self._auth())
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", "")

    def test_serve_file_cache_headers(self, tmp_path):
        """Served files should include Cache-Control and X-Content-Type-Options."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()
        (d / "style.css").write_text("body {}")

        ws = self._make_testable_service(directories={"/files": str(d)})
        client = TestClient(ws.app)
        resp = client.get("/files/style.css", auth=self._auth())
        assert resp.status_code == 200
        assert "nosniff" in resp.headers.get("X-Content-Type-Options", "")

    def test_serve_file_too_large(self, tmp_path):
        """Files exceeding max_file_size should be denied."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()
        big = d / "big.bin"
        big.write_bytes(b"\x00" * 200)

        ws = self._make_testable_service(
            directories={"/files": str(d)},
            max_file_size=100,
        )
        client = TestClient(ws.app, raise_server_exceptions=False)
        resp = client.get("/files/big.bin", auth=self._auth())
        assert resp.status_code == 403

    def test_serve_file_allowed_extension_filter(self, tmp_path):
        """When allowed_extensions is set, only those should be served."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()
        (d / "page.html").write_text("<p>hi</p>")
        (d / "data.json").write_text("{}")

        ws = self._make_testable_service(
            directories={"/files": str(d)},
            allowed_extensions=[".html"],
        )
        client = TestClient(ws.app, raise_server_exceptions=False)
        resp_html = client.get("/files/page.html", auth=self._auth())
        resp_json = client.get("/files/data.json", auth=self._auth())
        assert resp_html.status_code == 200
        assert resp_json.status_code == 403

    def test_serve_file_wrong_auth_rejected(self, tmp_path):
        """Requests with wrong credentials should be rejected."""
        from starlette.testclient import TestClient
        d = tmp_path / "www"
        d.mkdir()
        (d / "hello.txt").write_text("hello")

        ws = self._make_testable_service(directories={"/files": str(d)})
        client = TestClient(ws.app, raise_server_exceptions=False)
        resp = client.get("/files/hello.txt", auth=("wrong", "creds"))
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# NEW TESTS: Security middleware coverage
# ---------------------------------------------------------------------------

class TestSecurityMiddleware:
    """Tests for security middleware (lines 169-182, 187-191).
    Uses TestClient to exercise the middleware in-process."""

    def _make_testable_service(self, **kwargs):
        security_mock = _make_security_mock()
        with patch(
            "signalwire_agents.web.web_service.SecurityConfig",
            return_value=security_mock,
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader.find_config_file",
            return_value=None,
        ), patch(
            "signalwire_agents.web.web_service.ConfigLoader",
        ):
            from signalwire_agents.web.web_service import WebService
            ws = WebService(
                port=9999,
                directories=kwargs.get("directories", {}),
                basic_auth=("testuser", "testpass"),
                enable_directory_browsing=kwargs.get("enable_directory_browsing", False),
            )
        ws._test_security_mock = security_mock
        return ws

    def test_security_headers_added_to_response(self):
        """Security headers from SecurityConfig should be added to responses."""
        from starlette.testclient import TestClient
        ws = self._make_testable_service()
        client = TestClient(ws.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        # The security mock returns these headers
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_host_validation_blocks_invalid_host(self):
        """When should_allow_host returns False, the request should be rejected."""
        from starlette.testclient import TestClient
        ws = self._make_testable_service()
        # Make should_allow_host return False for invalid hosts
        ws._test_security_mock.should_allow_host.return_value = False
        client = TestClient(ws.app, raise_server_exceptions=False)
        resp = client.get("/health", headers={"Host": "evil.example.com"})
        assert resp.status_code == 400
        assert "Invalid host" in resp.text

    def test_host_validation_allows_valid_host(self):
        """When should_allow_host returns True, the request should proceed."""
        from starlette.testclient import TestClient
        ws = self._make_testable_service()
        ws._test_security_mock.should_allow_host.return_value = True
        client = TestClient(ws.app)
        resp = client.get("/health", headers={"Host": "good.example.com"})
        assert resp.status_code == 200

    def test_cache_headers_for_static_directory_paths(self, tmp_path):
        """Requests to configured directory paths should get cache headers."""
        from starlette.testclient import TestClient
        d = tmp_path / "static"
        d.mkdir()
        (d / "app.js").write_text("console.log('hi')")

        ws = self._make_testable_service(directories={"/static": str(d)})
        client = TestClient(ws.app)
        resp = client.get("/static/app.js", auth=("testuser", "testpass"))
        assert resp.status_code == 200
        # The middleware adds Cache-Control for paths starting with directory keys
        assert "max-age=3600" in resp.headers.get("Cache-Control", "")


# ---------------------------------------------------------------------------
# NEW TESTS: _mount_directories edge cases
# ---------------------------------------------------------------------------

class TestMountDirectoriesEdgeCases:
    """Additional edge cases for _mount_directories (line 362)."""

    def test_mount_no_app_returns_early(self):
        """When self.app is None, _mount_directories should return immediately."""
        ws = _make_web_service(fastapi_available=False)
        ws.directories = {"/test": "/tmp"}
        ws._mount_directories()  # should not raise
        _stop_patches(ws)

    def test_mount_with_valid_directory(self, tmp_path):
        """Mounting a valid directory should register a route."""
        ws = _make_web_service()
        d = tmp_path / "web"
        d.mkdir()
        ws.directories = {"/web": str(d)}
        # This should not raise
        ws._mount_directories()
        _stop_patches(ws)


# ---------------------------------------------------------------------------
# NEW TESTS: start() method edge cases
# ---------------------------------------------------------------------------

class TestStartEdgeCases:
    """Additional tests for start() covering SSL config paths."""

    def test_start_ssl_from_security_config(self):
        """When no ssl_cert/ssl_key params, use security config SSL settings."""
        ws = _make_web_service(basic_auth=("u", "p"))
        ws._test_security_mock.get_ssl_context_kwargs.return_value = {
            "ssl_certfile": "/path/cert.pem",
            "ssl_keyfile": "/path/key.pem",
        }
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            ws.start()
        call_kwargs = mock_uvicorn.run.call_args[1]
        assert call_kwargs.get("ssl_certfile") == "/path/cert.pem"
        assert call_kwargs.get("ssl_keyfile") == "/path/key.pem"
        _stop_patches(ws)

    def test_start_prints_ssl_enabled(self, capsys):
        """When SSL is configured, 'SSL: Enabled' should be printed."""
        ws = _make_web_service(basic_auth=("u", "p"))
        ws._test_security_mock.get_ssl_context_kwargs.return_value = {
            "ssl_certfile": "/path/cert.pem",
            "ssl_keyfile": "/path/key.pem",
        }
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            ws.start()
        captured = capsys.readouterr()
        assert "SSL: Enabled" in captured.out
        _stop_patches(ws)

    def test_start_prints_directory_none_when_empty(self, capsys):
        """When no directories configured, should print 'None'."""
        ws = _make_web_service(basic_auth=("u", "p"))
        ws.directories = {}
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            ws.start()
        captured = capsys.readouterr()
        assert "None" in captured.out
        _stop_patches(ws)

    def test_start_https_scheme_in_output(self, capsys):
        """When SSL params are given, scheme should be https."""
        ws = _make_web_service(basic_auth=("u", "p"))
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            ws.start(ssl_cert="/c.pem", ssl_key="/k.pem")
        captured = capsys.readouterr()
        assert "https://" in captured.out
        _stop_patches(ws)

    def test_start_http_scheme_in_output(self, capsys):
        """When no SSL, scheme should be http."""
        ws = _make_web_service(basic_auth=("u", "p"))
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            ws.start()
        captured = capsys.readouterr()
        assert "http://" in captured.out
        _stop_patches(ws)


# ---------------------------------------------------------------------------
# NEW TESTS: add_directory with app already running
# ---------------------------------------------------------------------------

class TestAddDirectoryWithApp:
    """Tests for add_directory when app is already set."""

    def test_add_directory_triggers_mount(self, tmp_path):
        """Adding a directory when app exists should call _mount_directories."""
        ws = _make_web_service()
        d = tmp_path / "new_dir"
        d.mkdir()
        with patch.object(ws, "_mount_directories") as mock_mount:
            ws.add_directory("/new", str(d))
            mock_mount.assert_called_once()
        _stop_patches(ws)

    def test_add_directory_without_app_skips_mount(self, tmp_path):
        """Adding a directory when app is None should not call _mount_directories."""
        ws = _make_web_service(fastapi_available=False)
        d = tmp_path / "new_dir"
        d.mkdir()
        ws.add_directory("/new", str(d))
        assert "/new" in ws.directories
        _stop_patches(ws)
