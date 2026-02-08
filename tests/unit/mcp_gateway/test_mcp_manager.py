"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for MCPManager, MCPClient, and MCPService classes
in signalwire_agents.mcp_gateway.mcp_manager
"""

import os
import json
import queue
import threading
import subprocess
import time
import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock, call

# Skip the entire module when Flask is not installed (mcp_gateway.__init__ imports flask)
pytest.importorskip("flask", reason="flask is required for MCP Gateway tests")

from signalwire_agents.mcp_gateway.mcp_manager import (
    MCPService,
    MCPClient,
    MCPManager,
)


# ---------------------------------------------------------------------------
# MCPService dataclass tests
# ---------------------------------------------------------------------------


class TestMCPService:
    """Tests for the MCPService dataclass."""

    def test_initialization_with_defaults(self):
        """MCPService should populate default sandbox_config via __post_init__."""
        service = MCPService(
            name="test_svc",
            command=["python", "-m", "test_server"],
            description="A test service",
        )

        assert service.name == "test_svc"
        assert service.command == ["python", "-m", "test_server"]
        assert service.description == "A test service"
        assert service.enabled is True
        assert service.sandbox_config == {
            "enabled": True,
            "resource_limits": True,
            "restricted_env": True,
        }

    def test_initialization_with_custom_sandbox_config(self):
        """MCPService should accept a custom sandbox_config."""
        custom_sandbox = {"enabled": False, "resource_limits": False}
        service = MCPService(
            name="custom",
            command=["node", "server.js"],
            description="custom service",
            sandbox_config=custom_sandbox,
        )

        assert service.sandbox_config is custom_sandbox
        assert service.sandbox_config["enabled"] is False

    def test_enabled_defaults_to_true(self):
        """The enabled flag should default to True."""
        service = MCPService(name="s", command=["cmd"], description="d")
        assert service.enabled is True

    def test_enabled_can_be_false(self):
        """The enabled flag can be set to False."""
        service = MCPService(
            name="s", command=["cmd"], description="d", enabled=False
        )
        assert service.enabled is False

    def test_hash_based_on_name(self):
        """MCPService hash should be based on the name field."""
        svc1 = MCPService(name="alpha", command=["a"], description="")
        svc2 = MCPService(name="alpha", command=["b"], description="different")

        assert hash(svc1) == hash(svc2)
        assert hash(svc1) == hash("alpha")

    def test_hash_differs_for_different_names(self):
        """Services with different names should have different hashes."""
        svc1 = MCPService(name="alpha", command=["a"], description="")
        svc2 = MCPService(name="beta", command=["a"], description="")

        assert hash(svc1) != hash(svc2)

    def test_post_init_does_not_override_explicit_none(self):
        """When sandbox_config is explicitly None at init-time it gets the default."""
        service = MCPService(
            name="x", command=["x"], description="", sandbox_config=None
        )
        assert service.sandbox_config is not None
        assert "enabled" in service.sandbox_config


# ---------------------------------------------------------------------------
# MCPClient tests
# ---------------------------------------------------------------------------


class TestMCPClientInit:
    """Tests for MCPClient.__init__."""

    def test_defaults(self):
        """MCPClient should initialise with correct defaults."""
        service = MCPService(name="svc", command=["cmd"], description="desc")
        client = MCPClient(service)

        assert client.service is service
        assert client.process is None
        assert client.request_id == 0
        assert client.pending_requests == {}
        assert isinstance(client.response_queue, queue.Queue)
        assert client.reader_thread is None
        assert client.tools == []
        assert client.sandbox_base_dir == "./sandbox"
        assert client.sandbox_dir is None
        assert not client._shutdown.is_set()

    def test_custom_sandbox_base_dir(self):
        """MCPClient should accept a custom sandbox_base_dir."""
        service = MCPService(name="svc", command=["cmd"], description="desc")
        client = MCPClient(service, sandbox_base_dir="/tmp/my_sandbox")

        assert client.sandbox_base_dir == "/tmp/my_sandbox"


class TestMCPClientSetupSandboxEnv:
    """Tests for MCPClient._setup_sandbox_env."""

    def _make_client(self, sandbox_config=None):
        svc = MCPService(
            name="test",
            command=["echo"],
            description="",
            sandbox_config=sandbox_config,
        )
        return MCPClient(svc, sandbox_base_dir="/tmp/sandbox_test")

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_sandboxing_disabled_returns_full_env(self, mock_makedirs):
        """When sandbox is disabled, the full environment is returned."""
        client = self._make_client(sandbox_config={"enabled": False})

        env, cwd = client._setup_sandbox_env()

        # Should not create sandbox dir
        mock_makedirs.assert_not_called()
        # Should return roughly the current environment
        assert "PATH" in env
        assert cwd == os.getcwd()

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_sandbox_enabled_restricted_env(self, mock_makedirs):
        """When sandbox is enabled with restricted_env, a minimal env is created."""
        client = self._make_client(
            sandbox_config={
                "enabled": True,
                "restricted_env": True,
                "resource_limits": True,
            }
        )

        env, cwd = client._setup_sandbox_env()

        mock_makedirs.assert_called_once()
        # Restricted env should have limited keys
        assert "PATH" in env
        assert "HOME" in env
        assert "LANG" in env
        assert env["LANG"] == "C.UTF-8"
        # Dangerous vars must be removed
        assert "LD_PRELOAD" not in env
        assert "LD_LIBRARY_PATH" not in env
        assert "DYLD_INSERT_LIBRARIES" not in env

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_sandbox_enabled_unrestricted_env(self, mock_makedirs):
        """When sandbox is enabled but restricted_env is False, full env with overrides is returned."""
        client = self._make_client(
            sandbox_config={
                "enabled": True,
                "restricted_env": False,
                "resource_limits": False,
            }
        )

        with patch.dict(os.environ, {"LD_PRELOAD": "evil.so"}, clear=False):
            env, cwd = client._setup_sandbox_env()

        # LD_PRELOAD should be stripped even with unrestricted
        assert "LD_PRELOAD" not in env
        # HOME should be overridden to sandbox dir
        assert client.sandbox_dir is not None
        assert env["HOME"] == client.sandbox_dir

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_sandbox_copies_essential_env_vars(self, mock_makedirs):
        """Essential vars like PYTHONPATH should be copied when present."""
        client = self._make_client(
            sandbox_config={"enabled": True, "restricted_env": True}
        )

        with patch.dict(
            os.environ, {"PYTHONPATH": "/my/path", "NODE_PATH": "/node"}, clear=False
        ):
            env, _ = client._setup_sandbox_env()

        assert env.get("PYTHONPATH") == "/my/path"
        assert env.get("NODE_PATH") == "/node"

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_sandbox_working_dir_from_config(self, mock_makedirs):
        """Working directory can be overridden via sandbox config."""
        client = self._make_client(
            sandbox_config={"enabled": True, "working_dir": "/custom/dir"}
        )

        _, cwd = client._setup_sandbox_env()
        assert cwd == "/custom/dir"

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_sandbox_disabled_working_dir_from_config(self, mock_makedirs):
        """When sandboxing disabled, working_dir from config is still honoured."""
        client = self._make_client(
            sandbox_config={"enabled": False, "working_dir": "/other/dir"}
        )

        _, cwd = client._setup_sandbox_env()
        assert cwd == "/other/dir"


class TestMCPClientSendMessage:
    """Tests for MCPClient._send_message."""

    def _make_client_with_process(self):
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.process = Mock()
        client.process.poll.return_value = None  # process running
        client.process.stdin = Mock()
        return client

    def test_send_message_writes_json_line(self):
        """_send_message should write JSON followed by newline."""
        client = self._make_client_with_process()
        msg = {"jsonrpc": "2.0", "id": 1, "method": "test"}

        client._send_message(msg)

        written = client.process.stdin.write.call_args[0][0]
        assert written.endswith("\n")
        assert json.loads(written.strip()) == msg
        client.process.stdin.flush.assert_called_once()

    def test_send_message_raises_when_process_is_none(self):
        """_send_message should raise RuntimeError when process is None."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.process = None

        with pytest.raises(RuntimeError, match="not running"):
            client._send_message({"jsonrpc": "2.0"})

    def test_send_message_raises_when_process_exited(self):
        """_send_message should raise RuntimeError when process has exited."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.process = Mock()
        client.process.poll.return_value = 1  # exited

        with pytest.raises(RuntimeError, match="not running"):
            client._send_message({"jsonrpc": "2.0"})


class TestMCPClientCallMethod:
    """Tests for MCPClient.call_method."""

    def _make_client(self):
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        return client

    def test_call_method_raises_when_shutting_down(self):
        """call_method should raise RuntimeError when shutdown is set."""
        client = self._make_client()
        client._shutdown.set()

        with pytest.raises(RuntimeError, match="shutting down"):
            client.call_method("test", {})

    def test_call_method_increments_request_id(self):
        """call_method should increment request_id for each call."""
        client = self._make_client()

        # Mock _send_message to capture what is sent and immediately resolve
        def fake_send(msg):
            req_id = msg["id"]
            with client.lock:
                if req_id in client.pending_requests:
                    client.pending_requests[req_id]["response"] = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"ok": True},
                    }
                    client.pending_requests[req_id]["event"].set()

        client._send_message = fake_send

        client.call_method("method1", {})
        client.call_method("method2", {})

        assert client.request_id == 2

    def test_call_method_returns_result(self):
        """call_method should return the 'result' portion of the response."""
        client = self._make_client()
        expected_result = {"tools": [{"name": "tool1"}]}

        def fake_send(msg):
            req_id = msg["id"]
            with client.lock:
                if req_id in client.pending_requests:
                    client.pending_requests[req_id]["response"] = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": expected_result,
                    }
                    client.pending_requests[req_id]["event"].set()

        client._send_message = fake_send

        result = client.call_method("tools/list", {})
        assert result == expected_result

    def test_call_method_raises_on_error_response(self):
        """call_method should raise Exception when server returns an error."""
        client = self._make_client()

        def fake_send(msg):
            req_id = msg["id"]
            with client.lock:
                if req_id in client.pending_requests:
                    client.pending_requests[req_id]["response"] = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32600, "message": "Invalid request"},
                    }
                    client.pending_requests[req_id]["event"].set()

        client._send_message = fake_send

        with pytest.raises(Exception, match="MCP Error"):
            client.call_method("bad_method", {})

    def test_call_method_timeout(self):
        """call_method should raise TimeoutError when no response arrives."""
        client = self._make_client()

        # _send_message does nothing (no response will arrive)
        client._send_message = Mock()

        with pytest.raises(TimeoutError, match="Timeout"):
            # Use a very short timeout by patching the Event.wait
            with patch.object(threading.Event, "wait", return_value=False):
                client.call_method("slow_method", {})

    def test_call_method_cleans_up_on_timeout(self):
        """Pending request should be removed on timeout."""
        client = self._make_client()
        client._send_message = Mock()

        with patch.object(threading.Event, "wait", return_value=False):
            with pytest.raises(TimeoutError):
                client.call_method("slow_method", {})

        # Pending requests should be empty after timeout cleanup
        assert len(client.pending_requests) == 0


class TestMCPClientCallTool:
    """Tests for MCPClient.call_tool."""

    def test_call_tool_delegates_to_call_method(self):
        """call_tool should delegate to call_method with the right params."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.call_method = Mock(return_value={"content": [{"text": "ok"}]})

        result = client.call_tool("my_tool", {"arg1": "val1"})

        client.call_method.assert_called_once_with(
            "tools/call", {"name": "my_tool", "arguments": {"arg1": "val1"}}
        )
        assert result == {"content": [{"text": "ok"}]}


class TestMCPClientGetTools:
    """Tests for MCPClient.get_tools."""

    def test_get_tools_returns_copy(self):
        """get_tools should return a copy of the tools list."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.tools = [{"name": "tool1"}, {"name": "tool2"}]

        tools = client.get_tools()

        assert tools == [{"name": "tool1"}, {"name": "tool2"}]
        assert tools is not client.tools  # must be a copy

    def test_get_tools_empty(self):
        """get_tools should return empty list when no tools available."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)

        assert client.get_tools() == []


class TestMCPClientStart:
    """Tests for MCPClient.start."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.subprocess.Popen")
    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_start_success(self, mock_makedirs, mock_popen):
        """start() should return True when initialization and tool listing succeed."""
        service = MCPService(
            name="svc",
            command=["echo", "hello"],
            description="test",
            sandbox_config={"enabled": False},
        )
        client = MCPClient(service)

        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # Mock _initialize and _list_tools
        client._initialize = Mock(return_value=True)
        client._list_tools = Mock(return_value=[{"name": "tool1"}])

        result = client.start()

        assert result is True
        assert client.process is mock_proc
        assert client.tools == [{"name": "tool1"}]
        client._initialize.assert_called_once()
        client._list_tools.assert_called_once()

    @patch("signalwire_agents.mcp_gateway.mcp_manager.subprocess.Popen")
    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_start_fails_on_initialize(self, mock_makedirs, mock_popen):
        """start() should return False when _initialize fails."""
        service = MCPService(
            name="svc",
            command=["echo"],
            description="",
            sandbox_config={"enabled": False},
        )
        client = MCPClient(service)

        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = Mock()
        mock_proc.wait = Mock()
        mock_proc.kill = Mock()
        mock_proc.stderr = Mock()
        mock_proc.stderr.read.return_value = ""
        mock_popen.return_value = mock_proc

        client._initialize = Mock(return_value=False)

        result = client.start()

        assert result is False

    @patch("signalwire_agents.mcp_gateway.mcp_manager.subprocess.Popen")
    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_start_fails_on_popen_exception(self, mock_makedirs, mock_popen):
        """start() should return False when Popen raises an exception."""
        service = MCPService(
            name="svc",
            command=["nonexistent_cmd"],
            description="",
            sandbox_config={"enabled": False},
        )
        client = MCPClient(service)
        mock_popen.side_effect = FileNotFoundError("command not found")

        result = client.start()

        assert result is False

    @patch("signalwire_agents.mcp_gateway.mcp_manager.subprocess.Popen")
    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_start_creates_reader_thread(self, mock_makedirs, mock_popen):
        """start() should launch a reader thread."""
        service = MCPService(
            name="svc",
            command=["echo"],
            description="",
            sandbox_config={"enabled": False},
        )
        client = MCPClient(service)

        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        client._initialize = Mock(return_value=True)
        client._list_tools = Mock(return_value=[])

        with patch("signalwire_agents.mcp_gateway.mcp_manager.threading.Thread") as mock_thread_cls:
            mock_thread_instance = Mock()
            mock_thread_cls.return_value = mock_thread_instance

            client.start()

            mock_thread_cls.assert_called_once()
            mock_thread_instance.start.assert_called_once()


class TestMCPClientStop:
    """Tests for MCPClient.stop."""

    def test_stop_terminates_running_process(self):
        """stop() should terminate a running process gracefully."""
        service = MCPService(
            name="svc",
            command=["cmd"],
            description="",
            sandbox_config={"enabled": True},
        )
        client = MCPClient(service)
        client.process = Mock()
        client.process.stdin = Mock()
        # First poll returns None (running), then 0 after terminate
        client.process.poll.side_effect = [None, None, 0]
        client.process.wait.return_value = None
        client.process.terminate.return_value = None
        client.sandbox_dir = None

        client.stop()

        assert client._shutdown.is_set()
        client.process is None

    def test_stop_force_kills_on_timeout(self):
        """stop() should force kill if terminate doesn't work in time."""
        service = MCPService(
            name="svc",
            command=["cmd"],
            description="",
            sandbox_config={"enabled": True},
        )
        client = MCPClient(service)

        mock_proc = Mock()
        mock_proc.stdin = Mock()
        # Always returns None (still running)
        mock_proc.poll.return_value = None
        mock_proc.terminate.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 1), None]
        mock_proc.kill.return_value = None
        client.process = mock_proc
        client.sandbox_dir = None

        client.stop()

        mock_proc.kill.assert_called()

    @patch("signalwire_agents.mcp_gateway.mcp_manager.shutil.rmtree")
    def test_stop_cleans_up_sandbox_dir(self, mock_rmtree):
        """stop() should remove the sandbox directory if it exists."""
        service = MCPService(
            name="svc", command=["cmd"], description="",
            sandbox_config={"enabled": True},
        )
        client = MCPClient(service)
        client.process = None  # already stopped
        client.sandbox_dir = "/tmp/sandbox_test/mcp_svc_123"

        with patch("signalwire_agents.mcp_gateway.mcp_manager.os.path.exists", return_value=True):
            client.stop()

        mock_rmtree.assert_called_once_with("/tmp/sandbox_test/mcp_svc_123")

    def test_stop_when_no_process(self):
        """stop() should not raise when process is None."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.process = None
        client.sandbox_dir = None

        # Should not raise
        client.stop()
        assert client._shutdown.is_set()

    def test_stop_joins_reader_thread(self):
        """stop() should join the reader thread with a timeout."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.process = None
        client.sandbox_dir = None

        mock_thread = Mock()
        # is_alive must return True so the join branch is entered
        # (the code checks `if self.reader_thread and self.reader_thread.is_alive():`)
        mock_thread.is_alive.side_effect = [True, False]
        client.reader_thread = mock_thread

        client.stop()

        mock_thread.join.assert_called_once_with(timeout=1.0)


class TestMCPClientInitialize:
    """Tests for MCPClient._initialize."""

    def test_initialize_success(self):
        """_initialize should return True when call_method succeeds."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.call_method = Mock(
            return_value={"serverInfo": {"name": "test", "version": "1.0"}}
        )

        result = client._initialize()

        assert result is True
        client.call_method.assert_called_once_with(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "mcp-gateway", "version": "1.0.0"},
            },
        )

    def test_initialize_failure(self):
        """_initialize should return False when call_method raises."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.call_method = Mock(side_effect=Exception("Connection failed"))

        result = client._initialize()

        assert result is False


class TestMCPClientListTools:
    """Tests for MCPClient._list_tools."""

    def test_list_tools_success(self):
        """_list_tools should return the tools list from the server."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        expected_tools = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
        ]
        client.call_method = Mock(return_value={"tools": expected_tools})

        tools = client._list_tools()

        assert tools == expected_tools
        client.call_method.assert_called_once_with("tools/list", {})

    def test_list_tools_returns_empty_on_error(self):
        """_list_tools should return empty list on failure."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.call_method = Mock(side_effect=TimeoutError("timeout"))

        tools = client._list_tools()

        assert tools == []

    def test_list_tools_handles_missing_tools_key(self):
        """_list_tools should return empty list when response has no 'tools' key."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client.call_method = Mock(return_value={})

        tools = client._list_tools()

        assert tools == []


class TestMCPClientReadLoop:
    """Tests for MCPClient._read_loop."""

    def test_read_loop_processes_response(self):
        """_read_loop should route responses to pending_requests by id."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)

        response_json = json.dumps(
            {"jsonrpc": "2.0", "id": 42, "result": {"data": "value"}}
        )

        mock_proc = Mock()
        # readline returns the response, then empty string (EOF)
        mock_proc.stdout.readline.side_effect = [response_json + "\n", ""]
        mock_proc.poll.side_effect = [None, None, 1]  # running, running, exited
        mock_proc.returncode = 0
        mock_proc.stderr.read.return_value = ""
        client.process = mock_proc

        # Set up a pending request for id 42
        event = threading.Event()
        client.pending_requests[42] = {"event": event, "response": None}

        client._read_loop()

        assert event.is_set()
        assert client.pending_requests[42]["response"]["result"] == {"data": "value"}

    def test_read_loop_handles_invalid_json(self):
        """_read_loop should log and skip invalid JSON lines."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)

        mock_proc = Mock()
        mock_proc.stdout.readline.side_effect = ["not valid json\n", ""]
        mock_proc.poll.side_effect = [None, None, 1]
        mock_proc.returncode = 0
        mock_proc.stderr.read.return_value = ""
        client.process = mock_proc

        # Should not raise
        client._read_loop()

    def test_read_loop_handles_notifications(self):
        """_read_loop should handle messages without an 'id' (notifications)."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)

        notification = json.dumps({"jsonrpc": "2.0", "method": "notify", "params": {}})

        mock_proc = Mock()
        mock_proc.stdout.readline.side_effect = [notification + "\n", ""]
        mock_proc.poll.side_effect = [None, None, 1]
        mock_proc.returncode = 0
        mock_proc.stderr.read.return_value = ""
        client.process = mock_proc

        # Should not raise
        client._read_loop()

    def test_read_loop_exits_on_shutdown(self):
        """_read_loop should stop when _shutdown event is set."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)
        client._shutdown.set()

        mock_proc = Mock()
        mock_proc.poll.return_value = None
        client.process = mock_proc

        # Should return immediately without trying to read
        client._read_loop()

    def test_read_loop_ignores_unknown_request_ids(self):
        """_read_loop should not fail on responses for unknown request ids."""
        service = MCPService(name="svc", command=["cmd"], description="")
        client = MCPClient(service)

        response = json.dumps({"jsonrpc": "2.0", "id": 999, "result": {}})

        mock_proc = Mock()
        mock_proc.stdout.readline.side_effect = [response + "\n", ""]
        mock_proc.poll.side_effect = [None, None, 1]
        mock_proc.returncode = 0
        mock_proc.stderr.read.return_value = ""
        client.process = mock_proc

        # Should not raise
        client._read_loop()
        # 999 should not appear in pending_requests
        assert 999 not in client.pending_requests


class TestMCPClientSandboxPreexec:
    """Tests for MCPClient._sandbox_preexec."""

    def test_preexec_does_nothing_when_sandbox_disabled(self):
        """_sandbox_preexec should be a no-op when sandbox is disabled."""
        service = MCPService(
            name="svc",
            command=["cmd"],
            description="",
            sandbox_config={"enabled": False},
        )
        client = MCPClient(service)

        # Should not raise or do anything
        client._sandbox_preexec()

    @patch("signalwire_agents.mcp_gateway.mcp_manager.resource.setrlimit")
    def test_preexec_sets_resource_limits(self, mock_setrlimit):
        """_sandbox_preexec should set resource limits when enabled."""
        service = MCPService(
            name="svc",
            command=["cmd"],
            description="",
            sandbox_config={"enabled": True, "resource_limits": True},
        )
        client = MCPClient(service)

        client._sandbox_preexec()

        # Should have called setrlimit multiple times
        assert mock_setrlimit.call_count == 4

    @patch("signalwire_agents.mcp_gateway.mcp_manager.resource.setrlimit")
    def test_preexec_skips_resource_limits_when_disabled(self, mock_setrlimit):
        """_sandbox_preexec should skip resource limits when disabled."""
        service = MCPService(
            name="svc",
            command=["cmd"],
            description="",
            sandbox_config={"enabled": True, "resource_limits": False},
        )
        client = MCPClient(service)

        client._sandbox_preexec()

        mock_setrlimit.assert_not_called()

    @patch("signalwire_agents.mcp_gateway.mcp_manager.resource.setrlimit")
    def test_preexec_handles_resource_limit_errors(self, mock_setrlimit):
        """_sandbox_preexec should not raise on resource limit errors."""
        mock_setrlimit.side_effect = OSError("Not permitted")

        service = MCPService(
            name="svc",
            command=["cmd"],
            description="",
            sandbox_config={"enabled": True, "resource_limits": True},
        )
        client = MCPClient(service)

        # Should not raise
        client._sandbox_preexec()


# ---------------------------------------------------------------------------
# MCPManager tests
# ---------------------------------------------------------------------------


class TestMCPManagerInit:
    """Tests for MCPManager.__init__."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_init_with_empty_config(self, mock_makedirs):
        """MCPManager should initialize with an empty config."""
        config = {}
        manager = MCPManager(config)

        assert manager.config is config
        assert manager.services == {}
        assert manager.clients == {}
        assert manager.sandbox_base_dir == "./sandbox"
        mock_makedirs.assert_called_once_with("./sandbox", exist_ok=True)

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_init_with_custom_sandbox_dir(self, mock_makedirs):
        """MCPManager should use sandbox_dir from session config."""
        config = {"session": {"sandbox_dir": "/custom/sandbox"}}
        manager = MCPManager(config)

        assert manager.sandbox_base_dir == "/custom/sandbox"
        mock_makedirs.assert_called_once_with("/custom/sandbox", exist_ok=True)

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_init_loads_services(self, mock_makedirs):
        """MCPManager should load services from config on init."""
        config = {
            "services": {
                "service1": {
                    "command": ["python", "server.py"],
                    "description": "Test service 1",
                    "enabled": True,
                },
                "service2": {
                    "command": ["node", "server.js"],
                    "description": "Test service 2",
                },
            }
        }

        manager = MCPManager(config)

        assert "service1" in manager.services
        assert "service2" in manager.services
        assert len(manager.services) == 2

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_init_skips_disabled_services(self, mock_makedirs):
        """MCPManager should skip services with enabled=False."""
        config = {
            "services": {
                "enabled_svc": {
                    "command": ["echo"],
                    "description": "enabled",
                    "enabled": True,
                },
                "disabled_svc": {
                    "command": ["echo"],
                    "description": "disabled",
                    "enabled": False,
                },
            }
        }

        manager = MCPManager(config)

        assert "enabled_svc" in manager.services
        assert "disabled_svc" not in manager.services
        assert len(manager.services) == 1


class TestMCPManagerLoadServices:
    """Tests for MCPManager._load_services."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_load_services_creates_mcp_service_objects(self, mock_makedirs):
        """Loaded services should be MCPService instances with correct attributes."""
        config = {
            "services": {
                "my_service": {
                    "command": ["python", "-m", "my_server"],
                    "description": "My custom service",
                    "enabled": True,
                    "sandbox": {"enabled": False},
                }
            }
        }

        manager = MCPManager(config)
        svc = manager.services["my_service"]

        assert isinstance(svc, MCPService)
        assert svc.name == "my_service"
        assert svc.command == ["python", "-m", "my_server"]
        assert svc.description == "My custom service"
        assert svc.enabled is True
        assert svc.sandbox_config == {"enabled": False}

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_load_services_default_description(self, mock_makedirs):
        """Services without description should default to empty string."""
        config = {
            "services": {
                "svc": {
                    "command": ["cmd"],
                }
            }
        }

        manager = MCPManager(config)

        assert manager.services["svc"].description == ""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_load_services_no_services_key(self, mock_makedirs):
        """Config without 'services' key should result in no services."""
        config = {"some_other_key": "value"}

        manager = MCPManager(config)

        assert manager.services == {}


class TestMCPManagerGetService:
    """Tests for MCPManager.get_service."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_get_existing_service(self, mock_makedirs):
        """get_service should return the MCPService for a known name."""
        config = {
            "services": {
                "known": {"command": ["cmd"], "description": "known service"}
            }
        }
        manager = MCPManager(config)

        svc = manager.get_service("known")

        assert svc is not None
        assert svc.name == "known"

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_get_nonexistent_service(self, mock_makedirs):
        """get_service should return None for an unknown name."""
        config = {}
        manager = MCPManager(config)

        assert manager.get_service("missing") is None


class TestMCPManagerListServices:
    """Tests for MCPManager.list_services."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_list_services_returns_dict(self, mock_makedirs):
        """list_services should return a dict with service info."""
        config = {
            "services": {
                "svc1": {
                    "command": ["python", "s1.py"],
                    "description": "Service 1",
                },
                "svc2": {
                    "command": ["node", "s2.js"],
                    "description": "Service 2",
                },
            }
        }
        manager = MCPManager(config)

        result = manager.list_services()

        assert len(result) == 2
        assert "svc1" in result
        assert "svc2" in result
        assert result["svc1"]["description"] == "Service 1"
        assert result["svc1"]["command"] == ["python", "s1.py"]
        assert result["svc1"]["enabled"] is True

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_list_services_empty(self, mock_makedirs):
        """list_services should return empty dict when no services."""
        manager = MCPManager({})

        assert manager.list_services() == {}


class TestMCPManagerCreateClient:
    """Tests for MCPManager.create_client."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_create_client_unknown_service(self, mock_makedirs):
        """create_client should raise ValueError for unknown service."""
        manager = MCPManager({})

        with pytest.raises(ValueError, match="Unknown service"):
            manager.create_client("nonexistent")

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_create_client_disabled_service(self, mock_makedirs):
        """create_client should raise ValueError for disabled service."""
        config = {
            "services": {
                "disabled": {
                    "command": ["cmd"],
                    "description": "",
                    "enabled": True,
                }
            }
        }
        manager = MCPManager(config)
        # Manually disable after load
        manager.services["disabled"].enabled = False

        with pytest.raises(ValueError, match="disabled"):
            manager.create_client("disabled")

    @patch("signalwire_agents.mcp_gateway.mcp_manager.MCPClient")
    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_create_client_success(self, mock_makedirs, mock_client_cls):
        """create_client should return a started MCPClient and track it."""
        config = {
            "services": {
                "my_svc": {"command": ["echo"], "description": "test"}
            }
        }
        manager = MCPManager(config)

        mock_client_instance = Mock()
        mock_client_instance.start.return_value = True
        mock_client_cls.return_value = mock_client_instance

        client = manager.create_client("my_svc")

        assert client is mock_client_instance
        mock_client_instance.start.assert_called_once()
        # Should be tracked in manager.clients
        assert len(manager.clients) == 1

    @patch("signalwire_agents.mcp_gateway.mcp_manager.MCPClient")
    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_create_client_start_failure(self, mock_makedirs, mock_client_cls):
        """create_client should raise RuntimeError when client.start() fails."""
        config = {
            "services": {
                "my_svc": {"command": ["bad_cmd"], "description": "test"}
            }
        }
        manager = MCPManager(config)

        mock_client_instance = Mock()
        mock_client_instance.start.return_value = False
        mock_client_cls.return_value = mock_client_instance

        with pytest.raises(RuntimeError, match="Failed to start"):
            manager.create_client("my_svc")

    @patch("signalwire_agents.mcp_gateway.mcp_manager.MCPClient")
    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_create_client_tracks_with_unique_key(self, mock_makedirs, mock_client_cls):
        """Each created client should get a unique key in the clients dict."""
        config = {
            "services": {
                "svc": {"command": ["cmd"], "description": ""}
            }
        }
        manager = MCPManager(config)

        client1 = Mock()
        client1.start.return_value = True
        client2 = Mock()
        client2.start.return_value = True
        mock_client_cls.side_effect = [client1, client2]

        manager.create_client("svc")
        manager.create_client("svc")

        assert len(manager.clients) == 2


class TestMCPManagerGetServiceTools:
    """Tests for MCPManager.get_service_tools."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_get_service_tools_returns_tools_and_cleans_up(self, mock_makedirs):
        """get_service_tools should return tools and clean up the temp client."""
        config = {
            "services": {
                "svc": {"command": ["cmd"], "description": ""}
            }
        }
        manager = MCPManager(config)

        mock_client = Mock()
        mock_client.start.return_value = True
        mock_client.get_tools.return_value = [{"name": "tool_a"}]

        with patch.object(manager, "create_client", return_value=mock_client) as mock_create:
            # We need to simulate what create_client does - track the client
            def side_effect(name):
                key = f"{name}_{id(mock_client)}"
                manager.clients[key] = mock_client
                return mock_client
            mock_create.side_effect = side_effect

            tools = manager.get_service_tools("svc")

        assert tools == [{"name": "tool_a"}]
        mock_client.stop.assert_called_once()
        # Client should be cleaned up from tracking
        assert len(manager.clients) == 0

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_get_service_tools_cleans_up_on_error(self, mock_makedirs):
        """get_service_tools should clean up even when get_tools raises."""
        config = {
            "services": {
                "svc": {"command": ["cmd"], "description": ""}
            }
        }
        manager = MCPManager(config)

        mock_client = Mock()
        mock_client.start.return_value = True
        mock_client.get_tools.side_effect = Exception("oops")

        with patch.object(manager, "create_client", return_value=mock_client) as mock_create:
            def side_effect(name):
                key = f"{name}_{id(mock_client)}"
                manager.clients[key] = mock_client
                return mock_client
            mock_create.side_effect = side_effect

            with pytest.raises(Exception, match="oops"):
                manager.get_service_tools("svc")

        mock_client.stop.assert_called_once()


class TestMCPManagerValidateServices:
    """Tests for MCPManager.validate_services."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_validate_services_all_pass(self, mock_makedirs):
        """validate_services should return True for all services that start OK."""
        config = {
            "services": {
                "svc1": {"command": ["cmd1"], "description": ""},
                "svc2": {"command": ["cmd2"], "description": ""},
            }
        }
        manager = MCPManager(config)

        mock_client = Mock()
        mock_client.start.return_value = True

        with patch.object(manager, "create_client", return_value=mock_client) as mock_create:
            def side_effect(name):
                key = f"{name}_{id(mock_client)}"
                manager.clients[key] = mock_client
                return mock_client
            mock_create.side_effect = side_effect

            results = manager.validate_services()

        assert results == {"svc1": True, "svc2": True}

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_validate_services_some_fail(self, mock_makedirs):
        """validate_services should return False for services that fail to start."""
        config = {
            "services": {
                "good": {"command": ["cmd1"], "description": ""},
                "bad": {"command": ["cmd2"], "description": ""},
            }
        }
        manager = MCPManager(config)

        good_client = Mock()
        good_client.start.return_value = True

        def create_side_effect(name):
            if name == "good":
                key = f"{name}_{id(good_client)}"
                manager.clients[key] = good_client
                return good_client
            raise RuntimeError("Failed to start")

        with patch.object(manager, "create_client", side_effect=create_side_effect):
            results = manager.validate_services()

        assert results["good"] is True
        assert results["bad"] is False


class TestMCPManagerShutdown:
    """Tests for MCPManager.shutdown."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_shutdown_stops_all_clients(self, mock_makedirs):
        """shutdown should stop all tracked clients and clear the dict."""
        manager = MCPManager({})

        client1 = Mock()
        client2 = Mock()
        manager.clients = {"svc1_123": client1, "svc2_456": client2}

        manager.shutdown()

        client1.stop.assert_called_once()
        client2.stop.assert_called_once()
        assert len(manager.clients) == 0

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_shutdown_handles_stop_errors(self, mock_makedirs):
        """shutdown should continue stopping other clients even if one fails."""
        manager = MCPManager({})

        client1 = Mock()
        client1.stop.side_effect = Exception("stop error")
        client2 = Mock()
        manager.clients = {"svc1_123": client1, "svc2_456": client2}

        # Should not raise
        manager.shutdown()

        client1.stop.assert_called_once()
        client2.stop.assert_called_once()

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_shutdown_no_clients(self, mock_makedirs):
        """shutdown should handle the case with no active clients."""
        manager = MCPManager({})

        # Should not raise
        manager.shutdown()
        assert len(manager.clients) == 0


# ---------------------------------------------------------------------------
# Integration-style tests (still unit-level, using mocks)
# ---------------------------------------------------------------------------


class TestMCPManagerServiceLifecycle:
    """Tests for service lifecycle management patterns."""

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_full_lifecycle(self, mock_makedirs):
        """Test creating a manager, listing services, and shutting down."""
        config = {
            "services": {
                "echo": {
                    "command": ["echo", "hello"],
                    "description": "Echo service",
                }
            }
        }

        manager = MCPManager(config)

        # Should have one service
        services = manager.list_services()
        assert len(services) == 1
        assert "echo" in services

        # Get service by name
        svc = manager.get_service("echo")
        assert svc.name == "echo"
        assert svc.command == ["echo", "hello"]

        # Shutdown should be clean
        manager.shutdown()

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_config_with_sandbox_overrides(self, mock_makedirs):
        """Services should correctly inherit sandbox config from the config file."""
        config = {
            "services": {
                "sandboxed": {
                    "command": ["cmd"],
                    "description": "sandboxed service",
                    "sandbox": {
                        "enabled": True,
                        "resource_limits": False,
                        "restricted_env": False,
                    },
                },
                "unsandboxed": {
                    "command": ["cmd"],
                    "description": "unsandboxed service",
                    "sandbox": {
                        "enabled": False,
                    },
                },
            }
        }

        manager = MCPManager(config)

        sandboxed = manager.get_service("sandboxed")
        assert sandboxed.sandbox_config["enabled"] is True
        assert sandboxed.sandbox_config["resource_limits"] is False

        unsandboxed = manager.get_service("unsandboxed")
        assert unsandboxed.sandbox_config["enabled"] is False

    @patch("signalwire_agents.mcp_gateway.mcp_manager.os.makedirs")
    def test_multiple_services_isolation(self, mock_makedirs):
        """Each service should be an independent MCPService instance."""
        config = {
            "services": {
                "svc_a": {"command": ["a"], "description": "A"},
                "svc_b": {"command": ["b"], "description": "B"},
            }
        }

        manager = MCPManager(config)

        svc_a = manager.get_service("svc_a")
        svc_b = manager.get_service("svc_b")

        assert svc_a is not svc_b
        assert svc_a.name != svc_b.name
        assert svc_a.command != svc_b.command
