"""Tests for calling namespace — command dispatch."""

from .conftest import MockResponse


class TestCallingNamespace:
    def test_dial(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"id": "call-1"})
        client.calling.dial(to="+15551234567", from_="+15559876543")
        call_args = mock_session.request.call_args
        assert call_args[0][0] == "POST"
        body = call_args[1]["json"]
        assert body["command"] == "dial"
        assert body["params"]["to"] == "+15551234567"
        assert "id" not in body  # dial has no top-level id

    def test_play_with_call_id(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.calling.play("call-123", play=[{"type": "tts", "text": "hello"}])
        call_args = mock_session.request.call_args
        body = call_args[1]["json"]
        assert body["command"] == "calling.play"
        assert body["id"] == "call-123"
        assert body["params"]["play"] == [{"type": "tts", "text": "hello"}]

    def test_end(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.calling.end("call-123", reason="hangup")
        body = mock_session.request.call_args[1]["json"]
        assert body["command"] == "calling.end"
        assert body["id"] == "call-123"
        assert body["params"]["reason"] == "hangup"

    def test_record_stop(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.calling.record_stop("call-123", control_id="ctrl-1")
        body = mock_session.request.call_args[1]["json"]
        assert body["command"] == "calling.record.stop"
        assert body["id"] == "call-123"

    def test_ai_message(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.calling.ai_message("call-123", role="user", message_text="hi")
        body = mock_session.request.call_args[1]["json"]
        assert body["command"] == "calling.ai_message"
        assert body["params"]["role"] == "user"

    def test_all_methods_exist(self, client):
        methods = [
            "dial", "update", "end", "transfer", "disconnect",
            "play", "play_pause", "play_resume", "play_stop", "play_volume",
            "record", "record_pause", "record_resume", "record_stop",
            "collect", "collect_stop", "collect_start_input_timers",
            "detect", "detect_stop",
            "tap", "tap_stop",
            "stream", "stream_stop",
            "denoise", "denoise_stop",
            "transcribe", "transcribe_stop",
            "ai_message", "ai_hold", "ai_unhold", "ai_stop",
            "live_transcribe", "live_translate",
            "send_fax_stop", "receive_fax_stop",
            "refer", "user_event",
        ]
        for method in methods:
            assert hasattr(client.calling, method), f"Missing calling method: {method}"
            assert callable(getattr(client.calling, method))
