"""Unit tests for relay Call object and Action classes."""

import asyncio
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch


from signalwire_agents.relay.call import (
    Call,
    Action,
    PlayAction,
    RecordAction,
    DetectAction,
    CollectAction,
    StandaloneCollectAction,
    FaxAction,
    TapAction,
    StreamAction,
    PayAction,
    TranscribeAction,
    AIAction,
)
from signalwire_agents.relay.event import RelayEvent
from signalwire_agents.relay.constants import (
    CALL_STATE_ENDED,
    CALL_STATE_ANSWERED,
    EVENT_CALL_STATE,
    EVENT_CALL_PLAY,
    EVENT_CALL_COLLECT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockRelayError(Exception):
    """Mock of RelayError for testing call-gone handling."""
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"RELAY error {code}: {message}")


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.execute = AsyncMock(return_value={"code": "200", "message": "OK"})
    return client


@pytest.fixture
async def call(mock_client):
    return Call(
        client=mock_client,
        call_id="call-1",
        node_id="node-1",
        project_id="proj-1",
        context="office",
        state=CALL_STATE_ANSWERED,
        direction="inbound",
    )


# ---------------------------------------------------------------------------
# Call._execute tests
# ---------------------------------------------------------------------------

class TestCallExecute:
    @pytest.mark.asyncio
    async def test_execute_sends_correct_method(self, call, mock_client):
        await call._execute("answer")
        mock_client.execute.assert_called_once_with(
            "calling.answer",
            {"node_id": "node-1", "call_id": "call-1"},
        )

    @pytest.mark.asyncio
    async def test_execute_merges_extra_params(self, call, mock_client):
        await call._execute("play", {"control_id": "ctl1", "play": []})
        mock_client.execute.assert_called_once_with(
            "calling.play",
            {"node_id": "node-1", "call_id": "call-1", "control_id": "ctl1", "play": []},
        )

    @pytest.mark.asyncio
    async def test_execute_returns_result(self, call, mock_client):
        mock_client.execute.return_value = {"code": "200", "message": "OK", "url": "http://rec.wav"}
        result = await call._execute("record", {"control_id": "ctl1"})
        assert result["url"] == "http://rec.wav"

    @pytest.mark.asyncio
    async def test_execute_swallows_404(self, call, mock_client):
        mock_client.execute.side_effect = MockRelayError(404, "Call not found")
        result = await call._execute("play", {"control_id": "ctl1"})
        assert result == {}

    @pytest.mark.asyncio
    async def test_execute_swallows_410(self, call, mock_client):
        mock_client.execute.side_effect = MockRelayError(410, "Call gone")
        result = await call._execute("play.stop", {"control_id": "ctl1"})
        assert result == {}

    @pytest.mark.asyncio
    async def test_execute_swallows_non_gone_relay_errors(self, call, mock_client):
        """Non-gone relay errors (e.g. 500) are logged and return {} — not raised."""
        mock_client.execute.side_effect = MockRelayError(500, "Server error")
        result = await call._execute("play", {"control_id": "ctl1"})
        assert result == {}

    @pytest.mark.asyncio
    async def test_execute_raises_non_relay_errors(self, call, mock_client):
        mock_client.execute.side_effect = ConnectionError("lost")
        with pytest.raises(ConnectionError):
            await call._execute("play", {"control_id": "ctl1"})


# ---------------------------------------------------------------------------
# Call lifecycle methods
# ---------------------------------------------------------------------------

class TestCallLifecycle:
    @pytest.mark.asyncio
    async def test_answer(self, call, mock_client):
        await call.answer()
        mock_client.execute.assert_called_once_with(
            "calling.answer", {"node_id": "node-1", "call_id": "call-1"}
        )

    @pytest.mark.asyncio
    async def test_hangup(self, call, mock_client):
        await call.hangup(reason="busy")
        mock_client.execute.assert_called_once_with(
            "calling.end",
            {"node_id": "node-1", "call_id": "call-1", "reason": "busy"},
        )

    @pytest.mark.asyncio
    async def test_pass(self, call, mock_client):
        await call.pass_()
        mock_client.execute.assert_called_once_with(
            "calling.pass", {"node_id": "node-1", "call_id": "call-1"}
        )


# ---------------------------------------------------------------------------
# Action-based methods
# ---------------------------------------------------------------------------

class TestPlayMethod:
    @pytest.mark.asyncio
    async def test_play_returns_play_action(self, call):
        action = await call.play([{"type": "tts", "params": {"text": "Hello"}}])
        assert isinstance(action, PlayAction)
        assert action.control_id  # should have a UUID

    @pytest.mark.asyncio
    async def test_play_with_options(self, call, mock_client):
        action = await call.play(
            [{"type": "tts", "params": {"text": "Hello"}}],
            volume=3.0,
            direction="listen",
            loop=2,
            control_id="my-ctl",
        )
        assert action.control_id == "my-ctl"
        args = mock_client.execute.call_args
        params = args[0][1]
        assert params["volume"] == 3.0
        assert params["direction"] == "listen"
        assert params["loop"] == 2

    @pytest.mark.asyncio
    async def test_play_action_stop(self, call, mock_client):
        action = await call.play([{"type": "tts", "params": {"text": "Hi"}}], control_id="ctl1")
        mock_client.execute.reset_mock()
        await action.stop()
        mock_client.execute.assert_called_once_with(
            "calling.play.stop",
            {"node_id": "node-1", "call_id": "call-1", "control_id": "ctl1"},
        )

    @pytest.mark.asyncio
    async def test_play_action_pause_resume_volume(self, call, mock_client):
        action = await call.play([{"type": "tts", "params": {"text": "Hi"}}], control_id="ctl1")

        mock_client.execute.reset_mock()
        await action.pause()
        assert mock_client.execute.call_args[0][0] == "calling.play.pause"

        mock_client.execute.reset_mock()
        await action.resume()
        assert mock_client.execute.call_args[0][0] == "calling.play.resume"

        mock_client.execute.reset_mock()
        await action.volume(-5.0)
        args = mock_client.execute.call_args
        assert args[0][0] == "calling.play.volume"
        assert args[0][1]["volume"] == -5.0


class TestRecordMethod:
    @pytest.mark.asyncio
    async def test_record_returns_record_action(self, call):
        action = await call.record()
        assert isinstance(action, RecordAction)

    @pytest.mark.asyncio
    async def test_record_with_audio_params(self, call, mock_client):
        action = await call.record(
            audio={"format": "wav", "stereo": True, "direction": "both"},
            control_id="r1",
        )
        args = mock_client.execute.call_args
        params = args[0][1]
        assert params["record"]["audio"]["format"] == "wav"
        assert params["control_id"] == "r1"


class TestDetectMethod:
    @pytest.mark.asyncio
    async def test_detect_returns_detect_action(self, call):
        action = await call.detect({"type": "machine"})
        assert isinstance(action, DetectAction)

    @pytest.mark.asyncio
    async def test_detect_with_timeout(self, call, mock_client):
        await call.detect({"type": "digit"}, timeout=60.0, control_id="d1")
        params = mock_client.execute.call_args[0][1]
        assert params["timeout"] == 60.0
        assert params["detect"]["type"] == "digit"


class TestCollectMethods:
    @pytest.mark.asyncio
    async def test_play_and_collect(self, call):
        action = await call.play_and_collect(
            [{"type": "tts", "params": {"text": "Press 1"}}],
            {"digits": {"max": 1}},
        )
        assert isinstance(action, CollectAction)

    @pytest.mark.asyncio
    async def test_standalone_collect(self, call, mock_client):
        action = await call.collect(
            digits={"max": 4, "terminators": "#"},
            speech={"language": "en-US"},
            partial_results=True,
            continuous=False,
            control_id="col1",
        )
        assert isinstance(action, StandaloneCollectAction)
        params = mock_client.execute.call_args[0][1]
        assert params["digits"]["max"] == 4
        assert params["speech"]["language"] == "en-US"
        assert params["partial_results"] is True
        assert mock_client.execute.call_args[0][0] == "calling.collect"

    @pytest.mark.asyncio
    async def test_standalone_collect_stop(self, call, mock_client):
        action = await call.collect(digits={"max": 1}, control_id="col1")
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.collect.stop"

    @pytest.mark.asyncio
    async def test_standalone_collect_start_input_timers(self, call, mock_client):
        action = await call.collect(digits={"max": 1}, control_id="col1")
        mock_client.execute.reset_mock()
        await action.start_input_timers()
        assert mock_client.execute.call_args[0][0] == "calling.collect.start_input_timers"


class TestConnectDisconnect:
    @pytest.mark.asyncio
    async def test_connect(self, call, mock_client):
        await call.connect(
            [[{"type": "phone", "params": {"to_number": "+15551234567", "from_number": "+15559876543"}}]],
            ringback=[{"type": "ringtone", "params": {"name": "us"}}],
        )
        params = mock_client.execute.call_args[0][1]
        assert params["devices"][0][0]["type"] == "phone"
        assert params["ringback"][0]["type"] == "ringtone"

    @pytest.mark.asyncio
    async def test_disconnect(self, call, mock_client):
        await call.disconnect()
        assert mock_client.execute.call_args[0][0] == "calling.disconnect"


class TestSendDigits:
    @pytest.mark.asyncio
    async def test_send_digits(self, call, mock_client):
        await call.send_digits("1234#", control_id="sd1")
        params = mock_client.execute.call_args[0][1]
        assert params["digits"] == "1234#"
        assert params["control_id"] == "sd1"


class TestRefer:
    @pytest.mark.asyncio
    async def test_refer(self, call, mock_client):
        await call.refer(
            {"type": "sip", "params": {"to": "user@example.com"}},
            status_url="https://example.com/refer",
        )
        params = mock_client.execute.call_args[0][1]
        assert params["device"]["params"]["to"] == "user@example.com"
        assert params["status_url"] == "https://example.com/refer"


class TestPay:
    @pytest.mark.asyncio
    async def test_pay_returns_pay_action(self, call):
        action = await call.pay("https://pay.example.com", control_id="pay1")
        assert isinstance(action, PayAction)

    @pytest.mark.asyncio
    async def test_pay_with_options(self, call, mock_client):
        await call.pay(
            "https://pay.example.com",
            control_id="pay1",
            input_method="dtmf",
            charge_amount="15.00",
            currency="usd",
        )
        params = mock_client.execute.call_args[0][1]
        assert params["payment_connector_url"] == "https://pay.example.com"
        assert params["input"] == "dtmf"
        assert params["charge_amount"] == "15.00"


class TestFax:
    @pytest.mark.asyncio
    async def test_send_fax(self, call, mock_client):
        action = await call.send_fax(
            "https://example.com/doc.pdf",
            identity="+15551234567",
            header_info="Test",
            control_id="fax1",
        )
        assert isinstance(action, FaxAction)
        params = mock_client.execute.call_args[0][1]
        assert params["document"] == "https://example.com/doc.pdf"
        assert params["identity"] == "+15551234567"
        assert mock_client.execute.call_args[0][0] == "calling.send_fax"

    @pytest.mark.asyncio
    async def test_receive_fax(self, call, mock_client):
        action = await call.receive_fax(control_id="fax2")
        assert isinstance(action, FaxAction)
        assert mock_client.execute.call_args[0][0] == "calling.receive_fax"

    @pytest.mark.asyncio
    async def test_fax_action_stop_uses_correct_prefix(self, call, mock_client):
        action = await call.send_fax("https://example.com/doc.pdf", control_id="fax1")
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.send_fax.stop"

        action2 = await call.receive_fax(control_id="fax2")
        mock_client.execute.reset_mock()
        await action2.stop()
        assert mock_client.execute.call_args[0][0] == "calling.receive_fax.stop"


class TestTap:
    @pytest.mark.asyncio
    async def test_tap(self, call, mock_client):
        action = await call.tap(
            {"type": "audio", "params": {"direction": "speak"}},
            {"type": "rtp", "params": {"addr": "1.2.3.4", "port": 5000}},
            control_id="tap1",
        )
        assert isinstance(action, TapAction)
        params = mock_client.execute.call_args[0][1]
        assert params["tap"]["type"] == "audio"
        assert params["device"]["params"]["port"] == 5000


class TestStream:
    @pytest.mark.asyncio
    async def test_stream(self, call, mock_client):
        action = await call.stream(
            "wss://example.com/audio",
            name="my_stream",
            codec="PCMU",
            track="inbound_track",
            control_id="str1",
        )
        assert isinstance(action, StreamAction)
        params = mock_client.execute.call_args[0][1]
        assert params["url"] == "wss://example.com/audio"
        assert params["name"] == "my_stream"
        assert params["codec"] == "PCMU"
        assert params["track"] == "inbound_track"


class TestTransfer:
    @pytest.mark.asyncio
    async def test_transfer(self, call, mock_client):
        await call.transfer("https://example.com/swml")
        params = mock_client.execute.call_args[0][1]
        assert params["dest"] == "https://example.com/swml"


class TestConference:
    @pytest.mark.asyncio
    async def test_join_conference(self, call, mock_client):
        await call.join_conference(
            "my_conf",
            muted=False,
            beep="onEnter",
            max_participants=10,
            record="record-from-start",
        )
        params = mock_client.execute.call_args[0][1]
        assert params["name"] == "my_conf"
        assert params["muted"] is False
        assert params["beep"] == "onEnter"
        assert params["max_participants"] == 10

    @pytest.mark.asyncio
    async def test_leave_conference(self, call, mock_client):
        await call.leave_conference("conf-1")
        params = mock_client.execute.call_args[0][1]
        assert params["conference_id"] == "conf-1"


class TestHoldUnhold:
    @pytest.mark.asyncio
    async def test_hold(self, call, mock_client):
        await call.hold()
        assert mock_client.execute.call_args[0][0] == "calling.hold"

    @pytest.mark.asyncio
    async def test_unhold(self, call, mock_client):
        await call.unhold()
        assert mock_client.execute.call_args[0][0] == "calling.unhold"


class TestDenoise:
    @pytest.mark.asyncio
    async def test_denoise(self, call, mock_client):
        await call.denoise()
        assert mock_client.execute.call_args[0][0] == "calling.denoise"

    @pytest.mark.asyncio
    async def test_denoise_stop(self, call, mock_client):
        await call.denoise_stop()
        assert mock_client.execute.call_args[0][0] == "calling.denoise.stop"


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe(self, call, mock_client):
        action = await call.transcribe(control_id="tr1", status_url="https://cb.example.com")
        assert isinstance(action, TranscribeAction)
        params = mock_client.execute.call_args[0][1]
        assert params["status_url"] == "https://cb.example.com"


class TestEcho:
    @pytest.mark.asyncio
    async def test_echo(self, call, mock_client):
        await call.echo(timeout=30.0)
        params = mock_client.execute.call_args[0][1]
        assert params["timeout"] == 30.0


class TestDigitBindings:
    @pytest.mark.asyncio
    async def test_bind_digit(self, call, mock_client):
        await call.bind_digit(
            "*1",
            "calling.play",
            bind_params={"play": [{"type": "tts", "params": {"text": "Hello"}}]},
            realm="menu",
            max_triggers=3,
        )
        params = mock_client.execute.call_args[0][1]
        assert params["digits"] == "*1"
        assert params["bind_method"] == "calling.play"
        assert params["params"]["play"][0]["type"] == "tts"
        assert params["realm"] == "menu"
        assert params["max_triggers"] == 3

    @pytest.mark.asyncio
    async def test_clear_digit_bindings(self, call, mock_client):
        await call.clear_digit_bindings(realm="menu")
        params = mock_client.execute.call_args[0][1]
        assert params["realm"] == "menu"


class TestLiveTranscribeTranslate:
    @pytest.mark.asyncio
    async def test_live_transcribe(self, call, mock_client):
        await call.live_transcribe({"start": {}})
        params = mock_client.execute.call_args[0][1]
        assert params["action"] == {"start": {}}

    @pytest.mark.asyncio
    async def test_live_translate(self, call, mock_client):
        await call.live_translate({"start": {}}, status_url="https://example.com")
        params = mock_client.execute.call_args[0][1]
        assert params["action"] == {"start": {}}
        assert params["status_url"] == "https://example.com"


class TestRoom:
    @pytest.mark.asyncio
    async def test_join_room(self, call, mock_client):
        await call.join_room("my_room", status_url="https://example.com")
        params = mock_client.execute.call_args[0][1]
        assert params["name"] == "my_room"

    @pytest.mark.asyncio
    async def test_leave_room(self, call, mock_client):
        await call.leave_room()
        assert mock_client.execute.call_args[0][0] == "calling.leave_room"


class TestAI:
    @pytest.mark.asyncio
    async def test_ai_returns_ai_action(self, call):
        action = await call.ai(
            prompt={"text": "You are helpful.", "temperature": 0.3},
            control_id="ai1",
        )
        assert isinstance(action, AIAction)

    @pytest.mark.asyncio
    async def test_ai_with_full_params(self, call, mock_client):
        await call.ai(
            control_id="ai1",
            agent="agent-uuid",
            prompt={"text": "Hello"},
            SWAIG={"functions": []},
            ai_params={"end_of_speech_timeout": 3000},
            global_data={"key": "value"},
            hints=["sales", "support"],
        )
        params = mock_client.execute.call_args[0][1]
        assert params["agent"] == "agent-uuid"
        assert params["prompt"]["text"] == "Hello"
        assert params["SWAIG"] == {"functions": []}
        assert params["params"]["end_of_speech_timeout"] == 3000
        assert params["global_data"]["key"] == "value"
        assert params["hints"] == ["sales", "support"]

    @pytest.mark.asyncio
    async def test_ai_stop(self, call, mock_client):
        action = await call.ai(control_id="ai1")
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.ai.stop"

    @pytest.mark.asyncio
    async def test_amazon_bedrock(self, call, mock_client):
        await call.amazon_bedrock(prompt="You are helpful.")
        params = mock_client.execute.call_args[0][1]
        assert params["prompt"] == "You are helpful."
        assert mock_client.execute.call_args[0][0] == "calling.amazon_bedrock"

    @pytest.mark.asyncio
    async def test_ai_message(self, call, mock_client):
        await call.ai_message(message_text="Order confirmed.", role="system")
        params = mock_client.execute.call_args[0][1]
        assert params["message_text"] == "Order confirmed."
        assert params["role"] == "system"

    @pytest.mark.asyncio
    async def test_ai_hold(self, call, mock_client):
        await call.ai_hold(timeout="60", prompt="Please hold.")
        params = mock_client.execute.call_args[0][1]
        assert params["timeout"] == "60"
        assert params["prompt"] == "Please hold."

    @pytest.mark.asyncio
    async def test_ai_unhold(self, call, mock_client):
        await call.ai_unhold(prompt="Thanks for holding.")
        params = mock_client.execute.call_args[0][1]
        assert params["prompt"] == "Thanks for holding."


class TestUserEvent:
    @pytest.mark.asyncio
    async def test_user_event(self, call, mock_client):
        await call.user_event(event="custom_event")
        params = mock_client.execute.call_args[0][1]
        assert params["event"] == "custom_event"


class TestQueue:
    @pytest.mark.asyncio
    async def test_queue_enter(self, call, mock_client):
        await call.queue_enter("support", control_id="q1", status_url="https://example.com")
        params = mock_client.execute.call_args[0][1]
        assert params["queue_name"] == "support"
        assert params["control_id"] == "q1"
        assert mock_client.execute.call_args[0][0] == "calling.queue.enter"

    @pytest.mark.asyncio
    async def test_queue_leave(self, call, mock_client):
        await call.queue_leave("support", control_id="q1", queue_id="qid1")
        params = mock_client.execute.call_args[0][1]
        assert params["queue_name"] == "support"
        assert params["queue_id"] == "qid1"
        assert mock_client.execute.call_args[0][0] == "calling.queue.leave"


# ---------------------------------------------------------------------------
# _start_action tests
# ---------------------------------------------------------------------------

class TestStartAction:
    @pytest.mark.asyncio
    async def test_ended_call_resolves_gracefully(self, call, mock_client):
        """Starting an action on an ended call logs a warning and returns a resolved action."""
        call.state = CALL_STATE_ENDED
        action = await call.play([{"type": "tts", "params": {"text": "Hi"}}])
        assert action.completed is True
        assert action.is_done is True

    @pytest.mark.asyncio
    async def test_execute_failure_resolves_action(self, call, mock_client):
        """Non-gone relay errors are swallowed; action is resolved immediately."""
        mock_client.execute.side_effect = MockRelayError(500, "Server error")
        action = await call.play([{"type": "tts", "params": {"text": "Hi"}}])
        assert action.completed is True
        assert action.control_id not in call._actions

    @pytest.mark.asyncio
    async def test_call_gone_resolves_action_immediately(self, call, mock_client):
        mock_client.execute.side_effect = MockRelayError(404, "Call not found")
        action = await call.play([{"type": "tts", "params": {"text": "Hi"}}])
        assert action.completed is True
        assert action.is_done is True
        # action.wait() should return immediately
        result = await action.wait(timeout=1.0)
        assert result is not None
        # action should be cleaned up from _actions
        assert action.control_id not in call._actions


# ---------------------------------------------------------------------------
# Event dispatch tests
# ---------------------------------------------------------------------------

class TestEventDispatch:
    @pytest.mark.asyncio
    async def test_state_event_updates_call_state(self, call):
        payload = {
            "event_type": EVENT_CALL_STATE,
            "params": {"call_id": "call-1", "call_state": "ending"},
        }
        await call._dispatch_event(payload)
        assert call.state == "ending"

    @pytest.mark.asyncio
    async def test_ended_event_resolves_ended_future(self, call):
        payload = {
            "event_type": EVENT_CALL_STATE,
            "params": {"call_id": "call-1", "call_state": CALL_STATE_ENDED},
        }
        await call._dispatch_event(payload)
        assert call.state == CALL_STATE_ENDED
        assert call._ended.done()

    @pytest.mark.asyncio
    async def test_action_resolved_by_event(self, call, mock_client):
        action = await call.play(
            [{"type": "tts", "params": {"text": "Hello"}}],
            control_id="ctl1",
        )
        assert not action.is_done
        # Simulate play finished event
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "ctl1", "state": "finished"},
        })
        assert action.is_done
        assert action.completed
        assert "ctl1" not in call._actions

    @pytest.mark.asyncio
    async def test_listener_called(self, call):
        events_received = []

        def handler(event):
            events_received.append(event)

        call.on(EVENT_CALL_PLAY, handler)
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "ctl1", "state": "playing"},
        })
        assert len(events_received) == 1
        assert events_received[0].params["state"] == "playing"

    @pytest.mark.asyncio
    async def test_wait_for(self, call):
        async def send_event_later():
            await asyncio.sleep(0.01)
            await call._dispatch_event({
                "event_type": EVENT_CALL_PLAY,
                "params": {"call_id": "call-1", "control_id": "ctl1", "state": "finished"},
            })

        task = asyncio.create_task(send_event_later())
        event = await call.wait_for(EVENT_CALL_PLAY, timeout=2.0)
        assert event.params["state"] == "finished"
        await task

    @pytest.mark.asyncio
    async def test_wait_for_with_predicate(self, call):
        async def send_events():
            await asyncio.sleep(0.01)
            await call._dispatch_event({
                "event_type": EVENT_CALL_PLAY,
                "params": {"call_id": "call-1", "control_id": "ctl1", "state": "playing"},
            })
            await asyncio.sleep(0.01)
            await call._dispatch_event({
                "event_type": EVENT_CALL_PLAY,
                "params": {"call_id": "call-1", "control_id": "ctl1", "state": "finished"},
            })

        task = asyncio.create_task(send_events())
        event = await call.wait_for(
            EVENT_CALL_PLAY,
            predicate=lambda e: e.params.get("state") == "finished",
            timeout=2.0,
        )
        assert event.params["state"] == "finished"
        await task

    @pytest.mark.asyncio
    async def test_wait_for_ended(self, call):
        async def end_call():
            await asyncio.sleep(0.01)
            await call._dispatch_event({
                "event_type": EVENT_CALL_STATE,
                "params": {"call_id": "call-1", "call_state": CALL_STATE_ENDED},
            })

        task = asyncio.create_task(end_call())
        event = await call.wait_for_ended(timeout=2.0)
        assert event.params["call_state"] == CALL_STATE_ENDED
        await task


class TestCollectActionEventRouting:
    """Test that CollectAction only resolves on collect events, not play events."""

    @pytest.mark.asyncio
    async def test_collect_ignores_play_events(self, call, mock_client):
        action = await call.play_and_collect(
            [{"type": "tts", "params": {"text": "Press 1"}}],
            {"digits": {"max": 1}},
            control_id="pac1",
        )
        # Simulate play event — should NOT resolve the collect action
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "pac1", "state": "finished"},
        })
        assert not action.is_done

        # Simulate collect result — should resolve
        await call._dispatch_event({
            "event_type": EVENT_CALL_COLLECT,
            "params": {
                "call_id": "call-1",
                "control_id": "pac1",
                "result": {"type": "digit", "params": {"digits": "1"}},
            },
        })
        assert action.is_done
        assert action.result.params["result"]["type"] == "digit"


class TestDetectActionEventRouting:
    """Test that DetectAction resolves on first detect result."""

    @pytest.mark.asyncio
    async def test_detect_resolves_on_first_result(self, call, mock_client):
        action = await call.detect({"type": "machine"}, control_id="det1")
        await call._dispatch_event({
            "event_type": "calling.call.detect",
            "params": {
                "call_id": "call-1",
                "control_id": "det1",
                "detect": {"type": "machine", "params": {"event": "HUMAN"}},
            },
        })
        assert action.is_done
        assert action.result.params["detect"]["params"]["event"] == "HUMAN"


class TestRecordActionMethods:
    @pytest.mark.asyncio
    async def test_record_stop(self, call, mock_client):
        action = await call.record(control_id="r1")
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.record.stop"
        assert mock_client.execute.call_args[0][1]["control_id"] == "r1"

    @pytest.mark.asyncio
    async def test_record_pause(self, call, mock_client):
        action = await call.record(control_id="r1")
        mock_client.execute.reset_mock()
        await action.pause(behavior="silence")
        params = mock_client.execute.call_args[0][1]
        assert mock_client.execute.call_args[0][0] == "calling.record.pause"
        assert params["behavior"] == "silence"

    @pytest.mark.asyncio
    async def test_record_resume(self, call, mock_client):
        action = await call.record(control_id="r1")
        mock_client.execute.reset_mock()
        await action.resume()
        assert mock_client.execute.call_args[0][0] == "calling.record.resume"


class TestDetectActionMethods:
    @pytest.mark.asyncio
    async def test_detect_stop(self, call, mock_client):
        action = await call.detect({"type": "machine"}, control_id="d1")
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.detect.stop"


class TestCollectActionMethods:
    @pytest.mark.asyncio
    async def test_play_and_collect_stop(self, call, mock_client):
        action = await call.play_and_collect(
            [{"type": "tts", "params": {"text": "Press 1"}}],
            {"digits": {"max": 1}},
            control_id="pac1",
        )
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.play_and_collect.stop"

    @pytest.mark.asyncio
    async def test_play_and_collect_volume(self, call, mock_client):
        action = await call.play_and_collect(
            [{"type": "tts", "params": {"text": "Press 1"}}],
            {"digits": {"max": 1}},
            control_id="pac1",
        )
        mock_client.execute.reset_mock()
        await action.volume(5.0)
        params = mock_client.execute.call_args[0][1]
        assert mock_client.execute.call_args[0][0] == "calling.play_and_collect.volume"
        assert params["volume"] == 5.0

    @pytest.mark.asyncio
    async def test_play_and_collect_with_volume_param(self, call, mock_client):
        await call.play_and_collect(
            [{"type": "tts", "params": {"text": "Press 1"}}],
            {"digits": {"max": 1}},
            volume=3.0,
        )
        params = mock_client.execute.call_args[0][1]
        assert params["volume"] == 3.0

    @pytest.mark.asyncio
    async def test_collect_start_input_timers_method(self, call, mock_client):
        action = await call.play_and_collect(
            [{"type": "tts", "params": {"text": "Press 1"}}],
            {"digits": {"max": 1}},
            control_id="pac1",
        )
        mock_client.execute.reset_mock()
        await action.start_input_timers()
        assert mock_client.execute.call_args[0][0] == "calling.collect.start_input_timers"


class TestStandaloneCollectEventRouting:
    @pytest.mark.asyncio
    async def test_standalone_collect_resolves_on_result(self, call, mock_client):
        action = await call.collect(digits={"max": 1}, control_id="col1")
        await call._dispatch_event({
            "event_type": EVENT_CALL_COLLECT,
            "params": {
                "call_id": "call-1",
                "control_id": "col1",
                "result": {"type": "digit", "params": {"digits": "5"}},
            },
        })
        assert action.is_done

    @pytest.mark.asyncio
    async def test_standalone_collect_ignores_non_collect_events(self, call, mock_client):
        action = await call.collect(digits={"max": 1}, control_id="col1")
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "col1", "state": "finished"},
        })
        assert not action.is_done

    @pytest.mark.asyncio
    async def test_standalone_collect_resolves_on_terminal_state(self, call, mock_client):
        action = await call.collect(digits={"max": 1}, control_id="col1")
        await call._dispatch_event({
            "event_type": EVENT_CALL_COLLECT,
            "params": {
                "call_id": "call-1",
                "control_id": "col1",
                "state": "no_input",
            },
        })
        assert action.is_done


class TestCollectOptionalParams:
    @pytest.mark.asyncio
    async def test_collect_with_all_params(self, call, mock_client):
        await call.collect(
            digits={"max": 4},
            speech={"language": "en-US"},
            initial_timeout=10.0,
            partial_results=True,
            continuous=True,
            send_start_of_input=True,
            start_input_timers=False,
            control_id="col1",
        )
        params = mock_client.execute.call_args[0][1]
        assert params["initial_timeout"] == 10.0
        assert params["send_start_of_input"] is True
        assert params["start_input_timers"] is False
        assert params["continuous"] is True


class TestConnectOptionalParams:
    @pytest.mark.asyncio
    async def test_connect_with_all_params(self, call, mock_client):
        await call.connect(
            [[{"type": "phone", "params": {"to_number": "+15551234567"}}]],
            tag="my-tag",
            max_duration=60,
            max_price_per_minute=1.50,
            status_url="https://example.com/status",
        )
        params = mock_client.execute.call_args[0][1]
        assert params["tag"] == "my-tag"
        assert params["max_duration"] == 60
        assert params["max_price_per_minute"] == 1.50
        assert params["status_url"] == "https://example.com/status"


class TestPayAllParams:
    @pytest.mark.asyncio
    async def test_pay_all_optional_params(self, call, mock_client):
        await call.pay(
            "https://pay.example.com",
            control_id="pay1",
            input_method="dtmf",
            status_url="https://status.example.com",
            payment_method="credit-card",
            timeout="30",
            max_attempts="3",
            security_code="required",
            postal_code="required",
            min_postal_code_length="5",
            token_type="reusable",
            charge_amount="25.99",
            currency="usd",
            language="en",
            voice="man",
            description="Test payment",
            valid_card_types="visa mastercard",
            parameters=[{"name": "key", "value": "val"}],
            prompts=[{"name": "greeting", "text": "Enter card"}],
        )
        params = mock_client.execute.call_args[0][1]
        assert params["input"] == "dtmf"
        assert params["status_url"] == "https://status.example.com"
        assert params["payment_method"] == "credit-card"
        assert params["timeout"] == "30"
        assert params["max_attempts"] == "3"
        assert params["security_code"] == "required"
        assert params["postal_code"] == "required"
        assert params["min_postal_code_length"] == "5"
        assert params["token_type"] == "reusable"
        assert params["charge_amount"] == "25.99"
        assert params["currency"] == "usd"
        assert params["language"] == "en"
        assert params["voice"] == "man"
        assert params["description"] == "Test payment"
        assert params["valid_card_types"] == "visa mastercard"
        assert params["parameters"] == [{"name": "key", "value": "val"}]
        assert params["prompts"] == [{"name": "greeting", "text": "Enter card"}]

    @pytest.mark.asyncio
    async def test_pay_action_stop(self, call, mock_client):
        action = await call.pay("https://pay.example.com", control_id="pay1")
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.pay.stop"


class TestTapActionMethods:
    @pytest.mark.asyncio
    async def test_tap_action_stop(self, call, mock_client):
        action = await call.tap(
            {"type": "audio", "params": {"direction": "speak"}},
            {"type": "rtp", "params": {"addr": "1.2.3.4", "port": 5000}},
            control_id="tap1",
        )
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.tap.stop"


class TestStreamAllParams:
    @pytest.mark.asyncio
    async def test_stream_all_params(self, call, mock_client):
        await call.stream(
            "wss://example.com/audio",
            name="my_stream",
            codec="PCMU",
            track="inbound_track",
            status_url="https://status.example.com",
            status_url_method="POST",
            authorization_bearer_token="Bearer abc",
            custom_parameters={"key": "val"},
            control_id="str1",
        )
        params = mock_client.execute.call_args[0][1]
        assert params["status_url"] == "https://status.example.com"
        assert params["status_url_method"] == "POST"
        assert params["authorization_bearer_token"] == "Bearer abc"
        assert params["custom_parameters"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_stream_action_stop(self, call, mock_client):
        action = await call.stream("wss://example.com/audio", control_id="str1")
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.stream.stop"


class TestTranscribeActionMethods:
    @pytest.mark.asyncio
    async def test_transcribe_stop(self, call, mock_client):
        action = await call.transcribe(control_id="tr1")
        mock_client.execute.reset_mock()
        await action.stop()
        assert mock_client.execute.call_args[0][0] == "calling.transcribe.stop"


class TestConferenceAllParams:
    @pytest.mark.asyncio
    async def test_join_conference_all_params(self, call, mock_client):
        await call.join_conference(
            "my_conf",
            muted=True,
            beep="onEnter",
            start_on_enter=True,
            end_on_exit=False,
            wait_url="https://example.com/wait",
            max_participants=50,
            record="record-from-start",
            region="us1",
            trim="trim-silence",
            coach="coach-id",
            status_callback="https://example.com/status",
            status_callback_event="start end",
            status_callback_event_type="webhook",
            status_callback_method="POST",
            recording_status_callback="https://example.com/rec",
            recording_status_callback_event="completed",
            recording_status_callback_event_type="webhook",
            recording_status_callback_method="POST",
            stream_obj={"url": "wss://example.com"},
        )
        params = mock_client.execute.call_args[0][1]
        assert params["muted"] is True
        assert params["start_on_enter"] is True
        assert params["end_on_exit"] is False
        assert params["wait_url"] == "https://example.com/wait"
        assert params["region"] == "us1"
        assert params["trim"] == "trim-silence"
        assert params["coach"] == "coach-id"
        assert params["status_callback"] == "https://example.com/status"
        assert params["status_callback_event"] == "start end"
        assert params["status_callback_event_type"] == "webhook"
        assert params["status_callback_method"] == "POST"
        assert params["recording_status_callback"] == "https://example.com/rec"
        assert params["recording_status_callback_event"] == "completed"
        assert params["recording_status_callback_event_type"] == "webhook"
        assert params["recording_status_callback_method"] == "POST"
        assert params["stream"] == {"url": "wss://example.com"}


class TestEchoOptionalParams:
    @pytest.mark.asyncio
    async def test_echo_with_status_url(self, call, mock_client):
        await call.echo(timeout=30.0, status_url="https://example.com/echo")
        params = mock_client.execute.call_args[0][1]
        assert params["status_url"] == "https://example.com/echo"


class TestAIAllParams:
    @pytest.mark.asyncio
    async def test_ai_with_post_prompt_params(self, call, mock_client):
        await call.ai(
            control_id="ai1",
            prompt={"text": "Hello"},
            post_prompt={"text": "Goodbye"},
            post_prompt_url="https://example.com/post",
            post_prompt_auth_user="user",
            post_prompt_auth_password="pass",
            pronounce=[{"replace": "SW", "with": "SignalWire"}],
            languages=[{"name": "English", "code": "en-US"}],
        )
        params = mock_client.execute.call_args[0][1]
        assert params["post_prompt"]["text"] == "Goodbye"
        assert params["post_prompt_url"] == "https://example.com/post"
        assert params["post_prompt_auth_user"] == "user"
        assert params["post_prompt_auth_password"] == "pass"
        assert params["pronounce"][0]["replace"] == "SW"
        assert params["languages"][0]["code"] == "en-US"


class TestAmazonBedrockAllParams:
    @pytest.mark.asyncio
    async def test_amazon_bedrock_all_params(self, call, mock_client):
        await call.amazon_bedrock(
            prompt="You are helpful.",
            SWAIG={"functions": []},
            ai_params={"end_of_speech_timeout": 3000},
            global_data={"key": "value"},
            post_prompt={"text": "Goodbye"},
            post_prompt_url="https://example.com/post",
        )
        params = mock_client.execute.call_args[0][1]
        assert params["prompt"] == "You are helpful."
        assert params["SWAIG"] == {"functions": []}
        assert params["params"]["end_of_speech_timeout"] == 3000
        assert params["global_data"]["key"] == "value"
        assert params["post_prompt"]["text"] == "Goodbye"
        assert params["post_prompt_url"] == "https://example.com/post"


class TestAIMessageAllParams:
    @pytest.mark.asyncio
    async def test_ai_message_with_control_id(self, call, mock_client):
        await call.ai_message(
            message_text="Hello",
            role="system",
            control_id="ai1",
        )
        params = mock_client.execute.call_args[0][1]
        assert params["control_id"] == "ai1"
        assert params["message_text"] == "Hello"
        assert params["role"] == "system"


class TestQueueOptionalParams:
    @pytest.mark.asyncio
    async def test_queue_leave_with_status_url(self, call, mock_client):
        await call.queue_leave("support", control_id="q1", status_url="https://example.com/q")
        params = mock_client.execute.call_args[0][1]
        assert params["status_url"] == "https://example.com/q"


class TestEventHandlerError:
    @pytest.mark.asyncio
    async def test_listener_exception_does_not_crash(self, call):
        """Verify that an exception in an event handler is caught."""
        def bad_handler(event):
            raise RuntimeError("handler crashed")

        call.on(EVENT_CALL_PLAY, bad_handler)
        # Should not raise
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "ctl1", "state": "playing"},
        })


class TestWaitForTimeout:
    @pytest.mark.asyncio
    async def test_wait_for_timeout_raises(self, call):
        with pytest.raises(asyncio.TimeoutError):
            await call.wait_for(EVENT_CALL_PLAY, timeout=0.01)

    @pytest.mark.asyncio
    async def test_wait_for_no_timeout(self, call):
        """wait_for without timeout resolves when event arrives."""
        async def send_later():
            await asyncio.sleep(0.01)
            await call._dispatch_event({
                "event_type": EVENT_CALL_PLAY,
                "params": {"call_id": "call-1", "control_id": "x", "state": "finished"},
            })

        task = asyncio.create_task(send_later())
        event = await asyncio.wait_for(
            call.wait_for(EVENT_CALL_PLAY),
            timeout=2.0,
        )
        assert event.params["state"] == "finished"
        await task


class TestOnCompleted:
    @pytest.mark.asyncio
    async def test_on_completed_sync_callback(self, call, mock_client):
        results = []
        action = await call.play(
            [{"type": "tts", "params": {"text": "Hi"}}],
            control_id="ctl1",
            on_completed=lambda event: results.append(event),
        )
        # Simulate play finished
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "ctl1", "state": "finished"},
        })
        assert action.is_done
        assert len(results) == 1
        assert results[0].params["state"] == "finished"

    @pytest.mark.asyncio
    async def test_on_completed_async_callback(self, call, mock_client):
        results = []

        async def on_done(event):
            results.append(event)

        action = await call.play(
            [{"type": "tts", "params": {"text": "Hi"}}],
            control_id="ctl1",
            on_completed=on_done,
        )
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "ctl1", "state": "finished"},
        })
        await asyncio.sleep(0.01)  # let the coroutine task run
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_on_completed_not_called_on_non_terminal(self, call, mock_client):
        results = []
        action = await call.play(
            [{"type": "tts", "params": {"text": "Hi"}}],
            control_id="ctl1",
            on_completed=lambda event: results.append(event),
        )
        # Non-terminal state — callback should NOT fire
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "ctl1", "state": "playing"},
        })
        assert not action.is_done
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_on_completed_error_does_not_crash(self, call, mock_client):
        def bad_callback(event):
            raise RuntimeError("callback error")

        action = await call.play(
            [{"type": "tts", "params": {"text": "Hi"}}],
            control_id="ctl1",
            on_completed=bad_callback,
        )
        # Should not raise
        await call._dispatch_event({
            "event_type": EVENT_CALL_PLAY,
            "params": {"call_id": "call-1", "control_id": "ctl1", "state": "finished"},
        })
        assert action.is_done

    @pytest.mark.asyncio
    async def test_on_completed_on_record(self, call, mock_client):
        results = []
        from signalwire_agents.relay.constants import EVENT_CALL_RECORD
        action = await call.record(
            control_id="r1",
            on_completed=lambda event: results.append(event),
        )
        await call._dispatch_event({
            "event_type": EVENT_CALL_RECORD,
            "params": {"call_id": "call-1", "control_id": "r1", "state": "finished"},
        })
        assert action.is_done
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_on_completed_on_call_gone(self, call, mock_client):
        """on_completed fires even when call is gone (404)."""
        mock_client.execute.side_effect = MockRelayError(404, "Call not found")
        results = []
        action = await call.play(
            [{"type": "tts", "params": {"text": "Hi"}}],
            control_id="ctl1",
            on_completed=lambda event: results.append(event),
        )
        assert action.is_done
        assert len(results) == 1


class TestCallRepr:
    def test_repr(self, call):
        r = repr(call)
        assert "call-1" in r
        assert "answered" in r
        assert "inbound" in r
