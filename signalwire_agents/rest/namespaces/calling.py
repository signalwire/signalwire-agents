"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Calling API namespace — REST-based call control via command dispatch.

All commands are sent as POST /api/calling/calls with a command field.
"""

from .._base import BaseResource


class CallingNamespace(BaseResource):
    """REST call control — all 37 commands dispatched via single POST endpoint."""

    def __init__(self, http):
        super().__init__(http, "/api/calling/calls")

    def _execute(self, command, call_id=None, **params):
        body = {"command": command, "params": params}
        if call_id is not None:
            body["id"] = call_id
        return self._http.post(self._base_path, body=body)

    # Call lifecycle
    def dial(self, **params):
        return self._execute("dial", **params)

    def update(self, **params):
        return self._execute("update", **params)

    def end(self, call_id, **params):
        return self._execute("calling.end", call_id, **params)

    def transfer(self, call_id, **params):
        return self._execute("calling.transfer", call_id, **params)

    def disconnect(self, call_id, **params):
        return self._execute("calling.disconnect", call_id, **params)

    # Play
    def play(self, call_id, **params):
        return self._execute("calling.play", call_id, **params)

    def play_pause(self, call_id, **params):
        return self._execute("calling.play.pause", call_id, **params)

    def play_resume(self, call_id, **params):
        return self._execute("calling.play.resume", call_id, **params)

    def play_stop(self, call_id, **params):
        return self._execute("calling.play.stop", call_id, **params)

    def play_volume(self, call_id, **params):
        return self._execute("calling.play.volume", call_id, **params)

    # Record
    def record(self, call_id, **params):
        return self._execute("calling.record", call_id, **params)

    def record_pause(self, call_id, **params):
        return self._execute("calling.record.pause", call_id, **params)

    def record_resume(self, call_id, **params):
        return self._execute("calling.record.resume", call_id, **params)

    def record_stop(self, call_id, **params):
        return self._execute("calling.record.stop", call_id, **params)

    # Collect
    def collect(self, call_id, **params):
        return self._execute("calling.collect", call_id, **params)

    def collect_stop(self, call_id, **params):
        return self._execute("calling.collect.stop", call_id, **params)

    def collect_start_input_timers(self, call_id, **params):
        return self._execute("calling.collect.start_input_timers", call_id, **params)

    # Detect
    def detect(self, call_id, **params):
        return self._execute("calling.detect", call_id, **params)

    def detect_stop(self, call_id, **params):
        return self._execute("calling.detect.stop", call_id, **params)

    # Tap
    def tap(self, call_id, **params):
        return self._execute("calling.tap", call_id, **params)

    def tap_stop(self, call_id, **params):
        return self._execute("calling.tap.stop", call_id, **params)

    # Stream
    def stream(self, call_id, **params):
        return self._execute("calling.stream", call_id, **params)

    def stream_stop(self, call_id, **params):
        return self._execute("calling.stream.stop", call_id, **params)

    # Denoise
    def denoise(self, call_id, **params):
        return self._execute("calling.denoise", call_id, **params)

    def denoise_stop(self, call_id, **params):
        return self._execute("calling.denoise.stop", call_id, **params)

    # Transcribe
    def transcribe(self, call_id, **params):
        return self._execute("calling.transcribe", call_id, **params)

    def transcribe_stop(self, call_id, **params):
        return self._execute("calling.transcribe.stop", call_id, **params)

    # AI
    def ai_message(self, call_id, **params):
        return self._execute("calling.ai_message", call_id, **params)

    def ai_hold(self, call_id, **params):
        return self._execute("calling.ai_hold", call_id, **params)

    def ai_unhold(self, call_id, **params):
        return self._execute("calling.ai_unhold", call_id, **params)

    def ai_stop(self, call_id, **params):
        return self._execute("calling.ai.stop", call_id, **params)

    # Live transcribe / translate
    def live_transcribe(self, call_id, **params):
        return self._execute("calling.live_transcribe", call_id, **params)

    def live_translate(self, call_id, **params):
        return self._execute("calling.live_translate", call_id, **params)

    # Fax
    def send_fax_stop(self, call_id, **params):
        return self._execute("calling.send_fax.stop", call_id, **params)

    def receive_fax_stop(self, call_id, **params):
        return self._execute("calling.receive_fax.stop", call_id, **params)

    # SIP
    def refer(self, call_id, **params):
        return self._execute("calling.refer", call_id, **params)

    # Custom events
    def user_event(self, call_id, **params):
        return self._execute("calling.user_event", call_id, **params)
