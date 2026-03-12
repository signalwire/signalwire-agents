"""Typed event wrappers for RELAY calling events.

These are convenience dataclasses over raw event dicts. All Call event handlers
also accept the raw dict, so these are optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RelayEvent:
    """Base event — wraps the raw params dict from a signalwire.event message."""

    event_type: str
    params: dict[str, Any]
    call_id: str = ""
    timestamp: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RelayEvent":
        event_type = payload.get("event_type", "")
        params = payload.get("params", {})
        return cls(
            event_type=event_type,
            params=params,
            call_id=params.get("call_id", ""),
            timestamp=params.get("timestamp", 0.0),
        )


@dataclass
class CallStateEvent(RelayEvent):
    """Event for calling.call.state."""

    call_state: str = ""
    end_reason: str = ""
    direction: str = ""
    device: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CallStateEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            call_state=p.get("call_state", ""),
            end_reason=p.get("end_reason", ""),
            direction=p.get("direction", ""),
            device=p.get("device", {}),
        )


@dataclass
class CallReceiveEvent(RelayEvent):
    """Event for calling.call.receive — inbound call notification."""

    call_state: str = ""
    direction: str = ""
    device: dict[str, Any] = field(default_factory=dict)
    node_id: str = ""
    project_id: str = ""
    context: str = ""
    segment_id: str = ""
    tag: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CallReceiveEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            call_state=p.get("call_state", ""),
            direction=p.get("direction", ""),
            device=p.get("device", {}),
            node_id=p.get("node_id", ""),
            project_id=p.get("project_id", ""),
            context=p.get("context", p.get("protocol", "")),
            segment_id=p.get("segment_id", ""),
            tag=p.get("tag", ""),
        )


@dataclass
class PlayEvent(RelayEvent):
    """Event for calling.call.play."""

    control_id: str = ""
    state: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PlayEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            state=p.get("state", ""),
        )


@dataclass
class RecordEvent(RelayEvent):
    """Event for calling.call.record."""

    control_id: str = ""
    state: str = ""
    url: str = ""
    duration: float = 0.0
    size: int = 0
    record: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RecordEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        rec = p.get("record", {})
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            state=p.get("state", ""),
            url=rec.get("url", p.get("url", "")),
            duration=rec.get("duration", p.get("duration", 0.0)),
            size=rec.get("size", p.get("size", 0)),
            record=rec,
        )


@dataclass
class CollectEvent(RelayEvent):
    """Event for calling.call.collect."""

    control_id: str = ""
    state: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    final: Optional[bool] = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CollectEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            state=p.get("state", ""),
            result=p.get("result", {}),
            final=p.get("final"),
        )


@dataclass
class ConnectEvent(RelayEvent):
    """Event for calling.call.connect."""

    connect_state: str = ""
    peer: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ConnectEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            connect_state=p.get("connect_state", ""),
            peer=p.get("peer", {}),
        )


@dataclass
class DetectEvent(RelayEvent):
    """Event for calling.call.detect."""

    control_id: str = ""
    detect: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DetectEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            detect=p.get("detect", {}),
        )


@dataclass
class FaxEvent(RelayEvent):
    """Event for calling.call.fax."""

    control_id: str = ""
    fax: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FaxEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            fax=p.get("fax", {}),
        )


@dataclass
class TapEvent(RelayEvent):
    """Event for calling.call.tap."""

    control_id: str = ""
    state: str = ""
    tap: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TapEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            state=p.get("state", ""),
            tap=p.get("tap", {}),
            device=p.get("device", {}),
        )


@dataclass
class StreamEvent(RelayEvent):
    """Event for calling.call.stream."""

    control_id: str = ""
    state: str = ""
    url: str = ""
    name: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StreamEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            state=p.get("state", ""),
            url=p.get("url", ""),
            name=p.get("name", ""),
        )


@dataclass
class SendDigitsEvent(RelayEvent):
    """Event for calling.call.send_digits."""

    control_id: str = ""
    state: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SendDigitsEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            state=p.get("state", ""),
        )


@dataclass
class DialEvent(RelayEvent):
    """Event for calling.call.dial."""

    tag: str = ""
    dial_state: str = ""
    call: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DialEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            tag=p.get("tag", ""),
            dial_state=p.get("dial_state", ""),
            call=p.get("call", {}),
        )


@dataclass
class ReferEvent(RelayEvent):
    """Event for calling.call.refer."""

    state: str = ""
    sip_refer_to: str = ""
    sip_refer_response_code: str = ""
    sip_notify_response_code: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReferEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            state=p.get("state", ""),
            sip_refer_to=p.get("sip_refer_to", ""),
            sip_refer_response_code=p.get("sip_refer_response_code", ""),
            sip_notify_response_code=p.get("sip_notify_response_code", ""),
        )


@dataclass
class DenoiseEvent(RelayEvent):
    """Event for calling.call.denoise."""

    denoised: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DenoiseEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            denoised=p.get("denoised", False),
        )


@dataclass
class PayEvent(RelayEvent):
    """Event for calling.call.pay."""

    control_id: str = ""
    state: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PayEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            state=p.get("state", ""),
        )


@dataclass
class QueueEvent(RelayEvent):
    """Event for calling.call.queue."""

    control_id: str = ""
    status: str = ""
    queue_id: str = ""
    queue_name: str = ""
    position: int = 0
    size: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "QueueEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            status=p.get("status", ""),
            queue_id=p.get("id", ""),
            queue_name=p.get("name", ""),
            position=p.get("position", 0),
            size=p.get("size", 0),
        )


@dataclass
class EchoEvent(RelayEvent):
    """Event for calling.call.echo."""

    state: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EchoEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            state=p.get("state", ""),
        )


@dataclass
class TranscribeEvent(RelayEvent):
    """Event for calling.call.transcribe."""

    control_id: str = ""
    state: str = ""
    url: str = ""
    recording_id: str = ""
    duration: float = 0.0
    size: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TranscribeEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            control_id=p.get("control_id", ""),
            state=p.get("state", ""),
            url=p.get("url", ""),
            recording_id=p.get("recording_id", ""),
            duration=p.get("duration", 0.0),
            size=p.get("size", 0),
        )


@dataclass
class HoldEvent(RelayEvent):
    """Event for calling.call.hold."""

    state: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HoldEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            state=p.get("state", ""),
        )


@dataclass
class ConferenceEvent(RelayEvent):
    """Event for calling.conference."""

    conference_id: str = ""
    name: str = ""
    status: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ConferenceEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            conference_id=p.get("conference_id", ""),
            name=p.get("name", ""),
            status=p.get("status", ""),
        )


@dataclass
class CallingErrorEvent(RelayEvent):
    """Event for calling.error."""

    code: str = ""
    message: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CallingErrorEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            code=p.get("code", ""),
            message=p.get("message", ""),
        )


@dataclass
class MessageReceiveEvent(RelayEvent):
    """Event for messaging.receive — inbound message notification."""

    message_id: str = ""
    context: str = ""
    direction: str = ""
    from_number: str = ""
    to_number: str = ""
    body: str = ""
    media: list[str] = field(default_factory=list)
    segments: int = 0
    message_state: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MessageReceiveEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            message_id=p.get("message_id", ""),
            context=p.get("context", ""),
            direction=p.get("direction", ""),
            from_number=p.get("from_number", ""),
            to_number=p.get("to_number", ""),
            body=p.get("body", ""),
            media=p.get("media", []),
            segments=p.get("segments", 0),
            message_state=p.get("message_state", ""),
            tags=p.get("tags", []),
        )


@dataclass
class MessageStateEvent(RelayEvent):
    """Event for messaging.state — outbound message state change."""

    message_id: str = ""
    context: str = ""
    direction: str = ""
    from_number: str = ""
    to_number: str = ""
    body: str = ""
    media: list[str] = field(default_factory=list)
    segments: int = 0
    message_state: str = ""
    reason: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MessageStateEvent":
        base = RelayEvent.from_payload(payload)
        p = base.params
        return cls(
            event_type=base.event_type,
            params=base.params,
            call_id=base.call_id,
            timestamp=base.timestamp,
            message_id=p.get("message_id", ""),
            context=p.get("context", ""),
            direction=p.get("direction", ""),
            from_number=p.get("from_number", ""),
            to_number=p.get("to_number", ""),
            body=p.get("body", ""),
            media=p.get("media", []),
            segments=p.get("segments", 0),
            message_state=p.get("message_state", ""),
            reason=p.get("reason", ""),
            tags=p.get("tags", []),
        )


# Map event_type string → typed event class
EVENT_CLASS_MAP: dict[str, type[RelayEvent]] = {
    "calling.call.state": CallStateEvent,
    "calling.call.receive": CallReceiveEvent,
    "calling.call.play": PlayEvent,
    "calling.call.record": RecordEvent,
    "calling.call.collect": CollectEvent,
    "calling.call.connect": ConnectEvent,
    "calling.call.detect": DetectEvent,
    "calling.call.fax": FaxEvent,
    "calling.call.tap": TapEvent,
    "calling.call.stream": StreamEvent,
    "calling.call.send_digits": SendDigitsEvent,
    "calling.call.dial": DialEvent,
    "calling.call.refer": ReferEvent,
    "calling.call.denoise": DenoiseEvent,
    "calling.call.pay": PayEvent,
    "calling.call.queue": QueueEvent,
    "calling.call.echo": EchoEvent,
    "calling.call.transcribe": TranscribeEvent,
    "calling.call.hold": HoldEvent,
    "calling.conference": ConferenceEvent,
    "calling.error": CallingErrorEvent,
    "messaging.receive": MessageReceiveEvent,
    "messaging.state": MessageStateEvent,
}


def parse_event(payload: dict[str, Any]) -> RelayEvent:
    """Parse a raw signalwire.event params dict into a typed event object."""
    event_type = payload.get("event_type", "")
    cls = EVENT_CLASS_MAP.get(event_type, RelayEvent)
    return cls.from_payload(payload)
