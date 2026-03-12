"""Call object — represents a live RELAY call with command methods."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Coroutine, Optional, TYPE_CHECKING

from signalwire_agents.core.logging_config import get_logger

from .constants import (
    CALL_STATE_ENDED,
    EVENT_CALL_PLAY,
    EVENT_CALL_RECORD,
    EVENT_CALL_DETECT,
    EVENT_CALL_STATE,
    EVENT_CALL_COLLECT,
    EVENT_CALL_FAX,
    EVENT_CALL_TAP,
    EVENT_CALL_STREAM,
    EVENT_CALL_PAY,
    EVENT_CALL_TRANSCRIBE,
    EVENT_CALL_ECHO,
    PLAY_STATE_FINISHED,
    PLAY_STATE_ERROR,
    RECORD_STATE_FINISHED,
    RECORD_STATE_NO_INPUT,
)
from .event import RelayEvent, parse_event

if TYPE_CHECKING:
    from .client import RelayClient

logger = get_logger("relay_call")

EventHandler = Callable[[RelayEvent], Coroutine[Any, Any, None] | None]


# ======================================================================
# Action classes — async handles for controllable operations
# ======================================================================

class Action:
    """Base class for async action handles (play, record, detect, etc.).

    Holds a control_id and back-reference to the Call. Resolves when the
    server sends a terminal event for this control_id.
    """

    def __init__(self, call: "Call", control_id: str, terminal_event: str,
                 terminal_states: tuple[str, ...]):
        self.call = call
        self.control_id = control_id
        self._terminal_event = terminal_event
        self._terminal_states = terminal_states
        self._done: asyncio.Future[RelayEvent] = asyncio.get_running_loop().create_future()
        self.result: Optional[RelayEvent] = None
        self.completed = False
        self._on_completed: Optional[Callable[[RelayEvent], Any]] = None

    def _check_event(self, event: RelayEvent) -> None:
        """Called by Call when an event matches our control_id."""
        state = event.params.get("state", "")
        if state in self._terminal_states and not self._done.done():
            self._resolve(event)

    def _resolve(self, event: RelayEvent) -> None:
        """Mark the action as completed and fire the on_completed callback."""
        self.result = event
        self.completed = True
        self._done.set_result(event)
        if self._on_completed is not None:
            try:
                result = self._on_completed(event)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                logger.exception(f"Error in on_completed callback for {self.control_id}")

    async def wait(self, timeout: Optional[float] = None) -> RelayEvent:
        """Wait for the action to complete. Returns the terminal event."""
        if timeout is not None:
            return await asyncio.wait_for(self._done, timeout=timeout)
        return await self._done

    @property
    def is_done(self) -> bool:
        return self._done.done()


class PlayAction(Action):
    """Handle for an active play operation."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_PLAY,
                         (PLAY_STATE_FINISHED, PLAY_STATE_ERROR))

    async def stop(self) -> dict:
        return await self.call._execute("play.stop", {"control_id": self.control_id})

    async def pause(self) -> dict:
        return await self.call._execute("play.pause", {"control_id": self.control_id})

    async def resume(self) -> dict:
        return await self.call._execute("play.resume", {"control_id": self.control_id})

    async def volume(self, volume: float) -> dict:
        return await self.call._execute("play.volume", {
            "control_id": self.control_id, "volume": volume,
        })


class RecordAction(Action):
    """Handle for an active record operation."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_RECORD,
                         (RECORD_STATE_FINISHED, RECORD_STATE_NO_INPUT))

    async def stop(self) -> dict:
        return await self.call._execute("record.stop", {"control_id": self.control_id})

    async def pause(self, behavior: Optional[str] = None) -> dict:
        params: dict[str, Any] = {"control_id": self.control_id}
        if behavior:
            params["behavior"] = behavior
        return await self.call._execute("record.pause", params)

    async def resume(self) -> dict:
        return await self.call._execute("record.resume", {"control_id": self.control_id})


class DetectAction(Action):
    """Handle for an active detect operation."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_DETECT, ("finished", "error"))

    def _check_event(self, event: RelayEvent) -> None:
        # Detect delivers results continuously. Resolve on first result or
        # when finished/error.
        detect = event.params.get("detect", {})
        state = event.params.get("state", "")
        if (detect or state in self._terminal_states) and not self._done.done():
            self._resolve(event)

    async def stop(self) -> dict:
        return await self.call._execute("detect.stop", {"control_id": self.control_id})


class CollectAction(Action):
    """Handle for play_and_collect or standalone collect."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_COLLECT,
                         ("finished", "error", "no_input", "no_match"))

    def _check_event(self, event: RelayEvent) -> None:
        # play_and_collect shares a control_id across play and collect
        # phases.  Only resolve on collect events, not play events.
        if event.event_type != EVENT_CALL_COLLECT:
            return
        result = event.params.get("result", {})
        if result and not self._done.done():
            self._resolve(event)
        else:
            super()._check_event(event)

    async def stop(self) -> dict:
        return await self.call._execute("play_and_collect.stop",
                                        {"control_id": self.control_id})

    async def volume(self, volume: float) -> dict:
        return await self.call._execute("play_and_collect.volume", {
            "control_id": self.control_id, "volume": volume,
        })

    async def start_input_timers(self) -> dict:
        """Start the initial_timeout timer on an active collect."""
        return await self.call._execute("collect.start_input_timers",
                                        {"control_id": self.control_id})


class StandaloneCollectAction(Action):
    """Handle for standalone calling.collect (without play)."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_COLLECT,
                         ("finished", "error", "no_input", "no_match"))

    def _check_event(self, event: RelayEvent) -> None:
        if event.event_type != EVENT_CALL_COLLECT:
            return
        result = event.params.get("result", {})
        state = event.params.get("state", "")
        if (result or state in self._terminal_states) and not self._done.done():
            self._resolve(event)

    async def stop(self) -> dict:
        return await self.call._execute("collect.stop",
                                        {"control_id": self.control_id})

    async def start_input_timers(self) -> dict:
        """Start the initial_timeout timer on an active collect."""
        return await self.call._execute("collect.start_input_timers",
                                        {"control_id": self.control_id})


class FaxAction(Action):
    """Handle for an active send_fax or receive_fax operation."""

    def __init__(self, call: "Call", control_id: str, method_prefix: str):
        super().__init__(call, control_id, EVENT_CALL_FAX,
                         ("finished", "error"))
        self._method_prefix = method_prefix

    async def stop(self) -> dict:
        return await self.call._execute(f"{self._method_prefix}.stop",
                                        {"control_id": self.control_id})


class TapAction(Action):
    """Handle for an active tap operation."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_TAP,
                         ("finished",))

    async def stop(self) -> dict:
        return await self.call._execute("tap.stop", {"control_id": self.control_id})


class StreamAction(Action):
    """Handle for an active stream operation."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_STREAM,
                         ("finished",))

    async def stop(self) -> dict:
        return await self.call._execute("stream.stop", {"control_id": self.control_id})


class PayAction(Action):
    """Handle for an active pay operation."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_PAY,
                         ("finished", "error"))

    async def stop(self) -> dict:
        return await self.call._execute("pay.stop", {"control_id": self.control_id})


class TranscribeAction(Action):
    """Handle for an active transcribe operation."""

    def __init__(self, call: "Call", control_id: str):
        super().__init__(call, control_id, EVENT_CALL_TRANSCRIBE,
                         ("finished",))

    async def stop(self) -> dict:
        return await self.call._execute("transcribe.stop", {"control_id": self.control_id})


class AIAction(Action):
    """Handle for an active AI agent session."""

    def __init__(self, call: "Call", control_id: str):
        # AI sessions don't have a standard event type with state field —
        # they end when the call ends or when stopped. We treat "finished"
        # and "error" as terminal states from calling.call.ai events if any.
        super().__init__(call, control_id, "calling.call.ai",
                         ("finished", "error"))

    async def stop(self) -> dict:
        return await self.call._execute("ai.stop", {"control_id": self.control_id})


# ======================================================================
# Call class
# ======================================================================

class Call:
    """Represents a live RELAY call.

    Created by RelayClient on inbound ``calling.call.receive`` events or
    outbound ``dial``/``begin`` responses.
    """

    def __init__(
        self,
        client: "RelayClient",
        call_id: str,
        node_id: str,
        project_id: str,
        context: str,
        *,
        tag: str = "",
        direction: str = "",
        device: Optional[dict[str, Any]] = None,
        state: str = "",
        segment_id: str = "",
    ):
        self._client = client
        self.call_id = call_id
        self.node_id = node_id
        self.project_id = project_id
        self.context = context
        self.tag = tag
        self.direction = direction
        self.device = device or {}
        self.state = state
        self.segment_id = segment_id

        # Event listeners: event_type -> list of handlers
        self._listeners: dict[str, list[EventHandler]] = {}
        # Active actions indexed by control_id
        self._actions: dict[str, Action] = {}
        # Future that resolves when the call ends
        self._ended: asyncio.Future[RelayEvent] = asyncio.get_running_loop().create_future()

    # ------------------------------------------------------------------
    # Internal RPC primitive
    # ------------------------------------------------------------------

    async def _execute(self, method: str, extra_params: Optional[dict[str, Any]] = None) -> dict:
        """Send a ``calling.<method>`` JSON-RPC request for this call.

        The outer JSON-RPC method is ``"calling.<method>"`` (e.g.
        ``"calling.answer"``) with ``node_id`` and ``call_id`` in params.

        All server errors are logged gracefully and return an empty dict
        rather than raising exceptions.  This is an SDK — stack traces
        from server-side errors should never bubble up to the developer.
        """
        rpc_method = f"calling.{method}"
        params: dict[str, Any] = {
            "node_id": self.node_id,
            "call_id": self.call_id,
        }
        if extra_params:
            params.update(extra_params)
        try:
            return await self._client.execute(rpc_method, params)
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code is not None:
                logger.warning(f"Call {self.call_id} error during {method} (code={code}): {exc}")
                return {}
            raise

    # ------------------------------------------------------------------
    # Event plumbing
    # ------------------------------------------------------------------

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register an event listener for this call."""
        self._listeners.setdefault(event_type, []).append(handler)

    async def _dispatch_event(self, payload: dict[str, Any]) -> None:
        """Called by RelayClient when an event arrives for this call."""
        event = parse_event(payload)
        event_type = event.event_type

        # Update call state
        if event_type == EVENT_CALL_STATE:
            self.state = event.params.get("call_state", self.state)
            if self.state == CALL_STATE_ENDED and not self._ended.done():
                self._ended.set_result(event)

        # Route to active actions by control_id
        control_id = event.params.get("control_id", "")
        if control_id and control_id in self._actions:
            action = self._actions[control_id]
            action._check_event(event)
            # Clean up completed actions
            if action.completed:
                self._actions.pop(control_id, None)

        # Notify registered listeners (snapshot list to allow modification during iteration)
        handlers = list(self._listeners.get(event_type, []))
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                logger.exception(f"Error in event handler for {event_type}")

    async def wait_for(
        self,
        event_type: str,
        predicate: Optional[Callable[[RelayEvent], bool]] = None,
        timeout: Optional[float] = None,
    ) -> RelayEvent:
        """Wait for a specific event, optionally filtered by predicate."""
        future: asyncio.Future[RelayEvent] = asyncio.get_running_loop().create_future()

        def _handler(event: RelayEvent) -> None:
            if future.done():
                return
            if predicate is None or predicate(event):
                future.set_result(event)

        self.on(event_type, _handler)
        try:
            if timeout is not None:
                return await asyncio.wait_for(future, timeout=timeout)
            return await future
        finally:
            # Clean up the one-shot handler
            if event_type in self._listeners:
                try:
                    self._listeners[event_type].remove(_handler)
                except ValueError:
                    pass

    async def wait_for_ended(self, timeout: Optional[float] = None) -> RelayEvent:
        """Wait for the call to reach the ended state."""
        if timeout is not None:
            return await asyncio.wait_for(self._ended, timeout=timeout)
        return await self._ended

    # ------------------------------------------------------------------
    # Action helper
    # ------------------------------------------------------------------

    async def _start_action(
        self,
        action: Action,
        method: str,
        params: dict[str, Any],
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
    ) -> Action:
        """Register an action, execute the RPC, and clean up on failure.

        If _execute raises, the action is removed from _actions and its
        Future is rejected so callers of action.wait() get the error
        instead of hanging forever.

        If the call is already gone (404/410 handled by _execute), the
        action is immediately marked completed so action.wait() returns
        right away instead of hanging.

        Args:
            on_completed: Optional callback invoked when the action reaches
                a terminal state.  Can be a regular function or a coroutine
                function — coroutines are scheduled as tasks.
        """
        if self.state == CALL_STATE_ENDED:
            logger.warning(f"Call {self.call_id} already ended, skipping {method}")
            gone_event = RelayEvent(event_type="", params={})
            action._resolve(gone_event)
            return action
        action._on_completed = on_completed
        self._actions[action.control_id] = action
        try:
            result = await self._execute(method, params)
        except Exception as exc:
            self._actions.pop(action.control_id, None)
            if not action._done.done():
                action._done.set_exception(exc)
            raise
        # _execute returns {} when the call is gone (404/410) — resolve
        # the action immediately so action.wait() doesn't hang forever.
        if not result:
            self._actions.pop(action.control_id, None)
            if not action._done.done():
                gone_event = RelayEvent(event_type="", params={})
                action._resolve(gone_event)
        return action

    # ------------------------------------------------------------------
    # Call lifecycle methods
    # ------------------------------------------------------------------

    async def answer(self, **kwargs: Any) -> dict:
        """Answer an inbound call."""
        return await self._execute("answer", kwargs or None)

    async def hangup(self, reason: str = "hangup") -> dict:
        """End/hang up the call."""
        return await self._execute("end", {"reason": reason})

    async def pass_(self) -> dict:
        """Decline control of an inbound call, returning it to routing."""
        return await self._execute("pass")

    # ------------------------------------------------------------------
    # Audio playback
    # ------------------------------------------------------------------

    async def play(
        self,
        media: list[dict[str, Any]],
        *,
        volume: Optional[float] = None,
        direction: Optional[str] = None,
        loop: Optional[int] = None,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> PlayAction:
        """Play audio content. Returns a PlayAction for stop/pause/resume/wait."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {"control_id": cid, "play": media}
        if volume is not None:
            params["volume"] = volume
        if direction is not None:
            params["direction"] = direction
        if loop is not None:
            params["loop"] = loop
        params.update(kwargs)
        action = PlayAction(self, cid)
        return await self._start_action(action, "play", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record(
        self,
        audio: Optional[dict[str, Any]] = None,
        *,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> RecordAction:
        """Record audio from the call. Returns a RecordAction."""
        cid = control_id or str(uuid.uuid4())
        record_obj = {"audio": audio or {}}
        params: dict[str, Any] = {"control_id": cid, "record": record_obj}
        params.update(kwargs)
        action = RecordAction(self, cid)
        return await self._start_action(action, "record", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # Input collection
    # ------------------------------------------------------------------

    async def play_and_collect(
        self,
        media: list[dict[str, Any]],
        collect: dict[str, Any],
        *,
        volume: Optional[float] = None,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> CollectAction:
        """Play audio and collect digit/speech input."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {
            "control_id": cid,
            "play": media,
            "collect": collect,
        }
        if volume is not None:
            params["volume"] = volume
        params.update(kwargs)
        action = CollectAction(self, cid)
        return await self._start_action(action, "play_and_collect", params, on_completed=on_completed)

    async def collect(
        self,
        *,
        digits: Optional[dict[str, Any]] = None,
        speech: Optional[dict[str, Any]] = None,
        initial_timeout: Optional[float] = None,
        partial_results: Optional[bool] = None,
        continuous: Optional[bool] = None,
        send_start_of_input: Optional[bool] = None,
        start_input_timers: Optional[bool] = None,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> StandaloneCollectAction:
        """Collect digit/speech input without playing media."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {"control_id": cid}
        if digits is not None:
            params["digits"] = digits
        if speech is not None:
            params["speech"] = speech
        if initial_timeout is not None:
            params["initial_timeout"] = initial_timeout
        if partial_results is not None:
            params["partial_results"] = partial_results
        if continuous is not None:
            params["continuous"] = continuous
        if send_start_of_input is not None:
            params["send_start_of_input"] = send_start_of_input
        if start_input_timers is not None:
            params["start_input_timers"] = start_input_timers
        params.update(kwargs)
        action = StandaloneCollectAction(self, cid)
        return await self._start_action(action, "collect", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # Bridging & connectivity
    # ------------------------------------------------------------------

    async def connect(
        self,
        devices: list[list[dict[str, Any]]],
        *,
        ringback: Optional[list[dict[str, Any]]] = None,
        tag: Optional[str] = None,
        max_duration: Optional[int] = None,
        max_price_per_minute: Optional[float] = None,
        status_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Bridge the call to one or more destinations."""
        params: dict[str, Any] = {"devices": devices}
        if ringback is not None:
            params["ringback"] = ringback
        if tag is not None:
            params["tag"] = tag
        if max_duration is not None:
            params["max_duration"] = max_duration
        if max_price_per_minute is not None:
            params["max_price_per_minute"] = max_price_per_minute
        if status_url is not None:
            params["status_url"] = status_url
        params.update(kwargs)
        return await self._execute("connect", params)

    async def disconnect(self) -> dict:
        """Disconnect (unbridge) a connected call."""
        return await self._execute("disconnect")

    # ------------------------------------------------------------------
    # DTMF
    # ------------------------------------------------------------------

    async def send_digits(
        self,
        digits: str,
        *,
        control_id: Optional[str] = None,
    ) -> dict:
        """Send DTMF digits on the call."""
        cid = control_id or str(uuid.uuid4())
        return await self._execute("send_digits", {
            "control_id": cid, "digits": digits,
        })

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def detect(
        self,
        detect: dict[str, Any],
        *,
        timeout: Optional[float] = None,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> DetectAction:
        """Start audio detection (machine, fax, digit). Returns a DetectAction."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {"control_id": cid, "detect": detect}
        if timeout is not None:
            params["timeout"] = timeout
        params.update(kwargs)
        action = DetectAction(self, cid)
        return await self._start_action(action, "detect", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # SIP Refer (transfer)
    # ------------------------------------------------------------------

    async def refer(
        self,
        device: dict[str, Any],
        *,
        status_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Transfer a SIP call to an external SIP endpoint via REFER."""
        params: dict[str, Any] = {"device": device}
        if status_url is not None:
            params["status_url"] = status_url
        params.update(kwargs)
        return await self._execute("refer", params)

    # ------------------------------------------------------------------
    # Payment
    # ------------------------------------------------------------------

    async def pay(
        self,
        payment_connector_url: str,
        *,
        control_id: Optional[str] = None,
        input_method: Optional[str] = None,
        status_url: Optional[str] = None,
        payment_method: Optional[str] = None,
        timeout: Optional[str] = None,
        max_attempts: Optional[str] = None,
        security_code: Optional[str] = None,
        postal_code: Optional[str] = None,
        min_postal_code_length: Optional[str] = None,
        token_type: Optional[str] = None,
        charge_amount: Optional[str] = None,
        currency: Optional[str] = None,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        description: Optional[str] = None,
        valid_card_types: Optional[str] = None,
        parameters: Optional[list[dict[str, Any]]] = None,
        prompts: Optional[list[dict[str, Any]]] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> PayAction:
        """Start a payment collection. Returns a PayAction."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {
            "control_id": cid,
            "payment_connector_url": payment_connector_url,
        }
        if input_method is not None:
            params["input"] = input_method
        if status_url is not None:
            params["status_url"] = status_url
        if payment_method is not None:
            params["payment_method"] = payment_method
        if timeout is not None:
            params["timeout"] = timeout
        if max_attempts is not None:
            params["max_attempts"] = max_attempts
        if security_code is not None:
            params["security_code"] = security_code
        if postal_code is not None:
            params["postal_code"] = postal_code
        if min_postal_code_length is not None:
            params["min_postal_code_length"] = min_postal_code_length
        if token_type is not None:
            params["token_type"] = token_type
        if charge_amount is not None:
            params["charge_amount"] = charge_amount
        if currency is not None:
            params["currency"] = currency
        if language is not None:
            params["language"] = language
        if voice is not None:
            params["voice"] = voice
        if description is not None:
            params["description"] = description
        if valid_card_types is not None:
            params["valid_card_types"] = valid_card_types
        if parameters is not None:
            params["parameters"] = parameters
        if prompts is not None:
            params["prompts"] = prompts
        params.update(kwargs)
        action = PayAction(self, cid)
        return await self._start_action(action, "pay", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # Faxing
    # ------------------------------------------------------------------

    async def send_fax(
        self,
        document: str,
        *,
        identity: Optional[str] = None,
        header_info: Optional[str] = None,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> FaxAction:
        """Send a fax document. Returns a FaxAction."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {"control_id": cid, "document": document}
        if identity is not None:
            params["identity"] = identity
        if header_info is not None:
            params["header_info"] = header_info
        params.update(kwargs)
        action = FaxAction(self, cid, "send_fax")
        return await self._start_action(action, "send_fax", params, on_completed=on_completed)

    async def receive_fax(
        self,
        *,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> FaxAction:
        """Receive a fax. Returns a FaxAction."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {"control_id": cid}
        params.update(kwargs)
        action = FaxAction(self, cid, "receive_fax")
        return await self._start_action(action, "receive_fax", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # Tap (media interception)
    # ------------------------------------------------------------------

    async def tap(
        self,
        tap: dict[str, Any],
        device: dict[str, Any],
        *,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> TapAction:
        """Intercept call media and stream it. Returns a TapAction."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {
            "control_id": cid,
            "tap": tap,
            "device": device,
        }
        params.update(kwargs)
        action = TapAction(self, cid)
        return await self._start_action(action, "tap", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        url: str,
        *,
        name: Optional[str] = None,
        codec: Optional[str] = None,
        track: Optional[str] = None,
        status_url: Optional[str] = None,
        status_url_method: Optional[str] = None,
        authorization_bearer_token: Optional[str] = None,
        custom_parameters: Optional[dict[str, Any]] = None,
        control_id: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> StreamAction:
        """Start streaming call audio to a WebSocket endpoint. Returns a StreamAction."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {"control_id": cid, "url": url}
        if name is not None:
            params["name"] = name
        if codec is not None:
            params["codec"] = codec
        if track is not None:
            params["track"] = track
        if status_url is not None:
            params["status_url"] = status_url
        if status_url_method is not None:
            params["status_url_method"] = status_url_method
        if authorization_bearer_token is not None:
            params["authorization_bearer_token"] = authorization_bearer_token
        if custom_parameters is not None:
            params["custom_parameters"] = custom_parameters
        params.update(kwargs)
        action = StreamAction(self, cid)
        return await self._start_action(action, "stream", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    async def transfer(
        self,
        dest: str,
        **kwargs: Any,
    ) -> dict:
        """Transfer call control to another RELAY app or SWML script."""
        params: dict[str, Any] = {"dest": dest}
        params.update(kwargs)
        return await self._execute("transfer", params)

    # ------------------------------------------------------------------
    # Conference
    # ------------------------------------------------------------------

    async def join_conference(
        self,
        name: str,
        *,
        muted: Optional[bool] = None,
        beep: Optional[str] = None,
        start_on_enter: Optional[bool] = None,
        end_on_exit: Optional[bool] = None,
        wait_url: Optional[str] = None,
        max_participants: Optional[int] = None,
        record: Optional[str] = None,
        region: Optional[str] = None,
        trim: Optional[str] = None,
        coach: Optional[str] = None,
        status_callback: Optional[str] = None,
        status_callback_event: Optional[str] = None,
        status_callback_event_type: Optional[str] = None,
        status_callback_method: Optional[str] = None,
        recording_status_callback: Optional[str] = None,
        recording_status_callback_event: Optional[str] = None,
        recording_status_callback_event_type: Optional[str] = None,
        recording_status_callback_method: Optional[str] = None,
        stream_obj: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Join an ad-hoc audio conference."""
        params: dict[str, Any] = {"name": name}
        if muted is not None:
            params["muted"] = muted
        if beep is not None:
            params["beep"] = beep
        if start_on_enter is not None:
            params["start_on_enter"] = start_on_enter
        if end_on_exit is not None:
            params["end_on_exit"] = end_on_exit
        if wait_url is not None:
            params["wait_url"] = wait_url
        if max_participants is not None:
            params["max_participants"] = max_participants
        if record is not None:
            params["record"] = record
        if region is not None:
            params["region"] = region
        if trim is not None:
            params["trim"] = trim
        if coach is not None:
            params["coach"] = coach
        if status_callback is not None:
            params["status_callback"] = status_callback
        if status_callback_event is not None:
            params["status_callback_event"] = status_callback_event
        if status_callback_event_type is not None:
            params["status_callback_event_type"] = status_callback_event_type
        if status_callback_method is not None:
            params["status_callback_method"] = status_callback_method
        if recording_status_callback is not None:
            params["recording_status_callback"] = recording_status_callback
        if recording_status_callback_event is not None:
            params["recording_status_callback_event"] = recording_status_callback_event
        if recording_status_callback_event_type is not None:
            params["recording_status_callback_event_type"] = recording_status_callback_event_type
        if recording_status_callback_method is not None:
            params["recording_status_callback_method"] = recording_status_callback_method
        if stream_obj is not None:
            params["stream"] = stream_obj
        params.update(kwargs)
        return await self._execute("join_conference", params)

    async def leave_conference(self, conference_id: str, **kwargs: Any) -> dict:
        """Leave an audio conference."""
        params: dict[str, Any] = {"conference_id": conference_id}
        params.update(kwargs)
        return await self._execute("leave_conference", params)

    # ------------------------------------------------------------------
    # Hold / Unhold
    # ------------------------------------------------------------------

    async def hold(self) -> dict:
        """Put the call on hold."""
        return await self._execute("hold")

    async def unhold(self) -> dict:
        """Release the call from hold."""
        return await self._execute("unhold")

    # ------------------------------------------------------------------
    # Denoise
    # ------------------------------------------------------------------

    async def denoise(self) -> dict:
        """Start noise reduction on the call."""
        return await self._execute("denoise")

    async def denoise_stop(self) -> dict:
        """Stop noise reduction on the call."""
        return await self._execute("denoise.stop")

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        *,
        control_id: Optional[str] = None,
        status_url: Optional[str] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> TranscribeAction:
        """Start transcribing the call. Returns a TranscribeAction."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {"control_id": cid}
        if status_url is not None:
            params["status_url"] = status_url
        params.update(kwargs)
        action = TranscribeAction(self, cid)
        return await self._start_action(action, "transcribe", params, on_completed=on_completed)

    # ------------------------------------------------------------------
    # Echo
    # ------------------------------------------------------------------

    async def echo(
        self,
        *,
        timeout: Optional[float] = None,
        status_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Echo audio back to the caller (useful for testing)."""
        params: dict[str, Any] = {}
        if timeout is not None:
            params["timeout"] = timeout
        if status_url is not None:
            params["status_url"] = status_url
        params.update(kwargs)
        return await self._execute("echo", params or None)

    # ------------------------------------------------------------------
    # Digit bindings
    # ------------------------------------------------------------------

    async def bind_digit(
        self,
        digits: str,
        bind_method: str,
        *,
        bind_params: Optional[dict[str, Any]] = None,
        realm: Optional[str] = None,
        max_triggers: Optional[int] = None,
        **kwargs: Any,
    ) -> dict:
        """Bind a DTMF digit sequence to trigger a RELAY method."""
        params: dict[str, Any] = {
            "digits": digits,
            "bind_method": bind_method,
        }
        if bind_params is not None:
            params["params"] = bind_params
        if realm is not None:
            params["realm"] = realm
        if max_triggers is not None:
            params["max_triggers"] = max_triggers
        params.update(kwargs)
        return await self._execute("bind_digit", params)

    async def clear_digit_bindings(
        self,
        *,
        realm: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Clear all digit bindings, optionally filtered by realm."""
        params: dict[str, Any] = {}
        if realm is not None:
            params["realm"] = realm
        params.update(kwargs)
        return await self._execute("clear_digit_bindings", params or None)

    # ------------------------------------------------------------------
    # Live transcribe / translate
    # ------------------------------------------------------------------

    async def live_transcribe(
        self,
        action: dict[str, Any],
        **kwargs: Any,
    ) -> dict:
        """Start or stop live transcription on the call."""
        params: dict[str, Any] = {"action": action}
        params.update(kwargs)
        return await self._execute("live_transcribe", params)

    async def live_translate(
        self,
        action: dict[str, Any],
        *,
        status_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Start or stop live translation on the call."""
        params: dict[str, Any] = {"action": action}
        if status_url is not None:
            params["status_url"] = status_url
        params.update(kwargs)
        return await self._execute("live_translate", params)

    # ------------------------------------------------------------------
    # Room
    # ------------------------------------------------------------------

    async def join_room(
        self,
        name: str,
        *,
        status_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Join a video/audio room."""
        params: dict[str, Any] = {"name": name}
        if status_url is not None:
            params["status_url"] = status_url
        params.update(kwargs)
        return await self._execute("join_room", params)

    async def leave_room(self, **kwargs: Any) -> dict:
        """Leave the current room."""
        return await self._execute("leave_room", kwargs or None)

    # ------------------------------------------------------------------
    # AI Agent
    # ------------------------------------------------------------------

    async def ai(
        self,
        *,
        control_id: Optional[str] = None,
        agent: Optional[str] = None,
        prompt: Optional[dict[str, Any]] = None,
        post_prompt: Optional[dict[str, Any]] = None,
        post_prompt_url: Optional[str] = None,
        post_prompt_auth_user: Optional[str] = None,
        post_prompt_auth_password: Optional[str] = None,
        global_data: Optional[dict[str, Any]] = None,
        pronounce: Optional[list[dict[str, Any]]] = None,
        hints: Optional[list[str]] = None,
        languages: Optional[list[dict[str, Any]]] = None,
        SWAIG: Optional[dict[str, Any]] = None,
        ai_params: Optional[dict[str, Any]] = None,
        on_completed: Optional[Callable[[RelayEvent], Any]] = None,
        **kwargs: Any,
    ) -> AIAction:
        """Start an AI agent session on the call. Returns an AIAction."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {"control_id": cid}
        if agent is not None:
            params["agent"] = agent
        if prompt is not None:
            params["prompt"] = prompt
        if post_prompt is not None:
            params["post_prompt"] = post_prompt
        if post_prompt_url is not None:
            params["post_prompt_url"] = post_prompt_url
        if post_prompt_auth_user is not None:
            params["post_prompt_auth_user"] = post_prompt_auth_user
        if post_prompt_auth_password is not None:
            params["post_prompt_auth_password"] = post_prompt_auth_password
        if global_data is not None:
            params["global_data"] = global_data
        if pronounce is not None:
            params["pronounce"] = pronounce
        if hints is not None:
            params["hints"] = hints
        if languages is not None:
            params["languages"] = languages
        if SWAIG is not None:
            params["SWAIG"] = SWAIG
        if ai_params is not None:
            params["params"] = ai_params
        params.update(kwargs)
        action = AIAction(self, cid)
        return await self._start_action(action, "ai", params, on_completed=on_completed)

    async def amazon_bedrock(
        self,
        *,
        prompt: Optional[Any] = None,
        SWAIG: Optional[dict[str, Any]] = None,
        ai_params: Optional[dict[str, Any]] = None,
        global_data: Optional[dict[str, Any]] = None,
        post_prompt: Optional[dict[str, Any]] = None,
        post_prompt_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Connect to an Amazon Bedrock AI agent."""
        params: dict[str, Any] = {}
        if prompt is not None:
            params["prompt"] = prompt
        if SWAIG is not None:
            params["SWAIG"] = SWAIG
        if ai_params is not None:
            params["params"] = ai_params
        if global_data is not None:
            params["global_data"] = global_data
        if post_prompt is not None:
            params["post_prompt"] = post_prompt
        if post_prompt_url is not None:
            params["post_prompt_url"] = post_prompt_url
        params.update(kwargs)
        return await self._execute("amazon_bedrock", params)

    async def ai_message(
        self,
        *,
        message_text: Optional[str] = None,
        role: Optional[str] = None,
        reset: Optional[dict[str, Any]] = None,
        global_data: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Send a message to an active AI agent session."""
        params: dict[str, Any] = {}
        if message_text is not None:
            params["message_text"] = message_text
        if role is not None:
            params["role"] = role
        if reset is not None:
            params["reset"] = reset
        if global_data is not None:
            params["global_data"] = global_data
        params.update(kwargs)
        return await self._execute("ai_message", params)

    async def ai_hold(
        self,
        *,
        timeout: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Put an AI agent session on hold."""
        params: dict[str, Any] = {}
        if timeout is not None:
            params["timeout"] = timeout
        if prompt is not None:
            params["prompt"] = prompt
        params.update(kwargs)
        return await self._execute("ai_hold", params or None)

    async def ai_unhold(
        self,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Resume an AI agent session from hold."""
        params: dict[str, Any] = {}
        if prompt is not None:
            params["prompt"] = prompt
        params.update(kwargs)
        return await self._execute("ai_unhold", params or None)

    # ------------------------------------------------------------------
    # User events
    # ------------------------------------------------------------------

    async def user_event(
        self,
        *,
        event: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Send a custom user-defined event."""
        params: dict[str, Any] = {}
        if event is not None:
            params["event"] = event
        params.update(kwargs)
        return await self._execute("user_event", params)

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------

    async def queue_enter(
        self,
        queue_name: str,
        *,
        control_id: Optional[str] = None,
        status_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Place the call in a queue."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {
            "control_id": cid,
            "queue_name": queue_name,
        }
        if status_url is not None:
            params["status_url"] = status_url
        params.update(kwargs)
        return await self._execute("queue.enter", params)

    async def queue_leave(
        self,
        queue_name: str,
        *,
        control_id: Optional[str] = None,
        queue_id: Optional[str] = None,
        status_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Remove the call from a queue."""
        cid = control_id or str(uuid.uuid4())
        params: dict[str, Any] = {
            "control_id": cid,
            "queue_name": queue_name,
        }
        if queue_id is not None:
            params["queue_id"] = queue_id
        if status_url is not None:
            params["status_url"] = status_url
        params.update(kwargs)
        return await self._execute("queue.leave", params)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Call id={self.call_id!r} state={self.state!r} "
            f"direction={self.direction!r}>"
        )
