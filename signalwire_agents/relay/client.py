"""RelayClient — WebSocket + JSON-RPC 2.0 protocol + event dispatch.

One instance = one persistent WebSocket connection to SignalWire RELAY.

Architecture notes (mirrors the JS SDK):
- JSON-RPC requests are tracked by ``id`` in ``_pending``; responses resolve
  the corresponding Future.
- ``signalwire.event`` messages are acknowledged back to the server (event ACK)
  and then dispatched by ``event_type`` → Call object → Action object.
- Each Action registers with a ``control_id`` and listens for its own
  event_type (e.g. ``calling.call.play``).  Actions filter events by
  ``control_id`` so multiple concurrent actions on the same call work.
- Result code checking accepts any 2xx (matching the JS SDK regex /^2[0-9][0-9]$/).
  ``signalwire.connect`` responses skip code checking entirely.
- Execute has a configurable timeout (default 10s) to detect half-open connections.
- Requests made while disconnected are queued and flushed after re-auth.
- Server pings are tracked; if no ping arrives within the check interval the
  connection is assumed half-open and force-closed for reconnect.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Any, Callable, Coroutine, Optional

import websockets
import websockets.exceptions

from signalwire_agents.core.logging_config import get_logger

from .call import Call
from .constants import (
    AGENT_STRING,
    PROTOCOL_VERSION,
    DEFAULT_RELAY_HOST,
    EVENT_AUTHORIZATION_STATE,
    EVENT_CALL_DIAL,
    EVENT_CALL_RECEIVE,
    EVENT_CALL_STATE,
    EVENT_MESSAGING_RECEIVE,
    EVENT_MESSAGING_STATE,
    METHOD_SIGNALWIRE_CONNECT,
    METHOD_SIGNALWIRE_DISCONNECT,
    METHOD_SIGNALWIRE_EVENT,
    METHOD_SIGNALWIRE_PING,
    METHOD_SIGNALWIRE_RECEIVE,
    METHOD_SIGNALWIRE_UNRECEIVE,
    RECONNECT_BACKOFF_FACTOR,
    RECONNECT_MAX_DELAY,
    RECONNECT_MIN_DELAY,
)
from .message import Message
logger = get_logger("relay_client")

CallHandler = Callable[["Call"], Coroutine[Any, Any, None]]
MessageHandler = Callable[["Message"], Coroutine[Any, Any, None]]

# Any 2xx code is considered success (matches JS SDK: /^2[0-9][0-9]$/)
_SUCCESS_CODE_RE = re.compile(r"^2\d{2}$")

# Default timeout for execute() requests (seconds)
_EXECUTE_TIMEOUT = 10.0

# If no server ping arrives within this interval, send a client ping as probe
_CHECK_PING_DELAY = 15.0

# Client-side ping interval (sends our own pings between server pings)
_CLIENT_PING_INTERVAL = 30.0

# Max consecutive ping failures before force-closing
_MAX_PING_FAILURES = 3

# Safety limits to prevent unbounded memory growth from malicious/buggy servers
_DEFAULT_MAX_ACTIVE_CALLS = 1000
_MAX_QUEUE_SIZE = 500

# Max concurrent RelayClient connections per process (env: RELAY_MAX_CONNECTIONS)
try:
    _MAX_CONNECTIONS = max(1, int(os.environ.get("RELAY_MAX_CONNECTIONS", "1")))
except ValueError:
    _MAX_CONNECTIONS = 1

# Process-wide tracking of active RelayClient connections
_active_clients: set[int] = set()


class RelayClient:
    """Manages a WebSocket connection to SignalWire RELAY.

    Usage::

        client = RelayClient(project="...", token="...", contexts=["default"])

        @client.on_call
        async def handle(call):
            await call.answer()
            await call.hangup()

        client.run()
    """

    def __init__(
        self,
        project: Optional[str] = None,
        token: Optional[str] = None,
        jwt_token: Optional[str] = None,
        host: Optional[str] = None,
        contexts: Optional[list[str]] = None,
        max_active_calls: Optional[int] = None,
    ):
        self.project = project or os.environ.get("SIGNALWIRE_PROJECT_ID", "")
        self.token = token or os.environ.get("SIGNALWIRE_API_TOKEN", "")
        self.jwt_token = jwt_token or os.environ.get("SIGNALWIRE_JWT_TOKEN", "")
        self.host = host or os.environ.get("SIGNALWIRE_SPACE", DEFAULT_RELAY_HOST)
        self.contexts = contexts or []

        if self.jwt_token:
            # JWT auth — project/token not required (project_id is inside the token)
            if not self.project:
                self.project = ""
        elif not self.project or not self.token:
            raise ValueError(
                "project and token are required (or provide jwt_token). "
                "Pass them directly or set SIGNALWIRE_PROJECT_ID / "
                "SIGNALWIRE_API_TOKEN / SIGNALWIRE_JWT_TOKEN env vars."
            )

        # Validate host to prevent SSRF / header injection
        if any(c in self.host for c in ('@', '/', '?', '\r', '\n', ' ')):
            raise ValueError(
                f"Invalid host: {self.host!r}. Must be a hostname, not a URL."
            )

        # Max concurrent calls (constructor > env var > default)
        if max_active_calls is not None:
            self._max_active_calls = max(1, max_active_calls)
        else:
            try:
                env_val = os.environ.get("RELAY_MAX_ACTIVE_CALLS", "")
                self._max_active_calls = max(1, int(env_val)) if env_val else _DEFAULT_MAX_ACTIVE_CALLS
            except ValueError:
                self._max_active_calls = _DEFAULT_MAX_ACTIVE_CALLS

        # Internal state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._pending: dict[str, asyncio.Future[dict]] = {}
        # Track the method for each pending request (for code-checking decisions)
        self._pending_methods: dict[str, str] = {}
        self._calls: dict[str, Call] = {}
        self._on_call_handler: Optional[CallHandler] = None
        self._on_message_handler: Optional[MessageHandler] = None
        self._messages: dict[str, Message] = {}  # message_id → Message
        # Pending outbound dials: tag → Future[Call] resolved when dial completes
        self._pending_dials: dict[str, asyncio.Future["Call"]] = {}
        # Calls created during dial, indexed by tag for event routing before
        # the winning call_id is known
        self._dial_calls_by_tag: dict[str, list["Call"]] = {}
        self._connected = False
        self._closing = False
        self._reconnect_delay = RECONNECT_MIN_DELAY
        self._recv_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        # Server-assigned protocol string from signalwire.connect response.
        # Sent back on reconnect to resume the same session.
        self._relay_protocol: str = ""
        self._identity: str = ""
        # Encrypted authorization state from signalwire.authorization.state
        # events. Sent back on reconnect for fast re-auth without full
        # authentication round-trip.
        self._authorization_state: str = ""
        # Half-open detection: reset on every server ping; fires if no ping
        # arrives within _CHECK_PING_DELAY seconds.
        self._check_ping_handle: Optional[asyncio.TimerHandle] = None
        # Track consecutive client ping failures for exponential backoff
        self._ping_failures: int = 0
        # Queue for requests made while disconnected/reconnecting
        self._execute_queue: list[tuple[dict, asyncio.Future[dict]]] = []

    def __del__(self) -> None:
        _active_clients.discard(id(self))

    async def __aenter__(self) -> "RelayClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect()

    @property
    def relay_protocol(self) -> str:
        """Server-assigned protocol string from the connect response."""
        return self._relay_protocol

    # ------------------------------------------------------------------
    # Decorator for inbound call handler
    # ------------------------------------------------------------------

    def on_call(self, handler: CallHandler) -> CallHandler:
        """Register the inbound call handler (decorator)."""
        self._on_call_handler = handler
        return handler

    def on_message(self, handler: MessageHandler) -> MessageHandler:
        """Register the inbound message handler (decorator)."""
        self._on_message_handler = handler
        return handler

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to RELAY and authenticate."""
        # Guard against connection leaks — enforce per-process limit
        # (don't count ourselves if we're already tracked, i.e. reconnecting)
        other_count = len(_active_clients - {id(self)})
        if other_count >= _MAX_CONNECTIONS:
            raise RuntimeError(
                f"RelayClient connection limit reached ({_MAX_CONNECTIONS}). "
                f"There are already {other_count} active connection(s) in this process. "
                f"Call disconnect() on existing clients first, or set "
                f"RELAY_MAX_CONNECTIONS env var to allow more."
            )
        _active_clients.add(id(self))

        uri = f"wss://{self.host}"
        logger.info(f"Connecting to {uri}")

        self._ws = await websockets.connect(uri, ping_interval=None, max_size=10 * 1024 * 1024)
        self._connected = True
        self._reconnect_delay = RECONNECT_MIN_DELAY

        # Start receive loop BEFORE authenticate so responses can be read
        self._recv_task = asyncio.ensure_future(self._recv_loop())

        # Authenticate
        await self._authenticate()

        # Start client-side ping loop for resilience
        self._ping_failures = 0
        self._ping_task = asyncio.ensure_future(self._ping_loop())

        # Flush any requests that were queued during reconnect
        self._flush_execute_queue()

        logger.info("Connected and authenticated to RELAY")

    async def _authenticate(self) -> None:
        """Send signalwire.connect and wait for the response.

        The server returns a ``protocol`` string that must be used as
        the ``protocol`` field in all subsequent calling commands.
        On reconnect the previously-received protocol is sent back to
        resume the same session.
        """
        if self.jwt_token:
            authentication = {"jwt_token": self.jwt_token}
        else:
            authentication = {"project": self.project, "token": self.token}
        params: dict[str, Any] = {
            "version": PROTOCOL_VERSION,
            "agent": AGENT_STRING,
            "event_acks": True,
            "authentication": authentication,
        }
        if self.contexts:
            params["contexts"] = self.contexts
        # Re-send the protocol on reconnect to resume the session
        if self._relay_protocol:
            params["protocol"] = self._relay_protocol
        # Send authorization_state for fast reconnection if available
        if self._authorization_state:
            params["authorization_state"] = self._authorization_state

        result = await self._send_request(METHOD_SIGNALWIRE_CONNECT, params)

        # Capture server-assigned protocol and identity
        self._relay_protocol = result.get("protocol", self._relay_protocol)
        self._identity = result.get("identity", self._identity)
        logger.debug(f"Auth response: protocol={self._relay_protocol} identity={self._identity}")

    async def disconnect(self) -> None:
        """Cleanly close the connection."""
        self._closing = True
        self._connected = False
        _active_clients.discard(id(self))

        self._cancel_check_ping()

        if self._ping_task:
            self._ping_task.cancel()
            self._ping_task = None

        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        # Cancel any pending request futures
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

        # Cancel any queued requests
        for _, fut in self._execute_queue:
            if not fut.done():
                fut.cancel()
        self._execute_queue.clear()

        # Cancel any pending dials
        for fut in self._pending_dials.values():
            if not fut.done():
                fut.cancel()
        self._pending_dials.clear()
        self._dial_calls_by_tag.clear()

        logger.info("Disconnected from RELAY")

    # ------------------------------------------------------------------
    # Public RPC interface
    # ------------------------------------------------------------------

    async def execute(self, method: str, params: dict[str, Any]) -> dict:
        """Send a JSON-RPC request and await the response.

        For calling methods, ``method`` is the full name (e.g.
        ``"calling.answer"``, ``"calling.play"``) with ``node_id``
        and ``call_id`` in ``params``.

        If the connection is not ready, the request is queued and sent
        after re-authentication completes.
        """
        return await self._send_request(method, params)

    async def dial(
        self,
        devices: list[list[dict[str, Any]]],
        *,
        tag: Optional[str] = None,
        max_duration: Optional[int] = None,
        dial_timeout: Optional[float] = None,
    ) -> Call:
        """Initiate an outbound call using dial. Returns a Call object.

        The ``calling.dial`` RPC response only contains
        ``{"code": "200", "message": "Dialing"}`` — no call_id. The real
        call_id and node_id arrive via ``calling.call.dial`` events
        matched by ``tag``.  This method waits for that event so the
        returned Call always has valid identifiers.

        Args:
            devices: Array of device lists (serial/parallel dial).
            tag: Client-provided tag for event correlation. Auto-generated
                 if not supplied.
            max_duration: Optional max call duration in minutes.
            dial_timeout: How long (seconds) to wait for the dial to
                          complete before raising TimeoutError.  Defaults
                          to 120s.
        """
        dial_tag = tag or str(uuid.uuid4())
        params: dict[str, Any] = {
            "tag": dial_tag,
            "devices": devices,
        }
        if max_duration:
            params["max_duration"] = max_duration

        # Register a Future that _handle_event will resolve when the
        # calling.call.dial event arrives with matching tag.
        loop = asyncio.get_running_loop()
        dial_future: asyncio.Future[Call] = loop.create_future()
        self._pending_dials[dial_tag] = dial_future
        self._dial_calls_by_tag[dial_tag] = []

        try:
            # Send the dial RPC — response is just {"code": "200", "message": "Dialing"}
            await self.execute("calling.dial", params)

            # Wait for the calling.call.dial event to resolve the call
            timeout = dial_timeout if dial_timeout is not None else 120.0
            call = await asyncio.wait_for(dial_future, timeout=timeout)
            return call
        except asyncio.TimeoutError:
            raise RelayError(-1, f"Dial timed out waiting for answer (tag={dial_tag})")
        except Exception:
            # Cancel the future if it hasn't resolved yet
            if not dial_future.done():
                dial_future.cancel()
            raise
        finally:
            self._pending_dials.pop(dial_tag, None)
            self._dial_calls_by_tag.pop(dial_tag, None)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        *,
        to_number: str,
        from_number: str,
        context: Optional[str] = None,
        body: Optional[str] = None,
        media: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        region: Optional[str] = None,
        on_completed: Optional[Callable] = None,
    ) -> Message:
        """Send an outbound SMS/MMS message.

        At least one of ``body`` or ``media`` must be provided.

        Returns a Message object that tracks state changes. Use
        ``await message.wait()`` to block until delivery confirmation
        (or failure).

        Args:
            to_number: Destination phone number in E.164 format.
            from_number: Sender phone number in E.164 format.
            context: Context for receiving state events. Defaults to
                     the relay protocol.
            body: Text body of the message.
            media: List of media URLs for MMS.
            tags: Optional tags for the message.
            region: Optional origination region.
            on_completed: Optional callback fired when a terminal state
                          (delivered/undelivered/failed) is reached.
        """
        if not body and not media:
            raise ValueError("At least one of body or media is required")

        msg_context = context or self._relay_protocol or "default"
        params: dict[str, Any] = {
            "context": msg_context,
            "to_number": to_number,
            "from_number": from_number,
        }
        if body:
            params["body"] = body
        if media:
            params["media"] = media
        if tags:
            params["tags"] = tags
        if region:
            params["region"] = region

        result = await self.execute("messaging.send", params)

        message_id = result.get("message_id", "")
        message = Message(
            message_id=message_id,
            context=msg_context,
            direction="outbound",
            from_number=from_number,
            to_number=to_number,
            body=body or "",
            media=media or [],
            tags=tags or [],
            state="queued",
        )
        if on_completed:
            message._on_completed = on_completed
        if message_id:
            self._messages[message_id] = message

        return message

    # ------------------------------------------------------------------
    # Dynamic context subscription
    # ------------------------------------------------------------------

    async def receive(self, contexts: list[str]) -> None:
        """Subscribe to additional contexts for inbound events.

        Sends ``signalwire.receive`` on the assigned protocol to start
        receiving inbound calls on the given contexts.  Can be called
        after ``connect()`` to dynamically add contexts without reconnecting.
        """
        if not contexts:
            return
        await self._send_request(METHOD_SIGNALWIRE_RECEIVE, {
            "contexts": contexts,
        })
        logger.info(f"Subscribed to contexts: {contexts}")

    async def unreceive(self, contexts: list[str]) -> None:
        """Unsubscribe from contexts for inbound events.

        Sends ``signalwire.unreceive`` to stop receiving inbound calls
        on the given contexts.
        """
        if not contexts:
            return
        await self._send_request(METHOD_SIGNALWIRE_UNRECEIVE, {
            "contexts": contexts,
        })
        logger.info(f"Unsubscribed from contexts: {contexts}")

    # ------------------------------------------------------------------
    # Run / event loop entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking entry point — runs the event loop until interrupted."""
        try:
            asyncio.run(self._run_forever())
        except KeyboardInterrupt:
            pass
        logger.info("RELAY client stopped")

    async def _run_forever(self) -> None:
        """Connect and maintain the connection with auto-reconnect."""
        # Register SIGINT handler so Ctrl+C triggers a clean shutdown
        # instead of dumping a stack trace.
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            __import__("signal").SIGINT,
            lambda: asyncio.ensure_future(self._shutdown()),
        )

        while not self._closing:
            try:
                await self.connect()
                # Block until the recv loop exits (connection lost)
                if self._recv_task:
                    await self._recv_task
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Connection error")

            if self._closing:
                break

            self._cancel_check_ping()

            # Reject all pending requests — connection is lost
            self._clear_pending_requests()

            # Auto-reconnect with exponential backoff
            logger.info(f"Reconnecting in {self._reconnect_delay:.1f}s ...")
            try:
                await asyncio.sleep(self._reconnect_delay)
            except asyncio.CancelledError:
                break
            self._reconnect_delay = min(
                self._reconnect_delay * RECONNECT_BACKOFF_FACTOR,
                RECONNECT_MAX_DELAY,
            )

        await self.disconnect()

    async def _shutdown(self) -> None:
        """Signal handler for clean Ctrl+C shutdown."""
        logger.info("Shutting down ...")
        self._closing = True
        # Cancel the recv task to unblock _run_forever
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()

    # ------------------------------------------------------------------
    # Internal: send / receive
    # ------------------------------------------------------------------

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict:
        """Send a JSON-RPC 2.0 request and return the result.

        If not connected, queues the request for later delivery.
        Applies a timeout to detect half-open connections.
        """
        req_id = str(uuid.uuid4())
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending[req_id] = future
        self._pending_methods[req_id] = method

        try:
            if not self._ws or not self._connected:
                # Queue for delivery after reconnect (like JS executeQueue)
                if len(self._execute_queue) >= _MAX_QUEUE_SIZE:
                    raise RelayError(-1, "Execute queue full — too many requests while disconnected")
                self._execute_queue.append((message, future))
                logger.debug(f"Request queued (not connected): {method}")
            else:
                raw = json.dumps(message)
                logger.debug(f">> {method} id={req_id}")
                await self._ws.send(raw)

            return await asyncio.wait_for(future, timeout=_EXECUTE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {method} {req_id}")
            # Timeout may indicate a half-open connection — force reconnect
            if method != METHOD_SIGNALWIRE_CONNECT:
                self._force_close()
            raise RelayError(-1, f"Request timeout for {method}")
        finally:
            self._pending.pop(req_id, None)
            self._pending_methods.pop(req_id, None)

    def _flush_execute_queue(self) -> None:
        """Send any requests that were queued while disconnected."""
        if not self._execute_queue or not self._ws:
            return
        queued = list(self._execute_queue)
        self._execute_queue.clear()
        logger.debug(f"Flushing {len(queued)} queued requests")
        for message, future in queued:
            if future.done():
                continue
            raw = json.dumps(message)
            logger.debug(f">> (queued) {message.get('method', '?')} id={message.get('id', '?')}")
            asyncio.ensure_future(self._safe_send(raw, future))

    async def _safe_send(self, raw: str, future: asyncio.Future) -> None:
        """Send a queued message, rejecting its future on failure."""
        try:
            if self._ws:
                await self._ws.send(raw)
        except Exception as exc:
            if not future.done():
                future.set_exception(
                    RelayError(-1, f"Failed to send queued request: {exc}")
                )

    def _clear_pending_requests(self) -> None:
        """Reject all pending request futures (connection lost)."""
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(
                    RelayError(-1, "Connection closed")
                )
        self._pending.clear()
        self._pending_methods.clear()
        # Also reject pending dials
        for fut in self._pending_dials.values():
            if not fut.done():
                fut.set_exception(
                    RelayError(-1, "Connection closed during dial")
                )
        self._pending_dials.clear()
        self._dial_calls_by_tag.clear()

    async def _recv_loop(self) -> None:
        """Read messages from the WebSocket until closed."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {raw}")
                    continue

                logger.debug(f"<< {raw}")
                await self._handle_message(msg)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in recv loop")
        finally:
            self._connected = False
            self._cancel_check_ping()
            if self._ping_task:
                self._ping_task.cancel()
                self._ping_task = None

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        """Route an incoming JSON-RPC message."""
        if not isinstance(msg, dict):
            logger.warning("Ignoring non-dict message")
            return

        # JSON-RPC level error (e.g. method not found)
        if "id" in msg and "error" in msg:
            req_id = msg["id"]
            future = self._pending.get(req_id)
            if future and not future.done():
                error = msg["error"]
                if not isinstance(error, dict):
                    error = {"code": -1, "message": str(error)}
                future.set_exception(
                    RelayError(error.get("code", -1), str(error.get("message", "Unknown error")))
                )
            else:
                logger.debug(f"Error response for unknown/expired request {req_id}")
            return

        # Response to a pending request
        if "id" in msg and "result" in msg:
            req_id = msg["id"]
            future = self._pending.get(req_id)
            if future and not future.done():
                result = msg["result"]
                if not isinstance(result, dict):
                    future.set_result({"raw": result})
                    return
                # Look up the original request method to decide code-checking
                request_method = self._get_request_method(req_id)
                if request_method == METHOD_SIGNALWIRE_CONNECT:
                    # signalwire.connect: return raw result, no code checking
                    future.set_result(result)
                else:
                    # Calling API: result contains code/message; any 2xx is success
                    code = result.get("code")
                    code_str = str(code) if code is not None else None
                    if code_str is not None and not _SUCCESS_CODE_RE.match(code_str):
                        try:
                            int_code = int(code)
                        except (ValueError, TypeError):
                            int_code = -1
                        future.set_exception(
                            RelayError(int_code, str(result.get("message", "Unknown error")))
                        )
                    else:
                        future.set_result(result)
            else:
                logger.debug(f"Response for unknown/expired request {req_id}")
            return

        # Server-initiated method call (event or ping)
        method = msg.get("method", "")
        params = msg.get("params", {})
        if not isinstance(params, dict):
            logger.warning("Ignoring message with non-dict params")
            return

        if method == METHOD_SIGNALWIRE_EVENT:
            # ACK the event back to the server (like JS _eventAcknowledgingHandler)
            if "id" in msg:
                await self._send_event_ack(msg["id"])
            await self._handle_event(params)
        elif method == METHOD_SIGNALWIRE_PING:
            # Reset the half-open detection timer
            self._reset_check_ping()
            if "id" in msg:
                await self._send_pong(msg["id"])
        elif method == METHOD_SIGNALWIRE_DISCONNECT:
            # Server is telling us to disconnect (e.g. during deployment).
            # Respond with empty result, then let the server close the socket.
            restart = params.get("restart", False)
            logger.info(f"Received signalwire.disconnect from server (restart={restart})")
            if "id" in msg:
                await self._send_event_ack(msg["id"])
            if restart:
                # Server says start fresh — clear protocol and auth state
                # so the next connect is a clean connection, not a hijack
                self._relay_protocol = ""
                self._authorization_state = ""
            # Don't set _closing — we want to reconnect after the server closes

    def _get_request_method(self, req_id: str) -> str:
        """Look up the original method for a pending request by ID.

        Used to skip code-checking for signalwire.connect responses.
        """
        return self._pending_methods.get(req_id, "")

    async def _handle_event(self, payload: dict[str, Any]) -> None:
        """Dispatch a signalwire.event to the appropriate Call.

        Routing logic:
        1. ``calling.call.receive`` → create inbound Call
        2. ``calling.call.dial`` → resolve pending dial by tag
        3. ``calling.call.state`` with tag matching a pending dial →
           create/register the Call leg so events flow before dial completes
        4. All other events → route by ``call_id`` in ``params``
        """
        event_type = payload.get("event_type", "")
        event_params = payload.get("params", {})
        call_id = event_params.get("call_id", "")

        if not event_type:
            logger.warning(f"Received event with empty event_type: {payload}")
            return

        logger.debug(f"Event: {event_type} call_id={call_id}")

        # Authorization state update — store for reconnection
        if event_type == EVENT_AUTHORIZATION_STATE:
            auth_state = event_params.get("authorization_state", "")
            if auth_state:
                self._authorization_state = auth_state
                logger.debug("Updated authorization_state for reconnection")
            return

        # Inbound message
        if event_type == EVENT_MESSAGING_RECEIVE:
            await self._handle_inbound_message(payload)
            return

        # Outbound message state change
        if event_type == EVENT_MESSAGING_STATE:
            await self._handle_message_state(payload)
            return

        # Inbound call
        if event_type == EVENT_CALL_RECEIVE:
            await self._handle_inbound_call(payload)
            return

        # Outbound dial progress — call_id is nested at params.call.call_id
        if event_type == EVENT_CALL_DIAL:
            await self._handle_dial_event(payload)
            return

        # State events during pending dial — the Call hasn't been returned
        # to the user yet but we need to create and register it so
        # subsequent events (play, record, etc.) can route to it.
        if event_type == EVENT_CALL_STATE:
            tag = event_params.get("tag", "")
            if tag and tag in self._pending_dials and call_id:
                if call_id not in self._calls:
                    self._register_dial_leg(tag, event_params)
                # Fall through to normal routing below so the Call gets
                # the state update dispatched to it.

        # Route to existing Call by call_id
        call = self._calls.get(call_id) if call_id else None
        if call:
            await call._dispatch_event(payload)

            # Clean up ended calls
            if call.state == "ended":
                self._calls.pop(call_id, None)

    async def _handle_inbound_call(self, payload: dict[str, Any]) -> None:
        """Create a Call object for an inbound call and invoke the handler."""
        if len(self._calls) >= self._max_active_calls:
            logger.error(f"Max active calls ({self._max_active_calls}) reached, dropping inbound call")
            return
        params = payload.get("params", {})
        call_id = params.get("call_id", "")

        call = Call(
            client=self,
            call_id=call_id,
            node_id=params.get("node_id", ""),
            project_id=params.get("project_id", self.project),
            context=self._relay_protocol or params.get("context", params.get("protocol", "")),
            tag=params.get("tag", ""),
            direction=params.get("direction", "inbound"),
            device=params.get("device"),
            state=params.get("call_state", "created"),
            segment_id=params.get("segment_id", ""),
        )
        self._calls[call_id] = call

        if self._on_call_handler:
            asyncio.ensure_future(self._safe_call_handler(call))
        else:
            logger.warning(f"Inbound call {call_id} but no on_call handler registered")

    async def _safe_call_handler(self, call: Call) -> None:
        """Run the on_call handler as a free task so the recv loop is never blocked."""
        try:
            await self._on_call_handler(call)
        except Exception:
            logger.exception(f"Error in on_call handler for {call.call_id}")

    async def _handle_inbound_message(self, payload: dict[str, Any]) -> None:
        """Create a Message object for an inbound message and invoke the handler."""
        params = payload.get("params", {})
        message = Message(
            message_id=params.get("message_id", ""),
            context=params.get("context", ""),
            direction=params.get("direction", "inbound"),
            from_number=params.get("from_number", ""),
            to_number=params.get("to_number", ""),
            body=params.get("body", ""),
            media=params.get("media", []),
            segments=params.get("segments", 0),
            state=params.get("message_state", "received"),
            tags=params.get("tags", []),
        )

        if self._on_message_handler:
            asyncio.ensure_future(self._safe_message_handler(message))
        else:
            logger.warning(f"Inbound message {message.message_id} but no on_message handler registered")

    async def _safe_message_handler(self, message: Message) -> None:
        """Run the on_message handler as a free task so the recv loop is never blocked."""
        try:
            await self._on_message_handler(message)
        except Exception:
            logger.exception(f"Error in on_message handler for {message.message_id}")

    async def _handle_message_state(self, payload: dict[str, Any]) -> None:
        """Route a messaging.state event to the tracked Message object."""
        params = payload.get("params", {})
        message_id = params.get("message_id", "")
        message = self._messages.get(message_id)
        if message:
            await message._dispatch_event(payload)
            # Clean up terminal messages
            if message.is_done:
                self._messages.pop(message_id, None)
        else:
            logger.debug(f"State event for unknown message {message_id}")

    def _register_dial_leg(self, tag: str, event_params: dict[str, Any]) -> Call:
        """Create a Call for a call leg spawned by calling.dial.

        Called when a ``calling.call.state`` event arrives with a tag
        matching a pending dial but the call_id isn't registered yet.
        The Call is registered in ``_calls`` so subsequent events route
        normally, and also tracked in ``_dial_calls_by_tag`` so
        ``_handle_dial_event`` can find the winner.
        """
        call_id = event_params.get("call_id", "")
        node_id = event_params.get("node_id", "")
        call = Call(
            client=self,
            call_id=call_id,
            node_id=node_id,
            project_id=event_params.get("project_id", self.project),
            context=self._relay_protocol,
            tag=tag,
            direction="outbound",
            device=event_params.get("device"),
            state=event_params.get("call_state", "created"),
        )
        self._calls[call_id] = call
        self._dial_calls_by_tag.setdefault(tag, []).append(call)
        logger.debug(f"Registered dial leg: call_id={call_id} tag={tag}")
        return call

    async def _handle_dial_event(self, payload: dict[str, Any]) -> None:
        """Handle a ``calling.call.dial`` event.

        The event structure (per spec):
        - ``params.tag`` — the dial tag
        - ``params.dial_state`` — ``dialing`` | ``answered`` | ``failed``
        - ``params.call`` — object with ``call_id``, ``node_id``,
          ``tag``, ``device``, ``dial_winner`` (only on answered)

        Resolves the pending dial Future with the winning Call when
        ``dial_state`` is ``"answered"``, or rejects it on ``"failed"``.
        """
        event_params = payload.get("params", {})
        tag = event_params.get("tag", "")
        dial_state = event_params.get("dial_state", "")
        call_info = event_params.get("call", {})

        logger.debug(f"Dial event: tag={tag} state={dial_state}")

        dial_future = self._pending_dials.get(tag)
        if not dial_future or dial_future.done():
            # No pending dial for this tag — may be a stale event
            # or a "dialing" progress event after we already resolved.
            # Still dispatch to any existing call for event listeners.
            if call_info:
                cid = call_info.get("call_id", "")
                call = self._calls.get(cid)
                if call:
                    await call._dispatch_event(payload)
            return

        if dial_state == "answered":
            # Find or create the winning Call
            winner_call_id = call_info.get("call_id", "")
            winner_node_id = call_info.get("node_id", "")
            call = self._calls.get(winner_call_id)
            if not call:
                # Call wasn't registered via state events — create it now
                call = Call(
                    client=self,
                    call_id=winner_call_id,
                    node_id=winner_node_id,
                    project_id=self.project,
                    context=self._relay_protocol,
                    tag=tag,
                    direction="outbound",
                    device=call_info.get("device"),
                    state="answered",
                )
                self._calls[winner_call_id] = call
            else:
                # Update node_id in case it wasn't set from state events
                if winner_node_id and not call.node_id:
                    call.node_id = winner_node_id
            dial_future.set_result(call)

        elif dial_state == "failed":
            dial_future.set_exception(
                RelayError(-1, f"Dial failed (tag={tag})")
            )

        # "dialing" events are progress — don't resolve the future yet

    async def _send_pong(self, req_id: str) -> None:
        """Respond to a server ping."""
        if not self._ws:
            return
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        })
        await self._ws.send(msg)

    async def _send_event_ack(self, req_id: str) -> None:
        """Acknowledge a signalwire.event back to the server."""
        if not self._ws:
            return
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        })
        await self._ws.send(msg)

    # ------------------------------------------------------------------
    # Client-side ping loop + half-open detection
    # ------------------------------------------------------------------

    async def _ping_loop(self) -> None:
        """Periodically send client pings for resilience.

        - Sends signalwire.ping every _CLIENT_PING_INTERVAL seconds.
        - On failure, retries with exponential backoff up to _MAX_PING_FAILURES.
        - After max failures, force-closes the connection for reconnect.
        - Server pings also reset the failure counter (see _handle_message).
        """
        try:
            while self._connected and self._ws:
                await asyncio.sleep(_CLIENT_PING_INTERVAL)
                if not self._connected or not self._ws:
                    break
                try:
                    await self._send_request(METHOD_SIGNALWIRE_PING, {})
                    # Success — reset failure counter
                    self._ping_failures = 0
                except Exception:
                    self._ping_failures += 1
                    backoff = min(
                        RECONNECT_MIN_DELAY * (RECONNECT_BACKOFF_FACTOR ** self._ping_failures),
                        RECONNECT_MAX_DELAY,
                    )
                    logger.warning(
                        f"Client ping failed ({self._ping_failures}/{_MAX_PING_FAILURES}), backoff {backoff:.1f}s"
                    )
                    if self._ping_failures >= _MAX_PING_FAILURES:
                        logger.error("Max ping failures reached, forcing reconnect")
                        self._force_close()
                        break
                    # Wait the backoff period before retrying
                    await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            pass

    def _reset_check_ping(self) -> None:
        """Reset the server-ping watchdog timer.

        Called when we receive a server ping — resets failure counter and
        restarts the watchdog that will probe if the server goes silent.
        """
        self._ping_failures = 0
        self._cancel_check_ping()
        loop = asyncio.get_running_loop()
        self._check_ping_handle = loop.call_later(
            _CHECK_PING_DELAY, self._on_check_ping_timeout
        )

    def _cancel_check_ping(self) -> None:
        if self._check_ping_handle:
            self._check_ping_handle.cancel()
            self._check_ping_handle = None

    def _on_check_ping_timeout(self) -> None:
        """No server ping received in time — send a client ping as probe.

        The client ping loop will handle failures with exponential backoff.
        This just logs; the ping loop does the actual probing.
        """
        logger.debug(f"No server ping received in {_CHECK_PING_DELAY:.1f}s, client pings will probe")

    def _force_close(self) -> None:
        """Force-close the WebSocket to trigger reconnect."""
        self._cancel_check_ping()
        self._connected = False
        if self._ping_task:
            self._ping_task.cancel()
            self._ping_task = None
        if self._ws:
            asyncio.ensure_future(self._ws.close())


class RelayError(Exception):
    """Error returned by the RELAY server."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"RELAY error {code}: {message}")
