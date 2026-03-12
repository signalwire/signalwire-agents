"""Message object — represents an SMS/MMS message in the RELAY messaging namespace.

A Message tracks the lifecycle of a sent or received message via state events.
Outbound messages progress through: queued → initiated → sent → delivered (or
undelivered/failed).  Inbound messages arrive fully formed with state "received".
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from signalwire_agents.core.logging_config import get_logger

from .constants import MESSAGE_TERMINAL_STATES
from .event import RelayEvent

logger = get_logger("relay_message")


class Message:
    """Represents a single SMS/MMS message.

    For outbound messages, use ``await message.wait()`` to block until a
    terminal state (delivered, undelivered, failed) is reached.
    """

    def __init__(
        self,
        *,
        message_id: str = "",
        context: str = "",
        direction: str = "",
        from_number: str = "",
        to_number: str = "",
        body: str = "",
        media: Optional[list[str]] = None,
        segments: int = 0,
        state: str = "",
        reason: str = "",
        tags: Optional[list[str]] = None,
    ):
        self.message_id = message_id
        self.context = context
        self.direction = direction
        self.from_number = from_number
        self.to_number = to_number
        self.body = body
        self.media: list[str] = media or []
        self.segments = segments
        self.state = state
        self.reason = reason
        self.tags: list[str] = tags or []

        # Completion tracking
        self._done: asyncio.Future[RelayEvent] = asyncio.get_event_loop().create_future()
        self._on_completed: Optional[Callable[[RelayEvent], Any]] = None
        self._listeners: list[Callable] = []

    @property
    def is_done(self) -> bool:
        """True if the message has reached a terminal state."""
        return self._done.done()

    @property
    def result(self) -> Optional[RelayEvent]:
        """The terminal RelayEvent, or None if not yet done."""
        if self._done.done():
            return self._done.result()
        return None

    def on(self, handler: Callable) -> None:
        """Register an event listener for state changes on this message."""
        self._listeners.append(handler)

    async def wait(self, timeout: Optional[float] = None) -> RelayEvent:
        """Block until the message reaches a terminal state.

        Returns the terminal RelayEvent. Raises asyncio.TimeoutError if
        timeout is specified and exceeded.
        """
        if timeout is not None:
            return await asyncio.wait_for(asyncio.shield(self._done), timeout=timeout)
        return await self._done

    async def _dispatch_event(self, payload: dict[str, Any]) -> None:
        """Handle a messaging.state event for this message."""
        event_params = payload.get("params", {})
        new_state = event_params.get("message_state", "")

        if new_state:
            self.state = new_state
        self.reason = event_params.get("reason", self.reason)

        event = RelayEvent.from_payload(payload)

        # Notify listeners
        for handler in self._listeners:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(f"Error in message event handler for {self.message_id}")

        # Check terminal state
        if new_state in MESSAGE_TERMINAL_STATES:
            self._resolve(event)

    def _resolve(self, event: RelayEvent) -> None:
        """Mark the message as completed and fire the on_completed callback."""
        if self._done.done():
            return
        self._done.set_result(event)
        if self._on_completed is not None:
            try:
                result = self._on_completed(event)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                logger.exception(f"Error in on_completed callback for message {self.message_id}")

    def __repr__(self) -> str:
        return (
            f"Message(id={self.message_id!r}, direction={self.direction!r}, "
            f"state={self.state!r}, from={self.from_number!r}, to={self.to_number!r})"
        )
