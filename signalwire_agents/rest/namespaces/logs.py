"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Logs namespace — message, voice, fax, and conference logs (read-only).
"""

from .._base import BaseResource


class MessageLogs(BaseResource):
    """Message log queries."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, log_id):
        return self._http.get(self._path(log_id))


class VoiceLogs(BaseResource):
    """Voice log queries."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, log_id):
        return self._http.get(self._path(log_id))

    def list_events(self, log_id, **params):
        return self._http.get(self._path(log_id, "events"), params=params or None)


class FaxLogs(BaseResource):
    """Fax log queries."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, log_id):
        return self._http.get(self._path(log_id))


class ConferenceLogs(BaseResource):
    """Conference log queries."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)


class LogsNamespace:
    """Logs API namespace."""

    def __init__(self, http):
        self.messages = MessageLogs(http, "/api/messaging/logs")
        self.voice = VoiceLogs(http, "/api/voice/logs")
        self.fax = FaxLogs(http, "/api/fax/logs")
        self.conferences = ConferenceLogs(http, "/api/logs/conferences")
