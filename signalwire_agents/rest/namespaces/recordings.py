"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Recordings namespace — list, get, delete (no create/update).
"""

from .._base import BaseResource


class RecordingsResource(BaseResource):
    """Recording management (read-only + delete)."""

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/recordings")

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, recording_id):
        return self._http.get(self._path(recording_id))

    def delete(self, recording_id):
        return self._http.delete(self._path(recording_id))
