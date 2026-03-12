"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Queues namespace — CRUD + member management.
"""

from .._base import CrudResource


class QueuesResource(CrudResource):
    """Queue management with member operations."""

    _update_method = "PUT"

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/queues")

    def list_members(self, queue_id, **params):
        return self._http.get(self._path(queue_id, "members"), params=params or None)

    def get_next_member(self, queue_id):
        return self._http.get(self._path(queue_id, "members", "next"))

    def get_member(self, queue_id, member_id):
        return self._http.get(self._path(queue_id, "members", member_id))
