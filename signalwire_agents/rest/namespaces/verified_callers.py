"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Verified Caller IDs namespace — CRUD + verification flow.
"""

from .._base import CrudResource


class VerifiedCallersResource(CrudResource):
    """Verified caller ID management with verification flow."""

    _update_method = "PUT"

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/verified_caller_ids")

    def redial_verification(self, caller_id):
        return self._http.post(self._path(caller_id, "verification"))

    def submit_verification(self, caller_id, **kwargs):
        return self._http.put(self._path(caller_id, "verification"), body=kwargs)
