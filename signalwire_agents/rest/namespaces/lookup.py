"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Phone Number Lookup namespace.
"""

from .._base import BaseResource


class LookupResource(BaseResource):
    """Phone number lookup (carrier, CNAM)."""

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/lookup")

    def phone_number(self, e164, **params):
        return self._http.get(self._path("phone_number", e164), params=params or None)
