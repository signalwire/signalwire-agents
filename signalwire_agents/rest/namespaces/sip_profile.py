"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
SIP Profile namespace — get and update project SIP profile.
"""

from .._base import BaseResource


class SipProfileResource(BaseResource):
    """Project SIP profile (singleton resource)."""

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/sip_profile")

    def get(self):
        return self._http.get(self._base_path)

    def update(self, **kwargs):
        return self._http.put(self._base_path, body=kwargs)
