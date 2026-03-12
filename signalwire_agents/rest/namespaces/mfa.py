"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
MFA (Multi-Factor Authentication) namespace.
"""

from .._base import BaseResource


class MfaResource(BaseResource):
    """Multi-factor authentication via SMS or phone call."""

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/mfa")

    def sms(self, **kwargs):
        return self._http.post(self._path("sms"), body=kwargs)

    def call(self, **kwargs):
        return self._http.post(self._path("call"), body=kwargs)

    def verify(self, request_id, **kwargs):
        return self._http.post(self._path(request_id, "verify"), body=kwargs)
