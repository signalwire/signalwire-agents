"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Chat API namespace — token creation.
"""

from .._base import BaseResource


class ChatResource(BaseResource):
    """Chat token generation."""

    def __init__(self, http):
        super().__init__(http, "/api/chat/tokens")

    def create_token(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)
