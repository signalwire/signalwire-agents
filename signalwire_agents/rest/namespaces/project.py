"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Project API namespace — API token management.
"""

from .._base import BaseResource


class ProjectTokens(BaseResource):
    """Project API token management."""

    def __init__(self, http):
        super().__init__(http, "/api/project/tokens")

    def create(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)

    def update(self, token_id, **kwargs):
        return self._http.patch(self._path(token_id), body=kwargs)

    def delete(self, token_id):
        return self._http.delete(self._path(token_id))


class ProjectNamespace:
    """Project API namespace."""

    def __init__(self, http):
        self.tokens = ProjectTokens(http)
