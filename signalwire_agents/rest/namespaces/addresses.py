"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Addresses namespace — list, create, get, delete (no update).
"""

from .._base import BaseResource


class AddressesResource(BaseResource):
    """Address management (no update endpoint)."""

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/addresses")

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def create(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)

    def get(self, address_id):
        return self._http.get(self._path(address_id))

    def delete(self, address_id):
        return self._http.delete(self._path(address_id))
