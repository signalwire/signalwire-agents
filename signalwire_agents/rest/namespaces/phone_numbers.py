"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Phone Numbers namespace — list, search, purchase, get, update, release.
"""

from .._base import CrudResource


class PhoneNumbersResource(CrudResource):
    """Phone number management."""

    _update_method = "PUT"

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/phone_numbers")

    def search(self, **params):
        return self._http.get(self._path("search"), params=params or None)
