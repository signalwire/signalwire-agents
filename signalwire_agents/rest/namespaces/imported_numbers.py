"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Imported Phone Numbers namespace — create only.
"""

from .._base import BaseResource


class ImportedNumbersResource(BaseResource):
    """Import externally-hosted phone numbers."""

    def __init__(self, http):
        super().__init__(http, "/api/relay/rest/imported_phone_numbers")

    def create(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)
