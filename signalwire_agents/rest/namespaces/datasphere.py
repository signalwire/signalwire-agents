"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Datasphere API namespace — document management and semantic search.
"""

from .._base import CrudResource


class DatasphereDocuments(CrudResource):
    """Document management with search and chunk operations."""

    def __init__(self, http):
        super().__init__(http, "/api/datasphere/documents")

    def search(self, **kwargs):
        return self._http.post(self._path("search"), body=kwargs)

    def list_chunks(self, document_id, **params):
        return self._http.get(self._path(document_id, "chunks"), params=params or None)

    def get_chunk(self, document_id, chunk_id):
        return self._http.get(self._path(document_id, "chunks", chunk_id))

    def delete_chunk(self, document_id, chunk_id):
        return self._http.delete(self._path(document_id, "chunks", chunk_id))


class DatasphereNamespace:
    """Datasphere API namespace."""

    def __init__(self, http):
        self.documents = DatasphereDocuments(http)
