"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Pagination support for list endpoints that return paged results.
"""


class PaginatedIterator:
    """Iterates items across paginated API responses.

    Usage:
        for item in PaginatedIterator(http, "/api/path", params={}, data_key="data"):
            print(item)
    """

    def __init__(self, http, path, params=None, data_key="data"):
        self._http = http
        self._path = path
        self._params = dict(params or {})
        self._data_key = data_key
        self._current_page = None
        self._items = []
        self._index = 0
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        while self._index >= len(self._items):
            if self._done:
                raise StopIteration
            self._fetch_next()

        item = self._items[self._index]
        self._index += 1
        return item

    def _fetch_next(self):
        resp = self._http.get(self._path, params=self._params or None)
        data = resp.get(self._data_key, [])
        self._items.extend(data)

        links = resp.get("links", {})
        next_url = links.get("next")
        if next_url and data:
            # Parse cursor/page token from next URL if present
            # Most SignalWire APIs use page_token or cursor param
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(next_url)
            query = parse_qs(parsed.query)
            # Flatten single-value lists
            self._params = {k: v[0] if len(v) == 1 else v for k, v in query.items()}
        else:
            self._done = True
