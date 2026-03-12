"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
HTTP client infrastructure and base resource classes for the REST client.
"""

import requests
from signalwire_agents.core.logging_config import get_logger

logger = get_logger("rest_client")


class SignalWireRestError(Exception):
    """Raised when the SignalWire REST API returns a non-2xx response."""

    def __init__(self, status_code, body, url, method="GET"):
        self.status_code = status_code
        self.body = body
        self.url = url
        self.method = method
        message = f"{method} {url} returned {status_code}: {body}"
        super().__init__(message)


class HttpClient:
    """Thin wrapper around requests.Session with Basic Auth and JSON handling."""

    def __init__(self, project, token, host):
        self._base_url = f"https://{host}"
        self._session = requests.Session()
        self._session.auth = (project, token)
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "signalwire-agents-python-rest/1.0",
        })
        logger.debug("HttpClient initialized", host=host, project=project[:8] + "...")

    def _request(self, method, path, body=None, params=None):
        url = self._base_url + path
        logger.debug("REST request", method=method, path=path)
        resp = self._session.request(method, url, json=body, params=params)
        if not resp.ok:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            raise SignalWireRestError(resp.status_code, err_body, path, method)
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    def get(self, path, params=None):
        return self._request("GET", path, params=params)

    def post(self, path, body=None, params=None):
        return self._request("POST", path, body=body, params=params)

    def put(self, path, body=None):
        return self._request("PUT", path, body=body)

    def patch(self, path, body=None):
        return self._request("PATCH", path, body=body)

    def delete(self, path):
        return self._request("DELETE", path)


class BaseResource:
    """Base for all namespace/resource classes."""

    def __init__(self, http, base_path):
        self._http = http
        self._base_path = base_path

    def _path(self, *parts):
        return "/".join([self._base_path] + [str(p) for p in parts])


class CrudResource(BaseResource):
    """Standard CRUD resource with list/create/get/update/delete."""

    _update_method = "PATCH"

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def create(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)

    def get(self, resource_id):
        return self._http.get(self._path(resource_id))

    def update(self, resource_id, **kwargs):
        method = getattr(self._http, self._update_method.lower())
        return method(self._path(resource_id), body=kwargs)

    def delete(self, resource_id):
        return self._http.delete(self._path(resource_id))


class CrudWithAddresses(CrudResource):
    """CRUD resource that also supports listing addresses."""

    def list_addresses(self, resource_id, **params):
        return self._http.get(self._path(resource_id, "addresses"), params=params or None)
