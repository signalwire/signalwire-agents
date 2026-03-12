"""Tests for _base.py — HttpClient, BaseResource, CrudResource, error handling."""

import pytest
from unittest.mock import MagicMock
from signalwire_agents.rest._base import (
    HttpClient,
    SignalWireRestError,
    BaseResource,
    CrudResource,
    CrudWithAddresses,
)
from .conftest import MockResponse


class TestSignalWireRestError:
    def test_error_attributes(self):
        err = SignalWireRestError(404, {"error": "not found"}, "/api/test", "GET")
        assert err.status_code == 404
        assert err.body == {"error": "not found"}
        assert err.url == "/api/test"
        assert err.method == "GET"
        assert "404" in str(err)

    def test_default_method(self):
        err = SignalWireRestError(500, "server error", "/api/x")
        assert err.method == "GET"


class TestHttpClient:
    def test_get(self, http, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": [1, 2]})
        result = http.get("/api/test", params={"page": 1})
        mock_session.request.assert_called_once_with(
            "GET", "https://test.signalwire.com/api/test",
            json=None, params={"page": 1},
        )
        assert result == {"data": [1, 2]}

    def test_post(self, http, mock_session):
        mock_session.request.return_value = MockResponse(201, {"id": "abc"})
        result = http.post("/api/test", body={"name": "x"})
        mock_session.request.assert_called_once_with(
            "POST", "https://test.signalwire.com/api/test",
            json={"name": "x"}, params=None,
        )
        assert result == {"id": "abc"}

    def test_put(self, http, mock_session):
        mock_session.request.return_value = MockResponse(200, {"ok": True})
        result = http.put("/api/test/1", body={"name": "y"})
        mock_session.request.assert_called_once_with(
            "PUT", "https://test.signalwire.com/api/test/1",
            json={"name": "y"}, params=None,
        )

    def test_patch(self, http, mock_session):
        mock_session.request.return_value = MockResponse(200, {"ok": True})
        http.patch("/api/test/1", body={"name": "z"})
        mock_session.request.assert_called_once_with(
            "PATCH", "https://test.signalwire.com/api/test/1",
            json={"name": "z"}, params=None,
        )

    def test_delete(self, http, mock_session):
        mock_session.request.return_value = MockResponse(204, None, content=b"")
        result = http.delete("/api/test/1")
        mock_session.request.assert_called_once_with(
            "DELETE", "https://test.signalwire.com/api/test/1",
            json=None, params=None,
        )
        assert result == {}

    def test_error_raises(self, http, mock_session):
        mock_session.request.return_value = MockResponse(
            404, {"error": "not found"}, content=b'{"error":"not found"}',
        )
        with pytest.raises(SignalWireRestError) as exc_info:
            http.get("/api/missing")
        assert exc_info.value.status_code == 404


class TestCrudResource:
    def test_list(self, http, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        res = CrudResource(http, "/api/items")
        result = res.list(page=1)
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/items",
            json=None, params={"page": 1},
        )

    def test_create(self, http, mock_session):
        mock_session.request.return_value = MockResponse(201, {"id": "new"})
        res = CrudResource(http, "/api/items")
        result = res.create(name="test")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/items",
            json={"name": "test"}, params=None,
        )

    def test_get(self, http, mock_session):
        mock_session.request.return_value = MockResponse(200, {"id": "abc"})
        res = CrudResource(http, "/api/items")
        result = res.get("abc")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/items/abc",
            json=None, params=None,
        )

    def test_update_patch(self, http, mock_session):
        mock_session.request.return_value = MockResponse(200, {"ok": True})
        res = CrudResource(http, "/api/items")
        res.update("abc", name="updated")
        mock_session.request.assert_called_with(
            "PATCH", "https://test.signalwire.com/api/items/abc",
            json={"name": "updated"}, params=None,
        )

    def test_update_put(self, http, mock_session):
        mock_session.request.return_value = MockResponse(200, {"ok": True})

        class PutResource(CrudResource):
            _update_method = "PUT"

        res = PutResource(http, "/api/items")
        res.update("abc", name="updated")
        mock_session.request.assert_called_with(
            "PUT", "https://test.signalwire.com/api/items/abc",
            json={"name": "updated"}, params=None,
        )

    def test_delete(self, http, mock_session):
        mock_session.request.return_value = MockResponse(204, None, content=b"")
        res = CrudResource(http, "/api/items")
        res.delete("abc")
        mock_session.request.assert_called_with(
            "DELETE", "https://test.signalwire.com/api/items/abc",
            json=None, params=None,
        )


class TestCrudWithAddresses:
    def test_list_addresses(self, http, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        res = CrudWithAddresses(http, "/api/fabric/resources/ai_agents")
        res.list_addresses("abc")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/fabric/resources/ai_agents/abc/addresses",
            json=None, params=None,
        )
