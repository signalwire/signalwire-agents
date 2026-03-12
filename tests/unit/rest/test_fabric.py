"""Tests for fabric namespace — resource types, addresses, tokens."""

from .conftest import MockResponse


class TestFabricResources:
    def test_ai_agents_list(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.fabric.ai_agents.list()
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/fabric/resources/ai_agents",
            json=None, params=None,
        )

    def test_ai_agents_create(self, client, mock_session):
        mock_session.request.return_value = MockResponse(201, {"id": "new"})
        client.fabric.ai_agents.create(name="Support", prompt="You are helpful")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/fabric/resources/ai_agents",
            json={"name": "Support", "prompt": "You are helpful"}, params=None,
        )

    def test_ai_agents_update_uses_patch(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.fabric.ai_agents.update("id-1", name="Updated")
        mock_session.request.assert_called_with(
            "PATCH", "https://test.signalwire.com/api/fabric/resources/ai_agents/id-1",
            json={"name": "Updated"}, params=None,
        )

    def test_swml_scripts_update_uses_put(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.fabric.swml_scripts.update("id-1", contents="{}")
        mock_session.request.assert_called_with(
            "PUT", "https://test.signalwire.com/api/fabric/resources/swml_scripts/id-1",
            json={"contents": "{}"}, params=None,
        )

    def test_list_addresses(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.fabric.ai_agents.list_addresses("id-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/fabric/resources/ai_agents/id-1/addresses",
            json=None, params=None,
        )


class TestFabricCallFlows:
    def test_list_versions(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.fabric.call_flows.list_versions("cf-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/fabric/resources/call_flow/cf-1/versions",
            json=None, params=None,
        )

    def test_deploy_version(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.fabric.call_flows.deploy_version("cf-1", document_version=2)
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/fabric/resources/call_flow/cf-1/versions",
            json={"document_version": 2}, params=None,
        )


class TestFabricSubscribers:
    def test_list_sip_endpoints(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.fabric.subscribers.list_sip_endpoints("sub-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/fabric/resources/subscribers/sub-1/sip_endpoints",
            json=None, params=None,
        )

    def test_create_sip_endpoint(self, client, mock_session):
        mock_session.request.return_value = MockResponse(201, {"id": "ep-1"})
        client.fabric.subscribers.create_sip_endpoint("sub-1", username="user1")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/fabric/resources/subscribers/sub-1/sip_endpoints",
            json={"username": "user1"}, params=None,
        )


class TestGenericResources:
    def test_assign_phone_route(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.fabric.resources.assign_phone_route("res-1", phone_route_id="pr-1")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/fabric/resources/res-1/phone_routes",
            json={"phone_route_id": "pr-1"}, params=None,
        )


class TestFabricTokens:
    def test_create_subscriber_token(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"token": "abc"})
        client.fabric.tokens.create_subscriber_token(reference="user@example.com")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/fabric/subscribers/tokens",
            json={"reference": "user@example.com"}, params=None,
        )

    def test_create_guest_token(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"token": "abc"})
        client.fabric.tokens.create_guest_token(allowed_addresses=["addr-1"])
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/fabric/guests/tokens",
            json={"allowed_addresses": ["addr-1"]}, params=None,
        )


class TestAllFabricResources:
    """Verify all 13+ resource types exist on the fabric namespace."""

    def test_resource_types_exist(self, client):
        resources = [
            "swml_scripts", "swml_webhooks", "ai_agents", "relay_applications",
            "call_flows", "conference_rooms", "freeswitch_connectors",
            "subscribers", "sip_endpoints", "sip_gateways",
            "cxml_scripts", "cxml_webhooks", "cxml_applications",
        ]
        for name in resources:
            assert hasattr(client.fabric, name), f"Missing fabric resource: {name}"

    def test_special_resources_exist(self, client):
        assert hasattr(client.fabric, "resources")
        assert hasattr(client.fabric, "addresses")
        assert hasattr(client.fabric, "tokens")
