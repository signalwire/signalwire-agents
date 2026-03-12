"""Tests for remaining namespaces — phone_numbers, video, compat, datasphere, etc."""

from .conftest import MockResponse


class TestPhoneNumbers:
    def test_search(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.phone_numbers.search(area_code="512")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/relay/rest/phone_numbers/search",
            json=None, params={"area_code": "512"},
        )

    def test_update_uses_put(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.phone_numbers.update("pn-1", name="Main")
        mock_session.request.assert_called_with(
            "PUT", "https://test.signalwire.com/api/relay/rest/phone_numbers/pn-1",
            json={"name": "Main"}, params=None,
        )


class TestQueues:
    def test_list_members(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.queues.list_members("q-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/relay/rest/queues/q-1/members",
            json=None, params=None,
        )

    def test_get_next_member(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.queues.get_next_member("q-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/relay/rest/queues/q-1/members/next",
            json=None, params=None,
        )


class TestNumberGroups:
    def test_add_membership(self, client, mock_session):
        mock_session.request.return_value = MockResponse(201, {})
        client.number_groups.add_membership("ng-1", phone_number_id="pn-1")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/relay/rest/number_groups/ng-1/number_group_memberships",
            json={"phone_number_id": "pn-1"}, params=None,
        )

    def test_get_membership(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.number_groups.get_membership("mem-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/relay/rest/number_group_memberships/mem-1",
            json=None, params=None,
        )


class TestVerifiedCallers:
    def test_redial_verification(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.verified_callers.redial_verification("vc-1")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/relay/rest/verified_caller_ids/vc-1/verification",
            json=None, params=None,
        )

    def test_submit_verification(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.verified_callers.submit_verification("vc-1", code="123456")
        mock_session.request.assert_called_with(
            "PUT", "https://test.signalwire.com/api/relay/rest/verified_caller_ids/vc-1/verification",
            json={"code": "123456"}, params=None,
        )


class TestSipProfile:
    def test_get(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"sip_uri": "test"})
        client.sip_profile.get()
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/relay/rest/sip_profile",
            json=None, params=None,
        )


class TestLookup:
    def test_phone_number(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.lookup.phone_number("+15551234567", include="carrier")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/relay/rest/lookup/phone_number/+15551234567",
            json=None, params={"include": "carrier"},
        )


class TestMfa:
    def test_sms(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"id": "mfa-1"})
        client.mfa.sms(to="+15551234567", from_="+15559876543")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/relay/rest/mfa/sms",
            json={"to": "+15551234567", "from_": "+15559876543"}, params=None,
        )

    def test_verify(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"success": True})
        client.mfa.verify("mfa-1", token="123456")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/relay/rest/mfa/mfa-1/verify",
            json={"token": "123456"}, params=None,
        )


class TestDatasphere:
    def test_search(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.datasphere.documents.search(query_string="billing")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/datasphere/documents/search",
            json={"query_string": "billing"}, params=None,
        )

    def test_list_chunks(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.datasphere.documents.list_chunks("doc-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/datasphere/documents/doc-1/chunks",
            json=None, params=None,
        )

    def test_delete_chunk(self, client, mock_session):
        mock_session.request.return_value = MockResponse(204, None, content=b"")
        client.datasphere.documents.delete_chunk("doc-1", "chunk-1")
        mock_session.request.assert_called_with(
            "DELETE", "https://test.signalwire.com/api/datasphere/documents/doc-1/chunks/chunk-1",
            json=None, params=None,
        )


class TestVideo:
    def test_rooms_create(self, client, mock_session):
        mock_session.request.return_value = MockResponse(201, {"id": "room-1"})
        client.video.rooms.create(name="standup")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/video/rooms",
            json={"name": "standup"}, params=None,
        )

    def test_room_tokens_create(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"token": "abc"})
        client.video.room_tokens.create(room_name="standup", user_name="alice")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/video/room_tokens",
            json={"room_name": "standup", "user_name": "alice"}, params=None,
        )

    def test_session_members(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.video.room_sessions.list_members("sess-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/video/room_sessions/sess-1/members",
            json=None, params=None,
        )

    def test_conference_streams(self, client, mock_session):
        mock_session.request.return_value = MockResponse(201, {})
        client.video.conferences.create_stream("conf-1", url="rtmp://example.com/live")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/video/conferences/conf-1/streams",
            json={"url": "rtmp://example.com/live"}, params=None,
        )


class TestLogs:
    def test_voice_events(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.logs.voice.list_events("log-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/voice/logs/log-1/events",
            json=None, params=None,
        )


class TestRegistry:
    def test_create_brand(self, client, mock_session):
        mock_session.request.return_value = MockResponse(201, {"id": "brand-1"})
        client.registry.brands.create(name="MyBrand")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/relay/rest/registry/beta/brands",
            json={"name": "MyBrand"}, params=None,
        )

    def test_campaign_orders(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"data": []})
        client.registry.campaigns.list_orders("camp-1")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/relay/rest/registry/beta/campaigns/camp-1/orders",
            json=None, params=None,
        )


class TestCompat:
    def test_calls_list(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"calls": []})
        client.compat.calls.list()
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/laml/2010-04-01/Accounts/test-project-id/Calls",
            json=None, params=None,
        )

    def test_calls_update_uses_post(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {})
        client.compat.calls.update("CA123", Status="completed")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/laml/2010-04-01/Accounts/test-project-id/Calls/CA123",
            json={"Status": "completed"}, params=None,
        )

    def test_call_start_recording(self, client, mock_session):
        mock_session.request.return_value = MockResponse(201, {})
        client.compat.calls.start_recording("CA123", channels="dual")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/laml/2010-04-01/Accounts/test-project-id/Calls/CA123/Recordings",
            json={"channels": "dual"}, params=None,
        )

    def test_messages_media(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"media": []})
        client.compat.messages.list_media("MM123")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/laml/2010-04-01/Accounts/test-project-id/Messages/MM123/Media",
            json=None, params=None,
        )

    def test_conference_participants(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"participants": []})
        client.compat.conferences.list_participants("CF123")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/laml/2010-04-01/Accounts/test-project-id/Conferences/CF123/Participants",
            json=None, params=None,
        )

    def test_phone_numbers_search_local(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"available": []})
        client.compat.phone_numbers.search_local("US", AreaCode="512")
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/laml/2010-04-01/Accounts/test-project-id/AvailablePhoneNumbers/US/Local",
            json=None, params={"AreaCode": "512"},
        )

    def test_accounts_list(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"accounts": []})
        client.compat.accounts.list()
        mock_session.request.assert_called_with(
            "GET", "https://test.signalwire.com/api/laml/2010-04-01/Accounts",
            json=None, params=None,
        )


class TestProjectTokens:
    def test_create_token(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"id": "tok-1"})
        client.project.tokens.create(name="test-token")
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/project/tokens",
            json={"name": "test-token"}, params=None,
        )


class TestPubSubChat:
    def test_pubsub_token(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"token": "abc"})
        client.pubsub.create_token(ttl=60)
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/pubsub/tokens",
            json={"ttl": 60}, params=None,
        )

    def test_chat_token(self, client, mock_session):
        mock_session.request.return_value = MockResponse(200, {"token": "abc"})
        client.chat.create_token(ttl=60)
        mock_session.request.assert_called_with(
            "POST", "https://test.signalwire.com/api/chat/tokens",
            json={"ttl": 60}, params=None,
        )
