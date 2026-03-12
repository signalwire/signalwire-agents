"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Video API namespace — rooms, sessions, recordings, conferences, tokens, streams.
"""

from .._base import BaseResource, CrudResource


class VideoRooms(CrudResource):
    """Video room management with streams."""

    _update_method = "PUT"

    def list_streams(self, room_id, **params):
        return self._http.get(self._path(room_id, "streams"), params=params or None)

    def create_stream(self, room_id, **kwargs):
        return self._http.post(self._path(room_id, "streams"), body=kwargs)


class VideoRoomTokens(BaseResource):
    """Video room token generation."""

    def create(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)


class VideoRoomSessions(BaseResource):
    """Video room session management."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, session_id):
        return self._http.get(self._path(session_id))

    def list_events(self, session_id, **params):
        return self._http.get(self._path(session_id, "events"), params=params or None)

    def list_members(self, session_id, **params):
        return self._http.get(self._path(session_id, "members"), params=params or None)

    def list_recordings(self, session_id, **params):
        return self._http.get(self._path(session_id, "recordings"), params=params or None)


class VideoRoomRecordings(BaseResource):
    """Video room recording management."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, recording_id):
        return self._http.get(self._path(recording_id))

    def delete(self, recording_id):
        return self._http.delete(self._path(recording_id))

    def list_events(self, recording_id, **params):
        return self._http.get(self._path(recording_id, "events"), params=params or None)


class VideoConferences(CrudResource):
    """Video conference management with tokens and streams."""

    _update_method = "PUT"

    def list_conference_tokens(self, conference_id, **params):
        return self._http.get(
            self._path(conference_id, "conference_tokens"),
            params=params or None,
        )

    def list_streams(self, conference_id, **params):
        return self._http.get(self._path(conference_id, "streams"), params=params or None)

    def create_stream(self, conference_id, **kwargs):
        return self._http.post(self._path(conference_id, "streams"), body=kwargs)


class VideoConferenceTokens(BaseResource):
    """Video conference token management."""

    def get(self, token_id):
        return self._http.get(self._path(token_id))

    def reset(self, token_id):
        return self._http.post(self._path(token_id, "reset"))


class VideoStreams(BaseResource):
    """Video stream management."""

    def get(self, stream_id):
        return self._http.get(self._path(stream_id))

    def update(self, stream_id, **kwargs):
        return self._http.put(self._path(stream_id), body=kwargs)

    def delete(self, stream_id):
        return self._http.delete(self._path(stream_id))


class VideoNamespace:
    """Video API namespace."""

    def __init__(self, http):
        base = "/api/video"
        self.rooms = VideoRooms(http, f"{base}/rooms")
        self.room_tokens = VideoRoomTokens(http, f"{base}/room_tokens")
        self.room_sessions = VideoRoomSessions(http, f"{base}/room_sessions")
        self.room_recordings = VideoRoomRecordings(http, f"{base}/room_recordings")
        self.conferences = VideoConferences(http, f"{base}/conferences")
        self.conference_tokens = VideoConferenceTokens(http, f"{base}/conference_tokens")
        self.streams = VideoStreams(http, f"{base}/streams")
