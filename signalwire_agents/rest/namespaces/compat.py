"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Compatibility API namespace — Twilio-compatible LAML API with AccountSid scoping.
"""

from .._base import BaseResource, CrudResource


class CompatAccounts(BaseResource):
    """Compat account/subproject management."""

    def __init__(self, http):
        super().__init__(http, "/api/laml/2010-04-01/Accounts")

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def create(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)

    def get(self, sid):
        return self._http.get(self._path(sid))

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)


class CompatCalls(CrudResource):
    """Compat call management with recording and stream sub-resources."""

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)

    def start_recording(self, call_sid, **kwargs):
        return self._http.post(self._path(call_sid, "Recordings"), body=kwargs)

    def update_recording(self, call_sid, recording_sid, **kwargs):
        return self._http.post(self._path(call_sid, "Recordings", recording_sid), body=kwargs)

    def start_stream(self, call_sid, **kwargs):
        return self._http.post(self._path(call_sid, "Streams"), body=kwargs)

    def stop_stream(self, call_sid, stream_sid, **kwargs):
        return self._http.post(self._path(call_sid, "Streams", stream_sid), body=kwargs)


class CompatMessages(CrudResource):
    """Compat message management with media sub-resources."""

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)

    def list_media(self, message_sid, **params):
        return self._http.get(self._path(message_sid, "Media"), params=params or None)

    def get_media(self, message_sid, media_sid):
        return self._http.get(self._path(message_sid, "Media", media_sid))

    def delete_media(self, message_sid, media_sid):
        return self._http.delete(self._path(message_sid, "Media", media_sid))


class CompatFaxes(CrudResource):
    """Compat fax management with media sub-resources."""

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)

    def list_media(self, fax_sid, **params):
        return self._http.get(self._path(fax_sid, "Media"), params=params or None)

    def get_media(self, fax_sid, media_sid):
        return self._http.get(self._path(fax_sid, "Media", media_sid))

    def delete_media(self, fax_sid, media_sid):
        return self._http.delete(self._path(fax_sid, "Media", media_sid))


class CompatConferences(BaseResource):
    """Compat conference management with participants, recordings, and streams."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, sid):
        return self._http.get(self._path(sid))

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)

    # Participants
    def list_participants(self, conference_sid, **params):
        return self._http.get(
            self._path(conference_sid, "Participants"),
            params=params or None,
        )

    def get_participant(self, conference_sid, call_sid):
        return self._http.get(self._path(conference_sid, "Participants", call_sid))

    def update_participant(self, conference_sid, call_sid, **kwargs):
        return self._http.post(
            self._path(conference_sid, "Participants", call_sid),
            body=kwargs,
        )

    def remove_participant(self, conference_sid, call_sid):
        return self._http.delete(self._path(conference_sid, "Participants", call_sid))

    # Conference recordings
    def list_recordings(self, conference_sid, **params):
        return self._http.get(
            self._path(conference_sid, "Recordings"),
            params=params or None,
        )

    def get_recording(self, conference_sid, recording_sid):
        return self._http.get(self._path(conference_sid, "Recordings", recording_sid))

    def update_recording(self, conference_sid, recording_sid, **kwargs):
        return self._http.post(
            self._path(conference_sid, "Recordings", recording_sid),
            body=kwargs,
        )

    def delete_recording(self, conference_sid, recording_sid):
        return self._http.delete(self._path(conference_sid, "Recordings", recording_sid))

    # Conference streams
    def start_stream(self, conference_sid, **kwargs):
        return self._http.post(self._path(conference_sid, "Streams"), body=kwargs)

    def stop_stream(self, conference_sid, stream_sid, **kwargs):
        return self._http.post(
            self._path(conference_sid, "Streams", stream_sid),
            body=kwargs,
        )


class CompatPhoneNumbers(BaseResource):
    """Compat phone number management."""

    def __init__(self, http, base):
        super().__init__(http, base)
        self._available_base = base.replace("/IncomingPhoneNumbers", "/AvailablePhoneNumbers")

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def purchase(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)

    def get(self, sid):
        return self._http.get(self._path(sid))

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)

    def delete(self, sid):
        return self._http.delete(self._path(sid))

    def import_number(self, **kwargs):
        path = self._base_path.replace("/IncomingPhoneNumbers", "/ImportedPhoneNumbers")
        return self._http.post(path, body=kwargs)

    def list_available_countries(self, **params):
        return self._http.get(self._available_base, params=params or None)

    def search_local(self, country, **params):
        return self._http.get(f"{self._available_base}/{country}/Local", params=params or None)

    def search_toll_free(self, country, **params):
        return self._http.get(f"{self._available_base}/{country}/TollFree", params=params or None)


class CompatApplications(CrudResource):
    """Compat application management."""

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)


class CompatLamlBins(CrudResource):
    """Compat cXML/LaML script management."""

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)


class CompatQueues(CrudResource):
    """Compat queue management with members."""

    def update(self, sid, **kwargs):
        return self._http.post(self._path(sid), body=kwargs)

    def list_members(self, queue_sid, **params):
        return self._http.get(self._path(queue_sid, "Members"), params=params or None)

    def get_member(self, queue_sid, call_sid):
        return self._http.get(self._path(queue_sid, "Members", call_sid))

    def dequeue_member(self, queue_sid, call_sid, **kwargs):
        return self._http.post(self._path(queue_sid, "Members", call_sid), body=kwargs)


class CompatRecordings(BaseResource):
    """Compat recording management."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, sid):
        return self._http.get(self._path(sid))

    def delete(self, sid):
        return self._http.delete(self._path(sid))


class CompatTranscriptions(BaseResource):
    """Compat transcription management."""

    def list(self, **params):
        return self._http.get(self._base_path, params=params or None)

    def get(self, sid):
        return self._http.get(self._path(sid))

    def delete(self, sid):
        return self._http.delete(self._path(sid))


class CompatTokens(BaseResource):
    """Compat API token management."""

    def create(self, **kwargs):
        return self._http.post(self._base_path, body=kwargs)

    def update(self, token_id, **kwargs):
        return self._http.patch(self._path(token_id), body=kwargs)

    def delete(self, token_id):
        return self._http.delete(self._path(token_id))


class CompatNamespace:
    """Twilio-compatible LAML API namespace with AccountSid scoping."""

    def __init__(self, http, account_sid):
        base = f"/api/laml/2010-04-01/Accounts/{account_sid}"

        self.accounts = CompatAccounts(http)
        self.calls = CompatCalls(http, f"{base}/Calls")
        self.messages = CompatMessages(http, f"{base}/Messages")
        self.faxes = CompatFaxes(http, f"{base}/Faxes")
        self.conferences = CompatConferences(http, f"{base}/Conferences")
        self.phone_numbers = CompatPhoneNumbers(http, f"{base}/IncomingPhoneNumbers")
        self.applications = CompatApplications(http, f"{base}/Applications")
        self.laml_bins = CompatLamlBins(http, f"{base}/LamlBins")
        self.queues = CompatQueues(http, f"{base}/Queues")
        self.recordings = CompatRecordings(http, f"{base}/Recordings")
        self.transcriptions = CompatTranscriptions(http, f"{base}/Transcriptions")
        self.tokens = CompatTokens(http, f"{base}/tokens")
