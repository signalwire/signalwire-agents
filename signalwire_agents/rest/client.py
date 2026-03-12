"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
SignalWireClient — top-level REST client with namespaced sub-objects.
"""

import os
from ._base import HttpClient
from .namespaces.fabric import FabricNamespace
from .namespaces.calling import CallingNamespace
from .namespaces.phone_numbers import PhoneNumbersResource
from .namespaces.addresses import AddressesResource
from .namespaces.queues import QueuesResource
from .namespaces.recordings import RecordingsResource
from .namespaces.number_groups import NumberGroupsResource
from .namespaces.verified_callers import VerifiedCallersResource
from .namespaces.sip_profile import SipProfileResource
from .namespaces.lookup import LookupResource
from .namespaces.short_codes import ShortCodesResource
from .namespaces.imported_numbers import ImportedNumbersResource
from .namespaces.mfa import MfaResource
from .namespaces.registry import RegistryNamespace
from .namespaces.datasphere import DatasphereNamespace
from .namespaces.video import VideoNamespace
from .namespaces.logs import LogsNamespace
from .namespaces.project import ProjectNamespace
from .namespaces.pubsub import PubSubResource
from .namespaces.chat import ChatResource
from .namespaces.compat import CompatNamespace


class SignalWireClient:
    """REST client for the SignalWire platform APIs.

    Usage:
        client = SignalWireClient(
            project="your-project-id",
            token="your-api-token",
            host="your-space.signalwire.com",
        )

        # Or use environment variables:
        #   SIGNALWIRE_PROJECT_ID, SIGNALWIRE_API_TOKEN, SIGNALWIRE_SPACE
        client = SignalWireClient()

        # Use namespaced resources
        client.fabric.ai_agents.list()
        client.calling.play(call_id, play=[...])
        client.phone_numbers.search(area_code="512")
        client.video.rooms.create(name="standup")
        client.compat.calls.list()
    """

    def __init__(self, project=None, token=None, host=None):
        project = project or os.environ.get("SIGNALWIRE_PROJECT_ID", "")
        token = token or os.environ.get("SIGNALWIRE_API_TOKEN", "")
        host = host or os.environ.get("SIGNALWIRE_SPACE", "")

        if not project or not token or not host:
            raise ValueError(
                "project, token, and host are required. "
                "Provide them as arguments or set SIGNALWIRE_PROJECT_ID, "
                "SIGNALWIRE_API_TOKEN, and SIGNALWIRE_SPACE environment variables."
            )

        self._project = project
        self._http = HttpClient(project, token, host)

        # Fabric API
        self.fabric = FabricNamespace(self._http)

        # Calling API
        self.calling = CallingNamespace(self._http)

        # Relay REST resources
        self.phone_numbers = PhoneNumbersResource(self._http)
        self.addresses = AddressesResource(self._http)
        self.queues = QueuesResource(self._http)
        self.recordings = RecordingsResource(self._http)
        self.number_groups = NumberGroupsResource(self._http)
        self.verified_callers = VerifiedCallersResource(self._http)
        self.sip_profile = SipProfileResource(self._http)
        self.lookup = LookupResource(self._http)
        self.short_codes = ShortCodesResource(self._http)
        self.imported_numbers = ImportedNumbersResource(self._http)
        self.mfa = MfaResource(self._http)
        self.registry = RegistryNamespace(self._http)

        # Datasphere API
        self.datasphere = DatasphereNamespace(self._http)

        # Video API
        self.video = VideoNamespace(self._http)

        # Logs
        self.logs = LogsNamespace(self._http)

        # Project management
        self.project = ProjectNamespace(self._http)

        # PubSub & Chat
        self.pubsub = PubSubResource(self._http)
        self.chat = ChatResource(self._http)

        # Compatibility (Twilio-compatible) API
        self.compat = CompatNamespace(self._http, project)
