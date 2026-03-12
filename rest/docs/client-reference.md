# SignalWireClient Reference

## Constructor

```python
SignalWireClient(
    project: str = None,   # SIGNALWIRE_PROJECT_ID
    token: str = None,     # SIGNALWIRE_API_TOKEN
    host: str = None,      # SIGNALWIRE_SPACE
)
```

All parameters fall back to their corresponding environment variables. A `ValueError` is raised if any are missing.

Authentication uses HTTP Basic Auth (`project:token`).

## Namespaces

Every API surface is available as a namespace attribute on the client:

### Fabric API

| Attribute | Description |
|-----------|-------------|
| `client.fabric.swml_scripts` | SWML script resources (CRUD + addresses) |
| `client.fabric.swml_webhooks` | SWML webhook resources |
| `client.fabric.ai_agents` | AI agent resources |
| `client.fabric.relay_applications` | Relay application resources |
| `client.fabric.call_flows` | Call flow resources (+ versions) |
| `client.fabric.conference_rooms` | Conference room resources |
| `client.fabric.freeswitch_connectors` | FreeSWITCH connector resources |
| `client.fabric.subscribers` | Subscriber resources (+ SIP endpoints) |
| `client.fabric.sip_endpoints` | SIP endpoint resources |
| `client.fabric.sip_gateways` | SIP gateway resources |
| `client.fabric.cxml_scripts` | cXML script resources |
| `client.fabric.cxml_webhooks` | cXML webhook resources |
| `client.fabric.cxml_applications` | cXML application resources (no create) |
| `client.fabric.resources` | Generic resource operations |
| `client.fabric.addresses` | Fabric addresses (list/get only) |
| `client.fabric.tokens` | Subscriber/guest/invite/embed token creation |

### Calling API

| Attribute | Description |
|-----------|-------------|
| `client.calling` | REST call control -- 37 commands via POST |

### Relay REST Resources

| Attribute | Description |
|-----------|-------------|
| `client.phone_numbers` | Phone number management (+ search) |
| `client.addresses` | Address management |
| `client.queues` | Queue management (+ members) |
| `client.recordings` | Recording management |
| `client.number_groups` | Number group management (+ memberships) |
| `client.verified_callers` | Verified caller ID management (+ verification flow) |
| `client.sip_profile` | Project SIP profile (get/update) |
| `client.lookup` | Phone number lookup |
| `client.short_codes` | Short code management |
| `client.imported_numbers` | Import external phone numbers |
| `client.mfa` | Multi-factor authentication (SMS/call/verify) |
| `client.registry` | 10DLC brand/campaign registry |

### Other APIs

| Attribute | Description |
|-----------|-------------|
| `client.datasphere` | Datasphere document management and semantic search |
| `client.video` | Video rooms, sessions, recordings, conferences |
| `client.logs` | Message, voice, fax, and conference logs |
| `client.project` | API token management |
| `client.pubsub` | PubSub token creation |
| `client.chat` | Chat token creation |
| `client.compat` | Twilio-compatible LAML API |

## Error Handling

```python
from signalwire_agents.rest import SignalWireRestError

try:
    client.fabric.ai_agents.get("bad-id")
except SignalWireRestError as e:
    print(e.status_code)  # 404
    print(e.body)         # {"error": "not found"}
    print(e.url)          # "/api/fabric/resources/ai_agents/bad-id"
    print(e.method)       # "GET"
```

`SignalWireRestError` is raised on any non-2xx HTTP response.

### Error Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `status_code` | `int` | HTTP status code |
| `body` | `dict` or `str` | Response body (parsed JSON or raw text) |
| `url` | `str` | Request path |
| `method` | `str` | HTTP method |

## Session Behavior

- A single `requests.Session` is shared across all namespaces for connection pooling.
- Content-Type is always `application/json`.
- User-Agent is `signalwire-agents-python-rest/1.0`.
- DELETE requests returning 204 return an empty dict.
