# Fabric Resources

The Fabric API (`/api/fabric`) manages all resource types in your SignalWire project. Every resource type supports CRUD operations and address listing.

## Standard CRUD Pattern

All 13 resource types share the same methods:

```python
# List all resources of this type
items = client.fabric.ai_agents.list()
items = client.fabric.ai_agents.list(page=2, page_size=10)

# Create a new resource
agent = client.fabric.ai_agents.create(
    name="Support Bot",
    prompt={"text": "You are a helpful support agent."},
)

# Get a resource by ID
agent = client.fabric.ai_agents.get("agent-uuid")

# Update a resource
client.fabric.ai_agents.update("agent-uuid", name="Updated Name")

# Delete a resource
client.fabric.ai_agents.delete("agent-uuid")

# List addresses assigned to this resource
addresses = client.fabric.ai_agents.list_addresses("agent-uuid")
```

## Resource Types

### PUT-Update Resources

These resources use `PUT` for updates (full replacement):

| Attribute | API Path |
|-----------|----------|
| `fabric.swml_scripts` | `/api/fabric/resources/swml_scripts` |
| `fabric.relay_applications` | `/api/fabric/resources/relay_applications` |
| `fabric.call_flows` | `/api/fabric/resources/call_flows` |
| `fabric.conference_rooms` | `/api/fabric/resources/conference_rooms` |
| `fabric.freeswitch_connectors` | `/api/fabric/resources/freeswitch_connectors` |
| `fabric.subscribers` | `/api/fabric/resources/subscribers` |
| `fabric.sip_endpoints` | `/api/fabric/resources/sip_endpoints` |
| `fabric.cxml_scripts` | `/api/fabric/resources/cxml_scripts` |
| `fabric.cxml_applications` | `/api/fabric/resources/cxml_applications` |

### PATCH-Update Resources

These resources use `PATCH` for updates (partial update):

| Attribute | API Path |
|-----------|----------|
| `fabric.swml_webhooks` | `/api/fabric/resources/swml_webhooks` |
| `fabric.ai_agents` | `/api/fabric/resources/ai_agents` |
| `fabric.sip_gateways` | `/api/fabric/resources/sip_gateways` |
| `fabric.cxml_webhooks` | `/api/fabric/resources/cxml_webhooks` |

## Call Flows -- Extra Methods

Call flows support version management:

```python
# List all versions of a call flow
versions = client.fabric.call_flows.list_versions("call-flow-uuid")

# Deploy a new version
client.fabric.call_flows.deploy_version("call-flow-uuid", document_version=3)
```

## Subscribers -- SIP Endpoints

Subscribers have nested SIP endpoint management:

```python
# List subscriber's SIP endpoints
endpoints = client.fabric.subscribers.list_sip_endpoints("subscriber-uuid")

# Create a SIP endpoint for a subscriber
endpoint = client.fabric.subscribers.create_sip_endpoint(
    "subscriber-uuid",
    username="user1",
    password="secret",
    caller_id="+15551234567",
)

# Get a specific SIP endpoint
endpoint = client.fabric.subscribers.get_sip_endpoint("subscriber-uuid", "endpoint-uuid")

# Update a SIP endpoint (uses PATCH)
client.fabric.subscribers.update_sip_endpoint(
    "subscriber-uuid", "endpoint-uuid",
    caller_id="+15559876543",
)

# Delete a SIP endpoint
client.fabric.subscribers.delete_sip_endpoint("subscriber-uuid", "endpoint-uuid")
```

## cXML Applications

cXML applications support list/get/update/delete but not create:

```python
apps = client.fabric.cxml_applications.list()
app = client.fabric.cxml_applications.get("app-uuid")
client.fabric.cxml_applications.update("app-uuid", voice_url="https://example.com/voice")
client.fabric.cxml_applications.delete("app-uuid")

# This raises NotImplementedError:
# client.fabric.cxml_applications.create(...)
```

## Generic Resources

Operate on any resource type by ID:

```python
# List all resources across all types
all_resources = client.fabric.resources.list()

# Get any resource by ID
resource = client.fabric.resources.get("resource-uuid")

# Delete any resource
client.fabric.resources.delete("resource-uuid")

# List addresses for any resource
addresses = client.fabric.resources.list_addresses("resource-uuid")

# Assign a resource to a phone route
client.fabric.resources.assign_phone_route("resource-uuid", phone_route_id="route-uuid")

# Assign a resource as a domain application handler
client.fabric.resources.assign_domain_application("resource-uuid", domain_application_id="da-uuid")
```

## Fabric Addresses

Read-only access to all fabric addresses:

```python
# List all addresses (filter by type or display_name)
addresses = client.fabric.addresses.list(type="room")

# Get a specific address
address = client.fabric.addresses.get("address-uuid")
```

## Tokens

Create tokens for subscribers, guests, invites, and embeds:

```python
# Subscriber token
token = client.fabric.tokens.create_subscriber_token(
    reference="user@example.com",
    password="secret",
)

# Refresh a subscriber token
refreshed = client.fabric.tokens.refresh_subscriber_token(
    refresh_token="existing-refresh-token",
)

# Guest token
token = client.fabric.tokens.create_guest_token(
    allowed_addresses=["address-uuid-1", "address-uuid-2"],
    expire_at="2025-12-31T23:59:59Z",
)

# Subscriber invite token
token = client.fabric.tokens.create_invite_token(
    address_id="address-uuid",
    expires_at="2025-12-31T23:59:59Z",
)

# Click-to-call embed token
token = client.fabric.tokens.create_embed_token(token="embed-source-token")
```
