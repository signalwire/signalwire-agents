# All Namespaces

Reference for every namespace beyond Fabric, Calling, and Compat (which have their own pages).

## Phone Numbers

```python
# List your phone numbers
numbers = client.phone_numbers.list()
numbers = client.phone_numbers.list(name="Main")

# Search available numbers to purchase
available = client.phone_numbers.search(area_code="512", number_type="local")

# Purchase a number
number = client.phone_numbers.create(number="+15551234567")

# Get / update / release
number = client.phone_numbers.get("pn-uuid")
client.phone_numbers.update("pn-uuid", name="Support Line")
client.phone_numbers.delete("pn-uuid")
```

## Addresses

```python
addresses = client.addresses.list()
address = client.addresses.create(label="Office", street="123 Main St", city="Austin", state="TX")
address = client.addresses.get("addr-uuid")
client.addresses.delete("addr-uuid")
```

## Queues

```python
queues = client.queues.list()
queue = client.queues.create(name="Support")
queue = client.queues.get("q-uuid")
client.queues.update("q-uuid", name="VIP Support")
client.queues.delete("q-uuid")

# Members
members = client.queues.list_members("q-uuid")
next_member = client.queues.get_next_member("q-uuid")
member = client.queues.get_member("q-uuid", "member-uuid")
```

## Recordings

```python
recordings = client.recordings.list()
recording = client.recordings.get("rec-uuid")
client.recordings.delete("rec-uuid")
```

## Number Groups

```python
groups = client.number_groups.list()
group = client.number_groups.create(name="Marketing")
group = client.number_groups.get("ng-uuid")
client.number_groups.update("ng-uuid", name="Sales")
client.number_groups.delete("ng-uuid")

# Memberships
memberships = client.number_groups.list_memberships("ng-uuid")
client.number_groups.add_membership("ng-uuid", phone_number_id="pn-uuid")
membership = client.number_groups.get_membership("mem-uuid")
client.number_groups.delete_membership("mem-uuid")
```

## Verified Caller IDs

```python
callers = client.verified_callers.list()
caller = client.verified_callers.create(phone_number="+15551234567", name="Office")
caller = client.verified_callers.get("vc-uuid")
client.verified_callers.update("vc-uuid", name="Main Office")
client.verified_callers.delete("vc-uuid")

# Verification flow
client.verified_callers.redial_verification("vc-uuid")
client.verified_callers.submit_verification("vc-uuid", code="123456")
```

## SIP Profile

Singleton resource -- no ID needed:

```python
profile = client.sip_profile.get()
client.sip_profile.update(username="myproject", password="newsecret")
```

## Phone Number Lookup

```python
info = client.lookup.phone_number("+15551234567")
info = client.lookup.phone_number("+15551234567", include="carrier,cnam")
```

Note: carrier and CNAM lookups are billable.

## Short Codes

```python
codes = client.short_codes.list()
code = client.short_codes.get("sc-uuid")
client.short_codes.update("sc-uuid", name="Alerts")
```

## Imported Phone Numbers

```python
client.imported_numbers.create(number="+15559999999", carrier="external")
```

## MFA (Multi-Factor Authentication)

```python
# Request a verification code via SMS
result = client.mfa.sms(
    to="+15551234567",
    from_="+15559876543",
    message="Your code is {code}",
)
request_id = result["id"]

# Or via phone call
result = client.mfa.call(
    to="+15551234567",
    from_="+15559876543",
)

# Verify the code
result = client.mfa.verify(request_id, token="123456")
```

## 10DLC Campaign Registry

```python
# Brands
brands = client.registry.brands.list()
brand = client.registry.brands.create(name="My Brand", ein="12-3456789")
brand = client.registry.brands.get("brand-uuid")

# Campaigns under a brand
campaigns = client.registry.brands.list_campaigns("brand-uuid")
campaign = client.registry.brands.create_campaign("brand-uuid", description="Alerts")

# Campaign management
campaign = client.registry.campaigns.get("camp-uuid")
client.registry.campaigns.update("camp-uuid", description="Updated alerts")

# Number assignments
numbers = client.registry.campaigns.list_numbers("camp-uuid")
orders = client.registry.campaigns.list_orders("camp-uuid")
order = client.registry.campaigns.create_order("camp-uuid", phone_number_ids=["pn-1"])
order = client.registry.orders.get("order-uuid")
client.registry.numbers.delete("number-assignment-uuid")
```

## Datasphere

```python
# Documents
docs = client.datasphere.documents.list()
doc = client.datasphere.documents.create(url="https://example.com/doc.pdf", tags=["support"])
doc = client.datasphere.documents.get("doc-uuid")
client.datasphere.documents.update("doc-uuid", tags=["support", "billing"])
client.datasphere.documents.delete("doc-uuid")

# Semantic search
results = client.datasphere.documents.search(
    query_string="How do I reset my password?",
    tags=["support"],
    count=5,
)

# Chunks
chunks = client.datasphere.documents.list_chunks("doc-uuid")
chunk = client.datasphere.documents.get_chunk("doc-uuid", "chunk-uuid")
client.datasphere.documents.delete_chunk("doc-uuid", "chunk-uuid")
```

## Video

```python
# Rooms
rooms = client.video.rooms.list()
room = client.video.rooms.create(name="standup", max_members=10)
room = client.video.rooms.get("room-uuid")
client.video.rooms.update("room-uuid", max_members=20)
client.video.rooms.delete("room-uuid")
client.video.rooms.list_streams("room-uuid")
client.video.rooms.create_stream("room-uuid", url="rtmp://example.com/live")

# Room tokens
token = client.video.room_tokens.create(room_name="standup", user_name="alice")

# Room sessions
sessions = client.video.room_sessions.list(room_name="standup")
session = client.video.room_sessions.get("session-uuid")
events = client.video.room_sessions.list_events("session-uuid")
members = client.video.room_sessions.list_members("session-uuid")
recordings = client.video.room_sessions.list_recordings("session-uuid")

# Room recordings
recs = client.video.room_recordings.list()
rec = client.video.room_recordings.get("rec-uuid")
client.video.room_recordings.delete("rec-uuid")
events = client.video.room_recordings.list_events("rec-uuid")

# Conferences
confs = client.video.conferences.list()
conf = client.video.conferences.create(name="all-hands", quality="720p")
conf = client.video.conferences.get("conf-uuid")
client.video.conferences.update("conf-uuid", quality="1080p")
client.video.conferences.delete("conf-uuid")
tokens = client.video.conferences.list_conference_tokens("conf-uuid")
client.video.conferences.list_streams("conf-uuid")
client.video.conferences.create_stream("conf-uuid", url="rtmp://example.com/live")

# Conference tokens
token = client.video.conference_tokens.get("token-uuid")
client.video.conference_tokens.reset("token-uuid")

# Streams
stream = client.video.streams.get("stream-uuid")
client.video.streams.update("stream-uuid", url="rtmp://example.com/new")
client.video.streams.delete("stream-uuid")
```

## Logs

All log endpoints are read-only.

```python
# Message logs
logs = client.logs.messages.list(include_deleted=True)
log = client.logs.messages.get("log-uuid")

# Voice logs (with events)
logs = client.logs.voice.list()
log = client.logs.voice.get("log-uuid")
events = client.logs.voice.list_events("log-uuid")

# Fax logs
logs = client.logs.fax.list()
log = client.logs.fax.get("log-uuid")

# Conference logs
logs = client.logs.conferences.list()
```

## Project Tokens

```python
token = client.project.tokens.create(
    name="ci-token",
    permissions=["calling", "messaging", "numbers"],
)
client.project.tokens.update("token-uuid", name="renamed-token")
client.project.tokens.delete("token-uuid")
```

## PubSub Tokens

```python
token = client.pubsub.create_token(
    ttl=60,
    channels=[{"name": "updates", "read": True, "write": False}],
    member_id="user-123",
)
```

## Chat Tokens

```python
token = client.chat.create_token(
    ttl=60,
    channels=[{"name": "support", "read": True, "write": True}],
    member_id="user-123",
)
```
