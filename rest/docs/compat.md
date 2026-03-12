# Compatibility API

The Compatibility API provides a Twilio-compatible LAML surface at `/api/laml/2010-04-01`. All paths are scoped under `/Accounts/{AccountSid}`, where AccountSid is your project ID.

## Sub-Resources

| Attribute | Description |
|-----------|-------------|
| `compat.accounts` | Account/subproject management |
| `compat.calls` | Call management + recordings + streams |
| `compat.messages` | SMS/MMS management + media |
| `compat.faxes` | Fax management + media |
| `compat.conferences` | Conference management + participants + recordings + streams |
| `compat.phone_numbers` | Incoming + available phone numbers |
| `compat.applications` | Application management |
| `compat.laml_bins` | cXML/LaML script management |
| `compat.queues` | Queue management + members |
| `compat.recordings` | Recording management |
| `compat.transcriptions` | Transcription management |
| `compat.tokens` | API token management |

## Accounts

```python
# List accounts/subprojects
accounts = client.compat.accounts.list()

# Create a subproject
sub = client.compat.accounts.create(FriendlyName="My Subproject")

# Get/update an account
account = client.compat.accounts.get("AC-sid")
client.compat.accounts.update("AC-sid", FriendlyName="Updated")
```

## Calls

```python
# List calls
calls = client.compat.calls.list(From="+15551234567")

# Create a call
call = client.compat.calls.create(
    To="+15552222222",
    From="+15551111111",
    Url="https://example.com/twiml",
)

# Get / update / delete
call = client.compat.calls.get("CA-sid")
client.compat.calls.update("CA-sid", Status="completed")
client.compat.calls.delete("CA-sid")

# Start/update recording on a call
client.compat.calls.start_recording("CA-sid", channels="dual")
client.compat.calls.update_recording("CA-sid", "RE-sid", Status="paused")

# Start/stop stream on a call
client.compat.calls.start_stream("CA-sid", Url="wss://example.com/stream")
client.compat.calls.stop_stream("CA-sid", "ST-sid")
```

## Messages

```python
# Send an SMS
msg = client.compat.messages.create(
    To="+15552222222",
    From="+15551111111",
    Body="Hello from SignalWire!",
)

# List / get / update / delete
messages = client.compat.messages.list()
msg = client.compat.messages.get("SM-sid")
client.compat.messages.update("SM-sid", Body="")  # redact
client.compat.messages.delete("SM-sid")

# Media sub-resources
media = client.compat.messages.list_media("SM-sid")
item = client.compat.messages.get_media("SM-sid", "ME-sid")
client.compat.messages.delete_media("SM-sid", "ME-sid")
```

## Faxes

```python
# Send a fax
fax = client.compat.faxes.create(MediaUrl="https://example.com/doc.pdf", To="+15552222222", From="+15551111111")

# List / get / cancel / delete
faxes = client.compat.faxes.list()
fax = client.compat.faxes.get("FX-sid")
client.compat.faxes.update("FX-sid", Status="canceled")
client.compat.faxes.delete("FX-sid")

# Media sub-resources
media = client.compat.faxes.list_media("FX-sid")
item = client.compat.faxes.get_media("FX-sid", "ME-sid")
client.compat.faxes.delete_media("FX-sid", "ME-sid")
```

## Conferences

```python
# List / get / update
conferences = client.compat.conferences.list()
conf = client.compat.conferences.get("CF-sid")
client.compat.conferences.update("CF-sid", Status="completed")

# Participants
participants = client.compat.conferences.list_participants("CF-sid")
p = client.compat.conferences.get_participant("CF-sid", "CA-sid")
client.compat.conferences.update_participant("CF-sid", "CA-sid", Muted=True)
client.compat.conferences.remove_participant("CF-sid", "CA-sid")

# Conference recordings
recs = client.compat.conferences.list_recordings("CF-sid")
rec = client.compat.conferences.get_recording("CF-sid", "RE-sid")
client.compat.conferences.update_recording("CF-sid", "RE-sid", Status="stopped")
client.compat.conferences.delete_recording("CF-sid", "RE-sid")

# Conference streams
client.compat.conferences.start_stream("CF-sid", Url="wss://example.com/stream")
client.compat.conferences.stop_stream("CF-sid", "ST-sid")
```

## Phone Numbers

```python
# List purchased numbers
numbers = client.compat.phone_numbers.list()

# Search available numbers
local = client.compat.phone_numbers.search_local("US", AreaCode="512")
toll_free = client.compat.phone_numbers.search_toll_free("US")
countries = client.compat.phone_numbers.list_available_countries()

# Purchase / get / update / release
num = client.compat.phone_numbers.purchase(PhoneNumber="+15551234567")
num = client.compat.phone_numbers.get("PN-sid")
client.compat.phone_numbers.update("PN-sid", VoiceUrl="https://example.com/voice")
client.compat.phone_numbers.delete("PN-sid")

# Import external number
client.compat.phone_numbers.import_number(PhoneNumber="+15559999999")
```

## Applications

```python
apps = client.compat.applications.list()
app = client.compat.applications.create(FriendlyName="My App", VoiceUrl="https://example.com/voice")
app = client.compat.applications.get("AP-sid")
client.compat.applications.update("AP-sid", VoiceUrl="https://example.com/new-voice")
client.compat.applications.delete("AP-sid")
```

## LaML Bins (cXML Scripts)

```python
bins = client.compat.laml_bins.list()
b = client.compat.laml_bins.create(Name="Greeting", Contents="<Response><Say>Hello</Say></Response>")
b = client.compat.laml_bins.get("LB-sid")
client.compat.laml_bins.update("LB-sid", Contents="<Response><Say>Updated</Say></Response>")
client.compat.laml_bins.delete("LB-sid")
```

## Queues

```python
queues = client.compat.queues.list()
q = client.compat.queues.create(FriendlyName="Support", MaxSize=100)
q = client.compat.queues.get("QU-sid")
client.compat.queues.update("QU-sid", MaxSize=200)
client.compat.queues.delete("QU-sid")

# Members
members = client.compat.queues.list_members("QU-sid")
member = client.compat.queues.get_member("QU-sid", "CA-sid")
client.compat.queues.dequeue_member("QU-sid", "CA-sid", Url="https://example.com/dequeue")
```

## Recordings & Transcriptions

```python
# Recordings
recs = client.compat.recordings.list()
rec = client.compat.recordings.get("RE-sid")
client.compat.recordings.delete("RE-sid")

# Transcriptions
txns = client.compat.transcriptions.list()
txn = client.compat.transcriptions.get("TR-sid")
client.compat.transcriptions.delete("TR-sid")
```

## Tokens

```python
token = client.compat.tokens.create(name="my-token", permissions=["calling", "messaging"])
client.compat.tokens.update("token-id", name="renamed")
client.compat.tokens.delete("token-id")
```
