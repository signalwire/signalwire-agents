"""Example: Twilio-compatible LAML migration — phone numbers, messaging, calls,
conferences, queues, recordings, project tokens, PubSub/Chat, and logs.

Set these env vars (or pass them directly to SignalWireClient):
  SIGNALWIRE_PROJECT_ID   - your SignalWire project ID
  SIGNALWIRE_API_TOKEN    - your SignalWire API token
  SIGNALWIRE_SPACE        - your SignalWire space (e.g. example.signalwire.com)

For full HTTP debug output:
  SIGNALWIRE_LOG_LEVEL=debug
"""

from signalwire_agents.rest import SignalWireClient, SignalWireRestError

client = SignalWireClient()


def safe(label, fn):
    """Execute a call, handling expected demo failures."""
    try:
        result = fn()
        print(f"  {label}: OK")
        return result
    except SignalWireRestError as e:
        print(f"  {label}: failed ({e.status_code})")
        return None


def main():
    # --- Compat Phone Numbers ---

    # 1. Search available numbers
    print("Searching compat phone numbers...")
    safe("Search local", lambda: client.compat.phone_numbers.search_local(
        "US", AreaCode="512",
    ))
    safe("Search toll-free", lambda: client.compat.phone_numbers.search_toll_free("US"))
    safe("List countries", lambda: client.compat.phone_numbers.list_available_countries())

    # 2. Purchase a number (demo — will fail without valid number)
    print("\nPurchasing compat number...")
    num = safe("Purchase", lambda: client.compat.phone_numbers.purchase(
        PhoneNumber="+15125551234",
    ))
    num_sid = num["sid"] if num else None

    # --- LaML Bin & Application ---

    # 3. Create a LaML bin and application
    print("\nCreating LaML resources...")
    laml = safe("LaML bin", lambda: client.compat.laml_bins.create(
        Name="Hold Music",
        Contents="<Response><Say>Please hold.</Say></Response>",
    ))
    laml_sid = laml["sid"] if laml else None

    app = safe("Application", lambda: client.compat.applications.create(
        FriendlyName="Demo App",
        VoiceUrl="https://example.com/voice",
    ))
    app_sid = app["sid"] if app else None

    # --- Messaging ---

    # 4. Send an SMS (demo — requires valid numbers)
    print("\nMessaging operations...")
    msg = safe("Send SMS", lambda: client.compat.messages.create(
        From="+15559876543", To="+15551234567", Body="Hello from SignalWire!",
    ))
    msg_sid = msg["sid"] if msg else None

    # 5. List and get messages
    safe("List messages", lambda: client.compat.messages.list())
    if msg_sid:
        safe("Get message", lambda: client.compat.messages.get(msg_sid))
        safe("List media", lambda: client.compat.messages.list_media(msg_sid))

    # --- Calls ---

    # 6. Outbound call with recording and streaming
    print("\nCall operations...")
    call = safe("Create call", lambda: client.compat.calls.create(
        From="+15559876543", To="+15551234567",
        Url="https://example.com/voice-handler",
    ))
    call_sid = call["sid"] if call else None

    if call_sid:
        safe("Start recording", lambda: client.compat.calls.start_recording(call_sid))
        safe("Start stream", lambda: client.compat.calls.start_stream(
            call_sid, Url="wss://example.com/stream",
        ))

    # --- Conferences ---

    # 7. Conference operations
    print("\nConference operations...")
    confs = safe("List conferences", lambda: client.compat.conferences.list())
    conf_sid = None
    if confs and confs.get("data"):
        conf_sid = confs["data"][0].get("sid")

    if conf_sid:
        safe("Get conference", lambda: client.compat.conferences.get(conf_sid))
        safe("List participants", lambda: client.compat.conferences.list_participants(conf_sid))
        safe("List conf recordings", lambda: client.compat.conferences.list_recordings(conf_sid))

    # --- Queues ---

    # 8. Queue operations (via non-compat API for reliable creation)
    print("\nQueue operations...")
    queue = safe("Create queue", lambda: client.compat.queues.create(
        FriendlyName="compat-support-queue",
    ))
    q_sid = queue["sid"] if queue else None

    if q_sid:
        safe("List queue members", lambda: client.compat.queues.list_members(q_sid))

    # --- Recordings & Transcriptions ---

    # 9. Recordings and transcriptions
    print("\nRecordings and transcriptions...")
    recs = safe("List recordings", lambda: client.compat.recordings.list())
    first_rec_sid = None
    if recs and recs.get("data"):
        first_rec_sid = recs["data"][0].get("sid")
    if first_rec_sid:
        safe("Get recording", lambda: client.compat.recordings.get(first_rec_sid))

    trans = safe("List transcriptions", lambda: client.compat.transcriptions.list())
    first_trans_sid = None
    if trans and trans.get("data"):
        first_trans_sid = trans["data"][0].get("sid")
    if first_trans_sid:
        safe("Get transcription", lambda: client.compat.transcriptions.get(first_trans_sid))

    # --- Faxes ---

    # 10. Fax operations
    print("\nFax operations...")
    fax = safe("Create fax", lambda: client.compat.faxes.create(
        From="+15559876543", To="+15551234567",
        MediaUrl="https://example.com/document.pdf",
    ))
    fax_sid = fax["sid"] if fax else None
    if fax_sid:
        safe("Get fax", lambda: client.compat.faxes.get(fax_sid))

    # --- Compat Accounts & Tokens ---

    # 11. Accounts and compat tokens
    print("\nAccounts and compat tokens...")
    safe("List accounts", lambda: client.compat.accounts.list())
    compat_token = safe("Create compat token", lambda: client.compat.tokens.create(
        name="demo-token",
    ))
    if compat_token and compat_token.get("id"):
        safe("Delete compat token", lambda: client.compat.tokens.delete(compat_token["id"]))

    # --- Project Tokens ---

    # 12. Project token management
    print("\nProject tokens...")
    proj_token = safe("Create project token", lambda: client.project.tokens.create(
        name="CI Token",
        permissions=["calling", "messaging", "video"],
    ))
    if proj_token and proj_token.get("id"):
        safe("Update project token", lambda: client.project.tokens.update(
            proj_token["id"], name="CI Token (updated)",
        ))
        safe("Delete project token", lambda: client.project.tokens.delete(proj_token["id"]))

    # --- PubSub & Chat Tokens ---

    # 13. PubSub and Chat tokens
    print("\nPubSub and Chat tokens...")
    safe("PubSub token", lambda: client.pubsub.create_token(
        channels={"notifications": {"read": True, "write": True}},
        ttl=3600,
    ))
    safe("Chat token", lambda: client.chat.create_token(
        member_id="user-alice",
        channels={"general": {"read": True, "write": True}},
        ttl=3600,
    ))

    # --- Logs ---

    # 14. Log queries
    print("\nQuerying logs...")
    safe("Message logs", lambda: client.logs.messages.list())
    safe("Voice logs", lambda: client.logs.voice.list())
    safe("Fax logs", lambda: client.logs.fax.list())
    safe("Conference logs", lambda: client.logs.conferences.list())

    # Get specific log entries with events
    voice_logs = safe("Voice log list", lambda: client.logs.voice.list()) or {}
    first_voice = (voice_logs.get("data") or [{}])[0]
    if first_voice.get("id"):
        safe("Voice log detail", lambda: client.logs.voice.get(first_voice["id"]))
        safe("Voice log events", lambda: client.logs.voice.list_events(first_voice["id"]))

    # --- Clean up ---

    print("\nCleaning up...")
    if q_sid:
        safe("Delete queue", lambda: client.compat.queues.delete(q_sid))
    if app_sid:
        safe("Delete application", lambda: client.compat.applications.delete(app_sid))
    if laml_sid:
        safe("Delete LaML bin", lambda: client.compat.laml_bins.delete(laml_sid))
    if num_sid:
        safe("Delete number", lambda: client.compat.phone_numbers.delete(num_sid))


if __name__ == "__main__":
    main()
