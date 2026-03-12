"""Example: Call queues, recording review, and MFA verification.

Set these env vars (or pass them directly to SignalWireClient):
  SIGNALWIRE_PROJECT_ID   - your SignalWire project ID
  SIGNALWIRE_API_TOKEN    - your SignalWire API token
  SIGNALWIRE_SPACE        - your SignalWire space (e.g. example.signalwire.com)

For full HTTP debug output:
  SIGNALWIRE_LOG_LEVEL=debug
"""

from signalwire_agents.rest import SignalWireClient, SignalWireRestError

client = SignalWireClient()


def main():
    # --- Queues ---

    # 1. Create a queue
    print("Creating call queue...")
    try:
        queue = client.queues.create(name="Support Queue", max_size=50)
        queue_id = queue["id"]
        print(f"  Created queue: {queue_id}")
    except SignalWireRestError as e:
        print(f"  Queue creation failed (expected in demo): {e.status_code}")
        queue_id = None

    # 2. List queues
    print("\nListing queues...")
    queues = client.queues.list()
    for q in queues.get("data", []):
        print(f"  - {q['id']}: {q.get('friendly_name', q.get('name', 'unnamed'))}")

    # 3. Get and update queue
    if queue_id:
        detail = client.queues.get(queue_id)
        print(f"\nQueue detail: {detail.get('friendly_name', 'N/A')} (max: {detail.get('max_size', 'N/A')})")

        client.queues.update(queue_id, name="Priority Support Queue")
        print("  Updated queue name")

    # 4. Queue members
    if queue_id:
        print("\nListing queue members...")
        try:
            members = client.queues.list_members(queue_id)
            for m in members.get("data", []):
                print(f"  - Member: {m.get('call_id', m.get('id', 'unknown'))}")

            next_member = client.queues.get_next_member(queue_id)
            print(f"  Next member: {next_member}")
        except SignalWireRestError as e:
            print(f"  Member ops failed (expected if queue empty): {e.status_code}")

    # --- Recordings ---

    # 5. List recordings
    print("\nListing recordings...")
    recordings = client.recordings.list()
    for r in recordings.get("data", [])[:5]:
        print(f"  - {r['id']}: {r.get('duration', 'N/A')}s")

    # 6. Get recording details
    first_rec = (recordings.get("data") or [{}])[0]
    if first_rec.get("id"):
        rec_detail = client.recordings.get(first_rec["id"])
        print(f"  Recording: {rec_detail.get('duration', 'N/A')}s, {rec_detail.get('format', 'N/A')}")

    # --- MFA ---

    # 7. Send MFA via SMS
    print("\nSending MFA SMS code...")
    try:
        sms_result = client.mfa.sms(
            to="+15551234567",
            from_="+15559876543",
            message="Your code is {{code}}",
            token_length=6,
        )
        request_id = sms_result.get("id", sms_result.get("request_id"))
        print(f"  MFA SMS sent: {request_id}")
    except SignalWireRestError as e:
        print(f"  MFA SMS failed (expected in demo): {e.status_code}")
        request_id = None

    # 8. Send MFA via voice call
    print("\nSending MFA voice code...")
    try:
        voice_result = client.mfa.call(
            to="+15551234567",
            from_="+15559876543",
            message="Your verification code is {{code}}",
            token_length=6,
        )
        print(f"  MFA call sent: {voice_result.get('id', voice_result.get('request_id'))}")
    except SignalWireRestError as e:
        print(f"  MFA call failed (expected in demo): {e.status_code}")

    # 9. Verify MFA token
    if request_id:
        print("\nVerifying MFA token...")
        try:
            verify = client.mfa.verify(request_id, token="123456")
            print(f"  Verification result: {verify}")
        except SignalWireRestError as e:
            print(f"  Verify failed (expected in demo): {e.status_code}")

    # 10. Clean up
    print("\nCleaning up...")
    if queue_id:
        client.queues.delete(queue_id)
        print(f"  Deleted queue {queue_id}")


if __name__ == "__main__":
    main()
