"""Example: Video rooms for team standup and conference streaming.

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
    # --- Video Rooms ---

    # 1. Create a video room
    print("Creating video room...")
    room = client.video.rooms.create(
        name="daily-standup",
        display_name="Daily Standup",
        max_members=10,
        layout="grid-responsive",
    )
    room_id = room["id"]
    print(f"  Created room: {room_id}")

    # 2. List video rooms
    print("\nListing video rooms...")
    rooms = client.video.rooms.list()
    for r in rooms.get("data", [])[:5]:
        print(f"  - {r['id']}: {r.get('name', 'unnamed')}")

    # 3. Generate a join token
    print("\nGenerating room token...")
    try:
        token = client.video.room_tokens.create(
            room_name="daily-standup",
            user_name="alice",
            permissions=["room.self.audio_mute", "room.self.video_mute"],
        )
        print(f"  Token: {str(token.get('token', ''))[:40]}...")
    except SignalWireRestError as e:
        print(f"  Token failed (expected in demo): {e.status_code}")

    # --- Sessions ---

    # 4. List room sessions
    print("\nListing room sessions...")
    sessions = client.video.room_sessions.list()
    for s in sessions.get("data", [])[:3]:
        print(f"  - Session {s['id']}: {s.get('status', 'unknown')}")

    # 5. Get session details with members, events, recordings
    first_session = (sessions.get("data") or [{}])[0]
    if first_session.get("id"):
        sid = first_session["id"]
        detail = client.video.room_sessions.get(sid)
        print(f"  Session: {detail.get('name', 'N/A')} ({detail.get('status', 'N/A')})")

        members = client.video.room_sessions.list_members(sid)
        print(f"  Members: {len(members.get('data', []))}")

        events = client.video.room_sessions.list_events(sid)
        print(f"  Events: {len(events.get('data', []))}")

        recs = client.video.room_sessions.list_recordings(sid)
        print(f"  Recordings: {len(recs.get('data', []))}")

    # --- Room Recordings ---

    # 6. List and get room recordings
    print("\nListing room recordings...")
    room_recs = client.video.room_recordings.list()
    for rr in room_recs.get("data", [])[:3]:
        print(f"  - Recording {rr['id']}: {rr.get('duration', 'N/A')}s")

    first_rec = (room_recs.get("data") or [{}])[0]
    if first_rec.get("id"):
        rec_detail = client.video.room_recordings.get(first_rec["id"])
        print(f"  Recording detail: {rec_detail.get('duration', 'N/A')}s")

        rec_events = client.video.room_recordings.list_events(first_rec["id"])
        print(f"  Recording events: {len(rec_events.get('data', []))}")

    # --- Video Conferences ---

    # 7. Create a video conference
    print("\nCreating video conference...")
    try:
        conf = client.video.conferences.create(
            name="all-hands-stream",
            display_name="All Hands Meeting",
        )
        conf_id = conf["id"]
        print(f"  Created conference: {conf_id}")
    except SignalWireRestError as e:
        print(f"  Conference creation failed (expected in demo): {e.status_code}")
        conf_id = None

    # 8. List conference tokens
    if conf_id:
        print("\nListing conference tokens...")
        try:
            tokens = client.video.conferences.list_conference_tokens(conf_id)
            for t in tokens.get("data", []):
                print(f"  - Token: {t.get('id', 'unknown')}")
        except SignalWireRestError as e:
            print(f"  Conference tokens failed: {e.status_code}")

    # 9. Create a stream on the conference
    stream_id = None
    if conf_id:
        print("\nCreating stream on conference...")
        try:
            stream = client.video.conferences.create_stream(
                conf_id, url="rtmp://live.example.com/stream-key",
            )
            stream_id = stream["id"]
            print(f"  Created stream: {stream_id}")
        except SignalWireRestError as e:
            print(f"  Stream creation failed (expected in demo): {e.status_code}")

    # 10. Get and update stream
    if stream_id:
        print(f"\nManaging stream {stream_id}...")
        try:
            s_detail = client.video.streams.get(stream_id)
            print(f"  Stream URL: {s_detail.get('url', 'N/A')}")

            client.video.streams.update(stream_id, url="rtmp://backup.example.com/stream-key")
            print("  Stream URL updated")
        except SignalWireRestError as e:
            print(f"  Stream ops failed: {e.status_code}")

    # 11. Clean up
    print("\nCleaning up...")
    if stream_id:
        try:
            client.video.streams.delete(stream_id)
            print(f"  Deleted stream {stream_id}")
        except SignalWireRestError as e:
            print(f"  Stream delete failed: {e.status_code}")
    if conf_id:
        client.video.conferences.delete(conf_id)
        print(f"  Deleted conference {conf_id}")
    client.video.rooms.delete(room_id)
    print(f"  Deleted room {room_id}")


if __name__ == "__main__":
    main()
