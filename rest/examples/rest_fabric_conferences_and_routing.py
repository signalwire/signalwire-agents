"""Example: Conference infrastructure, cXML resources, generic routing, and tokens.

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
    # 1. Create a conference room
    print("Creating conference room...")
    room = client.fabric.conference_rooms.create(name="team-standup")
    room_id = room["id"]
    print(f"  Created conference room: {room_id}")

    # 2. List conference room addresses
    print("\nListing conference room addresses...")
    try:
        addrs = client.fabric.conference_rooms.list_addresses(room_id)
        for a in addrs.get("data", []):
            print(f"  - {a.get('display_name', a.get('id', 'unknown'))}")
    except SignalWireRestError as e:
        print(f"  List addresses failed: {e.status_code}")

    # 3. Create a cXML script
    print("\nCreating cXML script...")
    cxml = client.fabric.cxml_scripts.create(
        name="Hold Music Script",
        contents="<Response><Say>Please hold.</Say><Play>https://example.com/hold.mp3</Play></Response>",
    )
    cxml_id = cxml["id"]
    print(f"  Created cXML script: {cxml_id}")

    # 4. Create a cXML webhook
    print("\nCreating cXML webhook...")
    cxml_wh = client.fabric.cxml_webhooks.create(
        name="External cXML Handler",
        primary_request_url="https://example.com/cxml-handler",
    )
    cxml_wh_id = cxml_wh["id"]
    print(f"  Created cXML webhook: {cxml_wh_id}")

    # 5. Create a relay application
    print("\nCreating relay application...")
    relay_app = client.fabric.relay_applications.create(
        name="Inbound Handler",
        topic="office",
    )
    relay_id = relay_app["id"]
    print(f"  Created relay application: {relay_id}")

    # 6. Generic resources: list all
    print("\nListing all fabric resources...")
    resources = client.fabric.resources.list()
    for r in resources.get("data", [])[:5]:
        print(f"  - {r.get('type', 'unknown')}: {r.get('display_name', r.get('id', 'unknown'))}")

    # 7. Get a specific generic resource
    first = (resources.get("data") or [{}])[0]
    if first.get("id"):
        detail = client.fabric.resources.get(first["id"])
        print(f"  Resource detail: {detail.get('display_name', 'N/A')} ({detail.get('type', 'N/A')})")

    # 8. Assign a phone route to a resource (demo)
    print("\nAssigning phone route (demo)...")
    try:
        client.fabric.resources.assign_phone_route(
            relay_id, phone_number="+15551234567",
        )
        print("  Phone route assigned")
    except SignalWireRestError as e:
        print(f"  Route assignment failed (expected in demo): {e.status_code}")

    # 9. Assign a domain application (demo)
    print("\nAssigning domain application (demo)...")
    try:
        client.fabric.resources.assign_domain_application(
            relay_id, domain="app.example.com",
        )
        print("  Domain application assigned")
    except SignalWireRestError as e:
        print(f"  Domain assignment failed (expected in demo): {e.status_code}")

    # 10. Generate tokens
    print("\nGenerating tokens...")
    try:
        guest = client.fabric.tokens.create_guest_token(resource_id=relay_id)
        print(f"  Guest token: {str(guest.get('token', ''))[:40]}...")
    except SignalWireRestError as e:
        print(f"  Guest token failed (expected in demo): {e.status_code}")

    try:
        invite = client.fabric.tokens.create_invite_token(resource_id=relay_id)
        print(f"  Invite token: {str(invite.get('token', ''))[:40]}...")
    except SignalWireRestError as e:
        print(f"  Invite token failed (expected in demo): {e.status_code}")

    try:
        embed = client.fabric.tokens.create_embed_token(resource_id=relay_id)
        print(f"  Embed token: {str(embed.get('token', ''))[:40]}...")
    except SignalWireRestError as e:
        print(f"  Embed token failed (expected in demo): {e.status_code}")

    # 11. Clean up
    print("\nCleaning up...")
    client.fabric.relay_applications.delete(relay_id)
    print(f"  Deleted relay application {relay_id}")
    client.fabric.cxml_webhooks.delete(cxml_wh_id)
    print(f"  Deleted cXML webhook {cxml_wh_id}")
    client.fabric.cxml_scripts.delete(cxml_id)
    print(f"  Deleted cXML script {cxml_id}")
    client.fabric.conference_rooms.delete(room_id)
    print(f"  Deleted conference room {room_id}")


if __name__ == "__main__":
    main()
