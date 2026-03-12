"""Example: Provision a SIP-enabled user on Fabric.

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
    # 1. Create a subscriber
    print("Creating subscriber...")
    subscriber = client.fabric.subscribers.create(
        name="Alice Johnson",
        email="alice@example.com",
    )
    sub_id = subscriber["id"]
    inner_sub_id = subscriber.get("subscriber", {}).get("id", sub_id)
    print(f"  Created subscriber: {sub_id}")

    # 2. Add a SIP endpoint to the subscriber
    print("\nCreating SIP endpoint on subscriber...")
    endpoint = client.fabric.subscribers.create_sip_endpoint(
        sub_id,
        username="alice_sip",
        password="SecurePass123!",
    )
    ep_id = endpoint["id"]
    print(f"  Created SIP endpoint: {ep_id}")

    # 3. List SIP endpoints on the subscriber
    print("\nListing subscriber SIP endpoints...")
    endpoints = client.fabric.subscribers.list_sip_endpoints(sub_id)
    for ep in endpoints.get("data", []):
        print(f"  - {ep['id']}: {ep.get('username', 'unknown')}")

    # 4. Get specific SIP endpoint details
    print(f"\nGetting SIP endpoint {ep_id}...")
    ep_detail = client.fabric.subscribers.get_sip_endpoint(sub_id, ep_id)
    print(f"  Username: {ep_detail.get('username', 'N/A')}")

    # 5. Create a standalone SIP gateway
    print("\nCreating SIP gateway...")
    gateway = client.fabric.sip_gateways.create(
        name="Office PBX Gateway",
        uri="sip:pbx.example.com",
        encryption="required",
        ciphers=["AES_256_CM_HMAC_SHA1_80"],
        codecs=["PCMU", "PCMA"],
    )
    gw_id = gateway["id"]
    print(f"  Created SIP gateway: {gw_id}")

    # 6. List fabric addresses
    print("\nListing fabric addresses...")
    try:
        addresses = client.fabric.addresses.list()
        for addr in addresses.get("data", [])[:5]:
            print(f"  - {addr.get('display_name', addr.get('id', 'unknown'))}")

        # 7. Get a specific fabric address
        first_addr = (addresses.get("data") or [{}])[0]
        if first_addr.get("id"):
            addr_detail = client.fabric.addresses.get(first_addr["id"])
            print(f"  Address detail: {addr_detail.get('display_name', 'N/A')}")
    except SignalWireRestError as e:
        print(f"  Fabric addresses failed: {e.status_code}")

    # 8. Generate a subscriber token
    print("\nGenerating subscriber token...")
    try:
        token = client.fabric.tokens.create_subscriber_token(
            subscriber_id=inner_sub_id,
            reference=inner_sub_id,
        )
        print(f"  Token: {str(token.get('token', ''))[:40]}...")
    except SignalWireRestError as e:
        print(f"  Token generation failed (expected in demo): {e.status_code}")

    # 9. Clean up
    print("\nCleaning up...")
    client.fabric.subscribers.delete_sip_endpoint(sub_id, ep_id)
    print(f"  Deleted SIP endpoint {ep_id}")
    client.fabric.subscribers.delete(sub_id)
    print(f"  Deleted subscriber {sub_id}")
    client.fabric.sip_gateways.delete(gw_id)
    print(f"  Deleted SIP gateway {gw_id}")


if __name__ == "__main__":
    main()
