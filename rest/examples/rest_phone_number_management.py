"""Example: Full phone number inventory lifecycle.

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
    # 1. Search for available phone numbers
    print("Searching available numbers...")
    available = client.phone_numbers.search(area_code="512", max_results=3)
    for num in available.get("data", []):
        print(f"  - {num.get('e164', num.get('number', 'unknown'))}")

    # 2. Purchase a number
    print("\nPurchasing a phone number...")
    try:
        first = (available.get("data") or [{}])[0]
        number = client.phone_numbers.create(number=first.get("e164", "+15125551234"))
        num_id = number["id"]
        print(f"  Purchased: {num_id}")
    except SignalWireRestError as e:
        print(f"  Purchase failed (expected in demo): {e.status_code}")
        num_id = None

    # 3. List and get owned numbers
    print("\nListing owned numbers...")
    owned = client.phone_numbers.list()
    for n in owned.get("data", [])[:5]:
        print(f"  - {n.get('number', 'unknown')} ({n['id']})")

    if num_id:
        detail = client.phone_numbers.get(num_id)
        print(f"  Detail: {detail.get('number', 'N/A')}")

    # 4. Update a number
    if num_id:
        print(f"\nUpdating number {num_id}...")
        client.phone_numbers.update(num_id, name="Main Line")
        print("  Updated name to 'Main Line'")

    # 5. Create a number group
    print("\nCreating number group...")
    try:
        group = client.number_groups.create(name="Sales Pool")
        group_id = group["id"]
        print(f"  Created group: {group_id}")
    except SignalWireRestError as e:
        print(f"  Group creation failed (expected in demo): {e.status_code}")
        group_id = None

    # 6. Add a membership and list memberships
    if group_id and num_id:
        print("\nAdding number to group...")
        try:
            membership = client.number_groups.add_membership(
                group_id, phone_number_id=num_id,
            )
            mem_id = membership.get("id")
            print(f"  Membership: {mem_id}")

            memberships = client.number_groups.list_memberships(group_id)
            for m in memberships.get("data", []):
                print(f"  - Member: {m.get('id', 'unknown')}")
        except SignalWireRestError as e:
            print(f"  Membership failed (expected in demo): {e.status_code}")
            mem_id = None

    # 7. Lookup carrier info
    print("\nLooking up carrier info...")
    try:
        info = client.lookup.phone_number("+15125551234")
        print(f"  Carrier: {info.get('carrier', {}).get('name', 'unknown')}")
    except SignalWireRestError as e:
        print(f"  Lookup failed (expected in demo): {e.status_code}")

    # 8. Create a verified caller
    print("\nCreating verified caller...")
    try:
        caller = client.verified_callers.create(phone_number="+15125559999")
        caller_id = caller["id"]
        print(f"  Created verified caller: {caller_id}")
        # Submit verification code
        client.verified_callers.submit_verification(caller_id, verification_code="123456")
        print("  Verification code submitted")
    except SignalWireRestError as e:
        print(f"  Verified caller failed (expected in demo): {e.status_code}")
        caller_id = None

    # 9. Get and update SIP profile
    print("\nGetting SIP profile...")
    try:
        profile = client.sip_profile.get()
        print(f"  SIP profile: {profile}")
        client.sip_profile.update(default_codecs=["PCMU", "PCMA"])
        print("  Updated SIP codecs")
    except SignalWireRestError as e:
        print(f"  SIP profile failed (expected in demo): {e.status_code}")

    # 10. List short codes
    print("\nListing short codes...")
    try:
        codes = client.short_codes.list()
        for sc in codes.get("data", []):
            print(f"  - {sc.get('short_code', 'unknown')}")
    except SignalWireRestError as e:
        print(f"  Short codes failed (expected in demo): {e.status_code}")

    # 11. Create an address
    print("\nCreating address...")
    try:
        addr = client.addresses.create(
            friendly_name="HQ Address",
            street="123 Main St",
            city="Austin",
            region="TX",
            postal_code="78701",
            iso_country="US",
        )
        addr_id = addr["id"]
        print(f"  Created address: {addr_id}")
    except SignalWireRestError as e:
        print(f"  Address creation failed (expected in demo): {e.status_code}")
        addr_id = None

    # 12. Clean up
    print("\nCleaning up...")
    if addr_id:
        client.addresses.delete(addr_id)
        print(f"  Deleted address {addr_id}")
    if caller_id:
        try:
            client.verified_callers.delete(caller_id)
            print(f"  Deleted verified caller {caller_id}")
        except SignalWireRestError as e:
            print(f"  Verified caller delete failed: {e.status_code}")
    if group_id:
        client.number_groups.delete(group_id)
        print(f"  Deleted number group {group_id}")
    if num_id:
        try:
            client.phone_numbers.delete(num_id)
            print(f"  Released number {num_id}")
        except SignalWireRestError as e:
            print(f"  Release number failed (recently purchased): {e.status_code}")


if __name__ == "__main__":
    main()
