"""Example: 10DLC brand and campaign compliance registration.

WARNING: This example interacts with the real 10DLC registration system.
Brand and campaign registrations may have side effects and costs.
Use with caution in production environments.

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
    # 1. Register a brand
    print("Registering 10DLC brand...")
    try:
        brand = client.registry.brands.create(
            company_name="Acme Corp",
            ein="12-3456789",
            entity_type="PRIVATE_PROFIT",
            vertical="TECHNOLOGY",
            website="https://acme.example.com",
            country="US",
        )
        brand_id = brand["id"]
        print(f"  Registered brand: {brand_id}")
    except SignalWireRestError as e:
        print(f"  Brand registration failed (expected in demo): {e.status_code}")
        brand_id = None

    # 2. List brands
    print("\nListing brands...")
    brands = client.registry.brands.list()
    for b in brands.get("data", []):
        print(f"  - {b['id']}: {b.get('name', 'unnamed')}")
    if not brand_id and brands.get("data"):
        brand_id = brands["data"][0]["id"]

    # 3. Get brand details
    if brand_id:
        detail = client.registry.brands.get(brand_id)
        print(f"\nBrand detail: {detail.get('name', 'N/A')} ({detail.get('state', 'N/A')})")

    # 4. Create a campaign under the brand
    campaign_id = None
    if brand_id:
        print("\nCreating campaign...")
        try:
            campaign = client.registry.brands.create_campaign(
                brand_id,
                use_case="MIXED",
                description="Customer notifications and support messages",
                sample_message="Your order #12345 has shipped.",
            )
            campaign_id = campaign["id"]
            print(f"  Created campaign: {campaign_id}")
        except SignalWireRestError as e:
            print(f"  Campaign creation failed (expected in demo): {e.status_code}")

    # 5. List campaigns for the brand
    if brand_id:
        print("\nListing brand campaigns...")
        campaigns = client.registry.brands.list_campaigns(brand_id)
        for c in campaigns.get("data", []):
            print(f"  - {c['id']}: {c.get('name', 'unknown')}")
            if not campaign_id:
                campaign_id = c["id"]

    # 6. Get and update campaign
    if campaign_id:
        camp_detail = client.registry.campaigns.get(campaign_id)
        print(f"\nCampaign: {camp_detail.get('name', 'N/A')} ({camp_detail.get('state', 'N/A')})")

        try:
            client.registry.campaigns.update(
                campaign_id, description="Updated: customer notifications",
            )
            print("  Campaign description updated")
        except SignalWireRestError as e:
            print(f"  Campaign update failed: {e.status_code}")

    # 7. Create an order to assign numbers
    order_id = None
    if campaign_id:
        print("\nCreating number assignment order...")
        try:
            order = client.registry.campaigns.create_order(
                campaign_id, phone_numbers=["+15125551234"],
            )
            order_id = order["id"]
            print(f"  Created order: {order_id}")
        except SignalWireRestError as e:
            print(f"  Order creation failed (expected in demo): {e.status_code}")

    # 8. Get order status
    if order_id:
        order_detail = client.registry.orders.get(order_id)
        print(f"  Order status: {order_detail.get('status', 'N/A')}")

    # 9. List campaign numbers and orders
    if campaign_id:
        print("\nListing campaign numbers...")
        numbers = client.registry.campaigns.list_numbers(campaign_id)
        for n in numbers.get("data", []):
            print(f"  - {n.get('phone_number', n.get('id', 'unknown'))}")

        orders = client.registry.campaigns.list_orders(campaign_id)
        for o in orders.get("data", []):
            print(f"  - Order {o['id']}: {o.get('status', 'unknown')}")

    # 10. Unassign a number (clean up)
    if campaign_id:
        print("\nUnassigning numbers...")
        nums = client.registry.campaigns.list_numbers(campaign_id)
        for n in nums.get("data", []):
            try:
                client.registry.numbers.delete(n["id"])
                print(f"  Unassigned number {n['id']}")
            except SignalWireRestError as e:
                print(f"  Unassign failed: {e.status_code}")


if __name__ == "__main__":
    main()
