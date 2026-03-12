"""Example: Deploy a voice application end-to-end with SWML and call flows.

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
    # 1. Create a SWML script
    print("Creating SWML script...")
    swml = client.fabric.swml_scripts.create(
        name="Greeting Script",
        contents={"sections": {"main": [{"play": {"url": "say:Hello from SignalWire"}}]}},
    )
    swml_id = swml["id"]
    print(f"  Created SWML script: {swml_id}")

    # 2. List SWML scripts to confirm
    print("\nListing SWML scripts...")
    scripts = client.fabric.swml_scripts.list()
    for s in scripts.get("data", []):
        print(f"  - {s['id']}: {s.get('display_name', 'unnamed')}")

    # 3. Create a call flow
    print("\nCreating call flow...")
    flow = client.fabric.call_flows.create(title="Main IVR Flow")
    flow_id = flow["id"]
    print(f"  Created call flow: {flow_id}")

    # 4. Deploy a version of the call flow
    print("\nDeploying call flow version...")
    try:
        version = client.fabric.call_flows.deploy_version(flow_id, label="v1")
        print(f"  Deployed version: {version}")
    except SignalWireRestError as e:
        print(f"  Deploy failed (expected in demo): {e.status_code}")

    # 5. List call flow versions
    print("\nListing call flow versions...")
    try:
        versions = client.fabric.call_flows.list_versions(flow_id)
        for v in versions.get("data", []):
            print(f"  - Version: {v.get('label', v.get('id', 'unknown'))}")
    except SignalWireRestError as e:
        print(f"  List versions failed: {e.status_code}")

    # 6. List addresses for the call flow
    print("\nListing call flow addresses...")
    try:
        addrs = client.fabric.call_flows.list_addresses(flow_id)
        for a in addrs.get("data", []):
            print(f"  - {a.get('display_name', a.get('id', 'unknown'))}")
    except SignalWireRestError as e:
        print(f"  List addresses failed: {e.status_code}")

    # 7. Create a SWML webhook as an alternative approach
    print("\nCreating SWML webhook...")
    webhook = client.fabric.swml_webhooks.create(
        name="External Handler",
        primary_request_url="https://example.com/swml-handler",
    )
    webhook_id = webhook["id"]
    print(f"  Created webhook: {webhook_id}")

    # 8. Clean up
    print("\nCleaning up...")
    client.fabric.swml_webhooks.delete(webhook_id)
    print(f"  Deleted webhook {webhook_id}")
    client.fabric.call_flows.delete(flow_id)
    print(f"  Deleted call flow {flow_id}")
    client.fabric.swml_scripts.delete(swml_id)
    print(f"  Deleted SWML script {swml_id}")


if __name__ == "__main__":
    main()
