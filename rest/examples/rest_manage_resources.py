"""Example: Create an AI agent, assign a phone number, and place a test call.

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
    # 1. Create an AI agent
    print("Creating AI agent...")
    agent = client.fabric.ai_agents.create(
        name="Demo Support Bot",
        prompt={"text": "You are a friendly support agent for Acme Corp."},
    )
    agent_id = agent["id"]
    print(f"  Created agent: {agent_id}")

    # 2. List all AI agents
    print("\nListing AI agents...")
    agents = client.fabric.ai_agents.list()
    for a in agents.get("data", []):
        print(f"  - {a['id']}: {a.get('name', 'unnamed')}")

    # 3. Search for a phone number
    print("\nSearching for available phone numbers...")
    available = client.phone_numbers.search(area_code="512", max_results=3)
    for num in available.get("data", []):
        print(f"  - {num.get('e164', num.get('number', 'unknown'))}")

    # 4. Place a test call (requires valid numbers)
    print("\nPlacing a test call...")
    try:
        result = client.calling.dial(
            from_="+15559876543",
            to="+15551234567",
            url="https://example.com/call-handler",
        )
        print(f"  Call initiated: {result}")
    except SignalWireRestError as e:
        print(f"  Call failed (expected in demo): {e.status_code}")

    # 5. Clean up: delete the agent
    print(f"\nDeleting agent {agent_id}...")
    client.fabric.ai_agents.delete(agent_id)
    print("  Deleted.")


if __name__ == "__main__":
    main()
