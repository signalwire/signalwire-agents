"""Example: Upload a document to Datasphere and run a semantic search.

Set these env vars (or pass them directly to SignalWireClient):
  SIGNALWIRE_PROJECT_ID   - your SignalWire project ID
  SIGNALWIRE_API_TOKEN    - your SignalWire API token
  SIGNALWIRE_SPACE        - your SignalWire space (e.g. example.signalwire.com)

For full HTTP debug output:
  SIGNALWIRE_LOG_LEVEL=debug
"""

import time

from signalwire_agents.rest import SignalWireClient, SignalWireRestError

client = SignalWireClient()


def main():
    # 1. Upload a document (a publicly accessible text file)
    print("Uploading document to Datasphere...")
    doc = client.datasphere.documents.create(
        url="https://filesamples.com/samples/document/txt/sample3.txt",
        tags=["support", "demo"],
    )
    doc_id = doc["id"]
    print(f"  Document created: {doc_id} (status: {doc.get('status')})")

    # 2. Wait for vectorization to complete
    print("\nWaiting for document to be vectorized...")
    for i in range(30):
        time.sleep(2)
        doc_status = client.datasphere.documents.get(doc_id)
        status = doc_status.get("status", "unknown")
        print(f"  Poll {i + 1}: status={status}")
        if status == "completed":
            print(f"  Vectorized! Chunks: {doc_status.get('number_of_chunks', 0)}")
            break
        if status in ("error", "failed"):
            print(f"  Document processing failed: {status}")
            client.datasphere.documents.delete(doc_id)
            return
    else:
        print("  Timed out waiting for vectorization.")
        client.datasphere.documents.delete(doc_id)
        return

    # 3. List chunks
    print(f"\nListing chunks for document {doc_id}...")
    chunks = client.datasphere.documents.list_chunks(doc_id)
    for chunk in chunks.get("data", [])[:5]:
        print(f"  - Chunk {chunk['id']}: {chunk.get('content', '')[:80]}...")

    # 4. Semantic search across all documents
    print("\nSearching Datasphere...")
    results = client.datasphere.documents.search(
        query_string="lorem ipsum dolor sit amet",
        count=3,
    )
    for chunk in results.get("chunks", []):
        print(f"  - {chunk.get('text', '')[:100]}...")

    # 5. Clean up
    print(f"\nDeleting document {doc_id}...")
    client.datasphere.documents.delete(doc_id)
    print("  Deleted.")


if __name__ == "__main__":
    main()
