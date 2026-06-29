import chromadb
from chromadb.config import Settings
from datetime import datetime
import json
import os


class ResidentMemory:
    """
    Persistent vector memory for CareAgent-AU.
    Stores and retrieves per-resident context across sessions
    using ChromaDB as the vector store.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        print("Initialising ResidentMemory...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="resident_records",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Memory store ready — {self.collection.count()} records loaded")

    def save_resident(self, resident_id: str, data: dict):
        """Save or update a resident record in memory."""
        document = f"""
Resident: {data.get('resident_info', '')}
Care Plan: {data.get('care_plan', '')}
Risk Assessment: {data.get('risk_assessment', '')}
Timestamp: {data.get('timestamp', '')}
        """.strip()

        metadata = {
            "resident_id": resident_id,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "agent": data.get("agent", "unknown"),
            "status": data.get("status", "completed")
        }

        # Upsert — update if exists, insert if new
        self.collection.upsert(
            documents=[document],
            metadatas=[metadata],
            ids=[resident_id]
        )
        print(f"✓ Resident {resident_id} saved to memory")

    def get_resident(self, resident_id: str) -> dict:
        """Retrieve a resident record by ID."""
        try:
            result = self.collection.get(ids=[resident_id])
            if result["documents"]:
                return {
                    "resident_id": resident_id,
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
            return {}
        except Exception as e:
            print(f"Resident {resident_id} not found: {e}")
            return {}

    def search_similar(self, query: str, n_results: int = 3) -> list:
        """
        Semantic search across all resident records.
        Used by agents to find relevant past cases.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )

        similar = []
        for i, doc in enumerate(results["documents"][0]):
            similar.append({
                "document": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        return similar

    def list_residents(self) -> list:
        """List all resident IDs in memory."""
        result = self.collection.get()
        return result["ids"]

    def delete_resident(self, resident_id: str):
        """Remove a resident record from memory."""
        self.collection.delete(ids=[resident_id])
        print(f"✓ Resident {resident_id} removed from memory")

    def get_stats(self) -> dict:
        """Return memory store statistics."""
        return {
            "total_residents": self.collection.count(),
            "collection_name": self.collection.name,
        }
