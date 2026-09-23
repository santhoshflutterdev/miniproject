import threading
from typing import Dict, List, Any, Optional

class LocalDatabase:
    """
    In-Memory / Local thread-safe database fallback when Firebase Firestore is unconfigured.
    Provides collection-based CRUD interface identical to Firestore collections.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.collections: Dict[str, Dict[str, Dict[str, Any]]] = {
            "users": {},
            "tasks": {},
            "gpu_nodes": {},
            "scheduling_results": {},
            "metrics": {},
            "predictions": {},
            "simulation_runs": {},
            "isolation_violations": {}
        }

    def set_document(self, collection_name: str, doc_id: str, data: Dict[str, Any]) -> bool:
        with self._lock:
            if collection_name not in self.collections:
                self.collections[collection_name] = {}
            self.collections[collection_name][doc_id] = dict(data)
            return True

    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.collections.get(collection_name, {}).get(doc_id)

    def get_all_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.collections.get(collection_name, {}).values())

    def update_document(self, collection_name: str, doc_id: str, update_data: Dict[str, Any]) -> bool:
        with self._lock:
            if collection_name in self.collections and doc_id in self.collections[collection_name]:
                self.collections[collection_name][doc_id].update(update_data)
                return True
            return False

    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        with self._lock:
            if collection_name in self.collections and doc_id in self.collections[collection_name]:
                del self.collections[collection_name][doc_id]
                return True
            return False

    def clear_collection(self, collection_name: str):
        with self._lock:
            if collection_name in self.collections:
                self.collections[collection_name] = {}

    def clear_all(self):
        with self._lock:
            for k in self.collections:
                self.collections[k] = {}

# Global instance of local database
local_db = LocalDatabase()
