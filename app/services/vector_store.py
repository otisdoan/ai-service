import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from app.core.config import settings

class VectorDocument:
    def __init__(
        self,
        doc_id: str,
        doc_type: str,  # "menu_item" | "branch"
        text: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ):
        self.doc_id = doc_id
        self.doc_type = doc_type
        self.text = text
        self.vector = np.array(vector, dtype=np.float32)
        self.metadata = metadata

class VectorStore:
    def __init__(self):
        self.documents: Dict[str, VectorDocument] = {}
        self.storage_dir = settings.VECTOR_STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        self.storage_file = os.path.join(self.storage_dir, "vector_store.json")
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        doc = VectorDocument(
                            doc_id=item["doc_id"],
                            doc_type=item["doc_type"],
                            text=item["text"],
                            vector=item["vector"],
                            metadata=item["metadata"]
                        )
                        self.documents[doc.doc_id] = doc
            except Exception as e:
                print(f"[VectorStore] Failed to load store: {e}")

    def save_to_disk(self):
        try:
            data = []
            for doc in self.documents.values():
                data.append({
                    "doc_id": doc.doc_id,
                    "doc_type": doc.doc_type,
                    "text": doc.text,
                    "vector": doc.vector.tolist(),
                    "metadata": doc.metadata
                })
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[VectorStore] Failed to save store: {e}")

    def add_or_update(self, doc_id: str, doc_type: str, text: str, vector: List[float], metadata: Dict[str, Any]):
        self.documents[doc_id] = VectorDocument(doc_id, doc_type, text, vector, metadata)
        self.save_to_disk()

    def remove(self, doc_id: str):
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.save_to_disk()

    def clear(self):
        self.documents.clear()
        self.save_to_disk()

    def search_hybrid(
        self,
        query_vector: List[float],
        doc_type: str = "menu_item",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs Cosine Similarity search with structured metadata filtering.
        """
        if not self.documents:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        candidates = []
        for doc in self.documents.values():
            if doc.doc_type != doc_type:
                continue

            meta = doc.metadata

            # Apply hard metadata filters
            if filters:
                # 1. Price filters
                price = meta.get("price", 0.0)
                if "max_price" in filters and filters["max_price"] is not None:
                    if price > filters["max_price"]:
                        continue
                if "min_price" in filters and filters["min_price"] is not None:
                    if price < filters["min_price"]:
                        continue

                # 2. Branch filter
                if "branch_id" in filters and filters["branch_id"]:
                    if str(meta.get("branch_id")) != str(filters["branch_id"]):
                        continue

                # 3. District / Location filter
                if "district" in filters and filters["district"]:
                    target_dist = filters["district"].lower()
                    doc_dist = meta.get("district", "").lower()
                    doc_addr = meta.get("address", "").lower()
                    if target_dist not in doc_dist and target_dist not in doc_addr:
                        continue

                # 4. Availability & Out of stock filter
                if filters.get("is_available") is True and meta.get("is_available") is False:
                    continue
                if filters.get("is_sold_out") is False and meta.get("is_sold_out") is True:
                    continue

                # 5. Rating filter
                if "min_rating" in filters and filters["min_rating"] is not None:
                    if meta.get("rating", 5.0) < filters["min_rating"]:
                        continue

                # 6. Dish topic filter
                if "dish_topic" in filters and filters["dish_topic"]:
                    topic = str(filters["dish_topic"]).lower()
                    doc_name = meta.get("name", "").lower()
                    if topic not in doc_name:
                        continue

            # Compute Cosine Similarity
            d_norm = np.linalg.norm(doc.vector)
            if d_norm == 0:
                d_norm = 1.0
            
            sim = float(np.dot(q_vec, doc.vector) / (q_norm * d_norm))
            candidates.append({
                "doc_id": doc.doc_id,
                "score": sim,
                "text": doc.text,
                "metadata": meta
            })

        # Sort by similarity score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

vector_store = VectorStore()
