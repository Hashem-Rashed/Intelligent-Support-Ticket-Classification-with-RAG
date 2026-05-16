"""
Retriever for similar tickets using Chroma vector database.
"""
import sys
from pathlib import Path
import chromadb
from typing import List, Dict, Any, Optional

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TicketRetriever:
    """
    Retrieve similar tickets from Chroma collection.
    Supports metadata filtering (by category, etc.)
    """
    def __init__(
        self,
        collection_name: str = "ticket_embeddings",
        persist_directory: Optional[str] = None,
    ):
        if persist_directory is None:
            persist_directory = Path(settings.DATA_EMBEDDINGS_PATH) / "chroma_db"
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_collection(collection_name)
        logger.info(f"Loaded Chroma collection '{collection_name}' with {self.collection.count()} vectors")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar tickets.

        Args:
            query: Query text
            top_k: Number of results
            score_threshold: Minimum similarity (1 - distance) to include
            filter_category: Optional category to filter results

        Returns:
            List of dicts with keys: 'score', 'metadata', 'document'
        """
        where_filter = None
        if filter_category:
            where_filter = {"category": filter_category}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["distances", "metadatas", "documents"],
        )

        # Chroma returns distances (0 = identical, larger = less similar)
        # Convert to similarity score (1 - distance) assuming embeddings normalized
        formatted = []
        if results['ids'] and results['ids'][0]:
            for i, id_ in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]  # cosine distance (0-2 range)
                # For normalized embeddings, cosine similarity = 1 - distance/2? Actually chroma uses cosine distance = 1 - similarity.
                # Simpler: similarity = 1 - distance (if distance is 0..1). We'll just cap.
                similarity = max(0.0, 1.0 - distance)
                if similarity < score_threshold:
                    continue
                formatted.append({
                    "score": similarity,
                    "metadata": results['metadatas'][0][i],
                    "document": results['documents'][0][i] if results['documents'] else "",
                })
        return formatted

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_category: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        return [self.retrieve(q, top_k, score_threshold, filter_category) for q in queries]


if __name__ == "__main__":
    retriever = TicketRetriever()
    test_query = "Someone stole my credit card and made unauthorized purchases"
    results = retriever.retrieve(test_query, top_k=3)
    print(f"Query: {test_query}")
    for res in results:
        print(f"  Score: {res['score']:.4f} | Category: {res['metadata']['category']} | Text: {res['document'][:80]}...")