"""
Build Chroma vector database from pre‑computed ticket embeddings and metadata.
Run once after preprocessing and embedding generation.
"""
import sys
from pathlib import Path
import numpy as np
import chromadb
from chromadb.api.types import Documents, Embeddings, Metadatas, IDs

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_chroma_index(
    embeddings_path: str = None,
    metadata_path: str = None,
    collection_name: str = "ticket_embeddings",
    persist_directory: str = None,
    batch_size: int = 5000,  # Chroma max is 5461
):
    if embeddings_path is None:
        embeddings_path = project_root / settings.DATA_EMBEDDINGS_PATH / "ticket_embeddings.npy"
    if metadata_path is None:
        metadata_path = project_root / settings.DATA_EMBEDDINGS_PATH / "ticket_metadata.csv"
    if persist_directory is None:
        persist_directory = project_root / settings.DATA_EMBEDDINGS_PATH / "chroma_db"

    logger.info("Loading embeddings...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    logger.info("Loading metadata...")
    import pandas as pd
    metadata_df = pd.read_csv(metadata_path)
    logger.info(f"Metadata shape: {metadata_df.shape}")

    # Ensure metadata has clean_text and category
    if 'embedding_index' not in metadata_df.columns:
        metadata_df['embedding_index'] = range(len(metadata_df))

    # Prepare data for Chroma
    ids = [str(i) for i in range(embeddings.shape[0])]
    documents = metadata_df['clean_text'].fillna('').tolist()
    metadatas = metadata_df[['category', 'embedding_index']].to_dict('records')
    # Add clean_text preview to metadata (truncated for storage efficiency)
    for i, row in enumerate(metadatas):
        row['clean_text'] = documents[i][:500]

    # Initialize Chroma client (persistent)
    client = chromadb.PersistentClient(path=str(persist_directory))

    # Delete existing collection if exists
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection {collection_name}")
    except Exception:
        pass

    # Create collection WITHOUT embedding function (we'll add embeddings directly)
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Created collection {collection_name}")

    # Add in batches
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logger.info(f"Adding batch {start} to {end} ({(end-start):,} vectors)...")

        # Convert embeddings to list of lists
        batch_embeddings = embeddings[start:end].tolist()

        collection.add(
            ids=ids[start:end],
            embeddings=batch_embeddings,
            metadatas=metadatas[start:end],
            documents=documents[start:end],  # optional, for querying with text
        )

    logger.info(f"Chroma index built with {collection.count()} vectors")
    logger.info(f"Persistent storage at {persist_directory}")
    return collection


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Chroma index from ticket embeddings")
    parser.add_argument("--embeddings", type=str, help="Path to embeddings .npy file")
    parser.add_argument("--metadata", type=str, help="Path to metadata CSV")
    parser.add_argument("--persist-dir", type=str, help="Directory to persist Chroma DB")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for adding vectors")
    args = parser.parse_args()

    build_chroma_index(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        persist_directory=args.persist_dir,
        batch_size=args.batch_size,
    )