import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_embeddings(
    input_path: str | None = None,
    output_dir: str | None = None,
    model_name: str | None = None,
):
    base_dir = settings.PROJECT_ROOT

    if input_path is None:
        input_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")

    if output_dir is None:
        output_dir = os.path.join(base_dir, settings.DATA_EMBEDDINGS_PATH)

    if model_name is None:
        model_name = settings.MODEL_NAME

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading cleaned data from {input_path}")
    data = pd.read_csv(input_path)

    # Validation
    if "clean_text" not in data.columns:
        raise ValueError("clean_text column is missing!")

    if "Issue_Category" not in data.columns:
        raise ValueError("Issue_Category column is missing!")

    if data["clean_text"].isnull().any():
        raise ValueError("clean_text contains null values")

    logger.info(f"Generating embeddings using model: {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        data["clean_text"].tolist(),
        show_progress_bar=True
    )

    # Save embeddings
    embeddings_path = os.path.join(output_dir, "ticket_embeddings.npy")
    np.save(embeddings_path, embeddings)

    # Save metadata
    metadata = data.copy()
    metadata["embedding_index"] = range(len(metadata))

    metadata_path = os.path.join(output_dir, "ticket_metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    logger.info("Embeddings and metadata saved successfully")

    return embeddings