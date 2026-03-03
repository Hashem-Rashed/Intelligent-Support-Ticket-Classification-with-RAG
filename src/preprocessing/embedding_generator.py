import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

def generate_embeddings(
    input_path: str | None = None,
    output_path: str | None = None,
    model_name: str | None = None,
):
    """Produce and save vector representations for cleaned tickets.

    Args:
        input_path: path to cleaned data CSV. Defaults to
            ``{settings.DATA_PROCESSED_PATH}/tickets_cleaned.csv``.
        output_path: destination for embedding array. Defaults to
            ``{settings.DATA_EMBEDDINGS_PATH}/ticket_embeddings.npy``.
        model_name: sentence-transformers model name to use; falls back to
            ``settings.MODEL_NAME``.

    Returns:
        The generated numpy array of embeddings.
    """
    base_dir = settings.PROJECT_ROOT

    if input_path is None:
        input_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")
    if output_path is None:
        output_path = os.path.join(base_dir, settings.DATA_EMBEDDINGS_PATH, "ticket_embeddings.npy")
    if model_name is None:
        model_name = settings.MODEL_NAME

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Generating embeddings with model '{model_name}'")
    data = pd.read_csv(input_path)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(data["clean_text"].tolist(), show_progress_bar=True)

    np.save(output_path, embeddings)
    logger.info(f"Embeddings saved to {output_path}")
    return embeddings
