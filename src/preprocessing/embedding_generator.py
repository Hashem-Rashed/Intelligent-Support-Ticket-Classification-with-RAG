import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_embeddings():

    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    input_path = os.path.join(
        BASE_DIR,
        "data/processed/tickets_cleaned.csv"
    )

    output_path = os.path.join(
        BASE_DIR,
        "data/embeddings/ticket_embeddings.npy"
    )

    # Load cleaned dataset
    data = pd.read_csv(input_path)

    # Load embedding model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(
        data["clean_text"].tolist(),
        show_progress_bar=True
    )

    # Save embeddings
    np.save(output_path, embeddings)

    print("Embeddings generated and saved successfully!")