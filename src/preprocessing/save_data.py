import os
import pandas as pd
import numpy as np

def save_cleaned_data(data, path="data/interim/tickets_cleaned.csv"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save the cleaned CSV
    data.to_csv(path, index=False)
    print(f"Cleaned data saved to: {path}")

def save_embeddings(embeddings, path="data/interim/ticket_embeddings.npy"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save embeddings as numpy array
    np.save(path, embeddings)
    print(f"Embeddings saved to: {path}")
