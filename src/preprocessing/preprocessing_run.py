import os
import sys

# Ensure project root is on sys.path when executed as a script.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.embedding_generator import generate_embeddings


# `preprocessing` package when the script is executed directly.


if __name__ == "__main__":
    run_pipeline()
    generate_embeddings()