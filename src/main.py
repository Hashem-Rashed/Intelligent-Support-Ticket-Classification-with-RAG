import os, sys

# ensure the src directory is on the import path so we can import the
# `preprocessing` package when the script is executed directly.
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from preprocessing.pipeline import run_pipeline
from preprocessing.embedding_generator import generate_embeddings

if __name__ == "__main__":

    run_pipeline()
    generate_embeddings()