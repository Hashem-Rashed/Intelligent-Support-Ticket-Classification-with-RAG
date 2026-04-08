import os
import sys

# Ensure project root is on sys.path when executed as a script.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.embedding_generator import generate_embeddings
from src.preprocessing.data_merger import merge_datasets

if __name__ == "__main__":
    # Merge datasets with intelligent categorization
    # Only categorizes tweets that don't have categories or have generic ones
    merge_datasets(
        categorize_tweets=True,
        overwrite_existing_categories=False  # Preserve existing specific categories
    )
    
    # Run full pipeline with merged data
    run_pipeline(use_merged_data=True)
    
    # Generate embeddings from cleaned data
    generate_embeddings()