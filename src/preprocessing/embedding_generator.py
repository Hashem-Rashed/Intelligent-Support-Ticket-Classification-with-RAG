import os
import numpy as np
import pandas as pd
import gc
import torch
from sentence_transformers import SentenceTransformer
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_embeddings(
    input_path: str | None = None,
    output_dir: str | None = None,
    model_name: str | None = None,
    batch_size: int = settings.BATCH_SIZE,
    use_gpu: bool = True,
):
    base_dir = settings.PROJECT_ROOT

    if input_path is None:
        input_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")

    if output_dir is None:
        output_dir = os.path.join(base_dir, settings.DATA_EMBEDDINGS_PATH)

    if model_name is None:
        model_name = settings.MODEL_NAME

    os.makedirs(output_dir, exist_ok=True)

    # Check GPU
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        batch_size = 256  # GPU can handle larger batches
    else:
        logger.warning("GPU not available, using CPU (will be slower)")
        batch_size = 64  # Smaller batches for CPU
    
    logger.info(f"Loading cleaned data from {input_path}")
    
    # Load data efficiently
    usecols = ['clean_text']
    category_col = None
    
    sample = pd.read_csv(input_path, nrows=1)
    if "Issue_Category" in sample.columns:
        category_col = "Issue_Category"
        usecols.append("Issue_Category")
    elif "category" in sample.columns:
        category_col = "category"
        usecols.append("category")
    
    # Use chunks to avoid loading all at once
    chunk_size = 50000  # Process 50k rows at a time
    all_texts = []
    all_categories = []
    
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunk_size):
        if chunk["clean_text"].isnull().any():
            chunk["clean_text"] = chunk["clean_text"].fillna("")
        all_texts.extend(chunk["clean_text"].tolist())
        if category_col:
            all_categories.extend(chunk[category_col].tolist())
        else:
            all_categories.extend(["unknown"] * len(chunk))
    
    total_texts = len(all_texts)
    logger.info(f"Loaded {total_texts} documents")
    
    # Save metadata
    logger.info("Saving metadata...")
    metadata = pd.DataFrame({
        "Issue_Category": all_categories,
        "embedding_index": range(total_texts)
    })
    metadata_path = os.path.join(output_dir, "ticket_metadata.csv")
    metadata.to_csv(metadata_path, index=False)
    
    # Free memory
    del all_categories
    gc.collect()
    
    logger.info(f"Generating embeddings with batch_size={batch_size}")
    
    # Load model on GPU
    model = SentenceTransformer(model_name, device=device)
    
    # Optional: Use FP16 for faster computation
    if device == 'cuda':
        model.half()  # Half precision = 2x faster
    
    # Create memory-mapped file
    embeddings_path = os.path.join(output_dir, "ticket_embeddings.npy")
    expected_size_gb = (total_texts * 384 * 4) / (1024**3)
    logger.info(f"Expected file size: {expected_size_gb:.2f} GB")
    
    embeddings_memmap = np.memmap(
        embeddings_path,
        dtype='float32',
        mode='w+',
        shape=(total_texts, 384)
    )
    
    # Process in batches with progress bar
    from tqdm import tqdm
    
    for i in tqdm(range(0, total_texts, batch_size), desc="Encoding"):
        batch_end = min(i + batch_size, total_texts)
        batch_texts = all_texts[i:batch_end]
        
        # Encode batch
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Write to memmap
        embeddings_memmap[i:batch_end] = batch_embeddings
        
        # Flush periodically
        if batch_end % (batch_size * 10) == 0:
            embeddings_memmap.flush()
        
        # Clear GPU cache if using CUDA
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    embeddings_memmap.flush()
    del embeddings_memmap
    
    logger.info(f"✅ Embeddings saved to {embeddings_path}")
    logger.info(f"Final file size: {os.path.getsize(embeddings_path) / (1024**3):.2f} GB")
    
    return embeddings_path