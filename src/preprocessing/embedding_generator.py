"""
Embedding Generator for Support Ticket Classification
Saves embeddings as batch files then combines
"""

import os
import sys
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import gc
import torch
import time
import json
import glob
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Reliable embedding generator using batch files"""
    
    def __init__(
        self,
        model_name: str = settings.MODEL_NAME,
        batch_size: int = settings.BATCH_SIZE,
        use_gpu: bool = True
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Batch size
        if batch_size is None:
            if self.use_gpu:
                self.batch_size = 256
            else:
                self.batch_size = 64
        else:
            self.batch_size = batch_size
        
        self.device = None
        self.model = None
        self.embedding_dim = None
        
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_docs': 0,
            'docs_per_second': 0
        }
    
    def setup_device(self) -> str:
        """Configure device"""
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[OK] GPU: {gpu_name}")
            logger.info(f"[OK] GPU Memory: {gpu_memory:.1f} GB")
            logger.info(f"[OK] Batch size: {self.batch_size}")
        else:
            self.device = 'cpu'
            self.batch_size = min(self.batch_size, 64)
            logger.info(f"[OK] Using CPU with batch_size={self.batch_size}")
        
        return self.device
    
    def load_model(self):
        """Load the sentence transformer model"""
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"[OK] Embedding dimension: {self.embedding_dim}")
        return self.model
    
    def load_data(self, input_path: str) -> Tuple[List[str], pd.DataFrame]:
        """Load and preprocess data efficiently"""
        logger.info(f"Loading data from {input_path}")
        
        df = pd.read_csv(input_path)
        
        if 'clean_text' not in df.columns:
            raise ValueError("'clean_text' column not found in input file")
        
        texts = df['clean_text'].fillna('').tolist()
        
        # Extract categories
        category_col = None
        for col in ['Issue_Category', 'category', 'Category', 'ticket_category']:
            if col in df.columns:
                category_col = col
                break
        
        if category_col:
            categories = df[category_col].fillna('unknown').tolist()
        else:
            categories = ['unknown'] * len(df)
        
        total_texts = len(texts)
        logger.info(f"[OK] Loaded {total_texts:,} documents")
        
        metadata = pd.DataFrame({
            "Issue_Category": categories,
            "embedding_index": range(total_texts)
        })
        
        category_counts = metadata['Issue_Category'].value_counts()
        logger.info(f"[OK] Categories: {len(category_counts)} unique")
        
        del df
        gc.collect()
        
        return texts, metadata
    
    def generate_embeddings(
        self,
        texts: List[str],
        output_dir: str,
        metadata: pd.DataFrame = None
    ) -> str:
        """
        Generate embeddings - saves each batch as separate file, then combines
        """
        total_texts = len(texts)
        self.stats['total_docs'] = total_texts
        self.stats['start_time'] = time.time()
        
        # Normalize output directory path for Windows
        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create batch directory
        batch_dir = os.path.join(output_dir, "batches")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save metadata
        if metadata is not None:
            metadata_path = os.path.join(output_dir, "ticket_metadata.csv")
            metadata.to_csv(metadata_path, index=False)
            logger.info(f"[OK] Metadata saved to {metadata_path}")
        
        expected_size_gb = (total_texts * self.embedding_dim * 4) / (1024**3)
        logger.info(f"Expected embeddings size: {expected_size_gb:.2f} GB")
        logger.info(f"Starting embedding generation...")
        logger.info(f"Batch size: {self.batch_size}")
        
        start_time = time.time()
        
        # Process batches and save each batch to a separate file
        batch_files = []
        
        with tqdm(total=total_texts, desc="Generating embeddings", unit="docs") as pbar:
            for i in range(0, total_texts, self.batch_size):
                batch_end = min(i + self.batch_size, total_texts)
                batch_texts = texts[i:batch_end]
                
                try:
                    # Encode batch
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    
                    # Ensure float32
                    if batch_embeddings.dtype != np.float32:
                        batch_embeddings = batch_embeddings.astype(np.float32)
                    
                    # Save batch to file
                    batch_file = os.path.join(batch_dir, f"batch_{i:08d}.npy")
                    np.save(batch_file, batch_embeddings)
                    batch_files.append(batch_file)
                    
                except Exception as e:
                    logger.error(f"Error encoding batch {i}-{batch_end}: {e}")
                    if self.batch_size > 32:
                        self.batch_size = self.batch_size // 2
                        logger.info(f"Reduced batch size to {self.batch_size}")
                        continue
                    else:
                        raise
                
                # Clear GPU cache periodically
                if self.device == 'cuda' and (i // self.batch_size) % 100 == 0:
                    torch.cuda.empty_cache()
                
                pbar.update(batch_end - i)
        
        # Combine all batch files
        logger.info("Combining batch files into single embeddings file...")
        
        # Normalize the output path for Windows
        embeddings_path = os.path.normpath(os.path.join(output_dir, "ticket_embeddings.npy"))

        if not batch_files:
            raise ValueError("No batch files were created during embedding generation")

        # Get the shape from the first batch
        first_batch = np.load(batch_files[0])
        embeddings_shape = (total_texts, first_batch.shape[1])
        
        # FIXED: Use regular numpy array instead of memmap to avoid Windows path issues
        logger.info(f"Creating array of shape {embeddings_shape}...")
        embeddings_array = np.zeros(embeddings_shape, dtype=np.float32)

        written = 0
        for batch_file in tqdm(batch_files, desc="Combining batches"):
            batch_data = np.load(batch_file)
            batch_size = batch_data.shape[0]
            embeddings_array[written:written + batch_size] = batch_data
            written += batch_size

            # Delete batch file to save space
            try:
                os.remove(batch_file)
            except Exception as e:
                logger.warning(f"Could not remove batch file {batch_file}: {e}")

        # Save the combined array
        logger.info(f"Saving embeddings to {embeddings_path}...")
        np.save(embeddings_path, embeddings_array)
        
        # Remove batch directory
        try:
            os.rmdir(batch_dir)
        except Exception as e:
            logger.warning(f"Could not remove batch directory: {e}")
        
        # Calculate performance
        elapsed_time = time.time() - start_time
        docs_per_second = total_texts / elapsed_time
        
        self.stats['end_time'] = time.time()
        self.stats['docs_per_second'] = docs_per_second
        self.stats['elapsed_minutes'] = elapsed_time / 60
        
        logger.info(f"[OK] Generation complete!")
        logger.info(f"[OK] Time: {elapsed_time/60:.2f} minutes")
        logger.info(f"[OK] Speed: {docs_per_second:.1f} docs/second")
        
        # Verify saved file
        self._verify_embeddings(embeddings_path, total_texts)
        
        # Save statistics
        self._save_stats(output_dir)
        
        return embeddings_path
    
    def _verify_embeddings(self, embeddings_path: str, expected_shape: int):
        """Verify the saved embeddings file"""
        logger.info("Verifying saved embeddings...")
        
        try:
            test_load = np.load(embeddings_path, allow_pickle=False)
            actual_shape = test_load.shape[0]
            
            if actual_shape == expected_shape:
                logger.info(f"[OK] Verification successful! Shape: {test_load.shape}")
                file_size_gb = os.path.getsize(embeddings_path) / (1024**3)
                logger.info(f"[OK] File size: {file_size_gb:.2f} GB")
            else:
                logger.error(f"[ERROR] Shape mismatch! Expected {expected_shape}, got {actual_shape}")
                raise ValueError("Shape verification failed")
        except Exception as e:
            logger.error(f"[ERROR] Verification failed: {e}")
            raise
    
    def _save_stats(self, output_dir: str):
        """Save embedding generation statistics"""
        stats = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'batch_size': self.batch_size,
            'device': self.device,
            'total_documents': self.stats['total_docs'],
            'time_minutes': self.stats['elapsed_minutes'],
            'docs_per_second': self.stats['docs_per_second'],
            'timestamp': datetime.now().isoformat()
        }
        
        stats_path = os.path.join(output_dir, "embedding_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"[OK] Statistics saved to {stats_path}")
    
    def run(
        self,
        input_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Tuple[str, pd.DataFrame]:
        """Main pipeline to generate embeddings"""
        base_dir = settings.PROJECT_ROOT
        
        if input_path is None:
            input_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")
        
        if output_dir is None:
            output_dir = os.path.join(base_dir, settings.DATA_EMBEDDINGS_PATH)
        
        # Normalize paths for Windows
        input_path = os.path.normpath(input_path)
        output_dir = os.path.normpath(output_dir)
        
        self.setup_device()
        self.load_model()
        texts, metadata = self.load_data(input_path)
        embeddings_path = self.generate_embeddings(texts, output_dir, metadata)
        
        return embeddings_path, metadata


def load_embeddings(
    embeddings_path: str,
    metadata_path: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata for use"""
    logger.info(f"Loading embeddings from {embeddings_path}")
    
    # Normalize paths for Windows
    embeddings_path = os.path.normpath(embeddings_path)
    metadata_path = os.path.normpath(metadata_path)
    
    # Load the entire array
    embeddings = np.load(embeddings_path, allow_pickle=False)
    logger.info(f"[OK] Embeddings shape: {embeddings.shape}")
    
    metadata = pd.read_csv(metadata_path)
    logger.info(f"[OK] Metadata shape: {metadata.shape}")
    
    return embeddings, metadata


def generate_embeddings(
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 256,
    use_gpu: bool = True
) -> str:
    """Convenience function to generate embeddings"""
    generator = EmbeddingGenerator(
        batch_size=batch_size,
        use_gpu=use_gpu
    )
    embeddings_path, _ = generator.run(input_path, output_dir)
    return embeddings_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for support tickets")
    parser.add_argument("--input", type=str, help="Input CSV file path")
    parser.add_argument("--output", type=str, help="Output directory path")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    
    args = parser.parse_args()
    
    embeddings_path = generate_embeddings(
        input_path=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu
    )
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Embedding generation complete!")
    print(f"[OUTPUT] Embeddings saved to: {embeddings_path}")
    print(f"{'='*60}\n")