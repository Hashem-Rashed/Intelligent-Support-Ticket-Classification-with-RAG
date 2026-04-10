"""
Main preprocessing pipeline runner
Handles dataset merging, cleaning, and embedding generation
With interactive menu to choose which steps to run
"""

import os
import sys
import time
import argparse

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure project root is on sys.path when executed as a script.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.embedding_generator import EmbeddingGenerator
from src.preprocessing.data_merger import merge_datasets
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_menu():
    """Print the interactive menu"""
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE MENU")
    print("="*60)
    print("\nAvailable Pipeline Steps:")
    print("  1. Merge Datasets (Combine CRM tickets + Twitter data)")
    print("  2. Clean Data (Text cleaning, preprocessing)")
    print("  3. Generate Embeddings (Create vector embeddings)")
    print("  4. Run Complete Pipeline (Steps 1, 2, and 3)")
    print("\nOptions:")
    print("  Enter numbers separated by commas (e.g., 1,2,3)")
    print("  Enter 'all' to run all steps")
    print("  Enter 'q' to quit")
    print("\n" + "="*60)


def check_gpu():
    """Check GPU availability and return status"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[OK] GPU detected: {gpu_name}")
            logger.info(f"[OK] GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            logger.warning("[WARNING] CUDA not available. GPU will not be used.")
            return False
    except Exception as e:
        logger.warning(f"[WARNING] GPU check failed: {e}")
        return False


def run_step_1_merge(categorize_tweets=True, overwrite_categories=False):
    """Step 1: Merge datasets"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Merging Datasets")
    logger.info("="*60)
    
    try:
        merge_datasets(
            categorize_tweets=categorize_tweets,
            overwrite_existing_categories=overwrite_categories
        )
        logger.info("[OK] Dataset merge completed")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Dataset merge failed: {e}")
        return False


def run_step_2_clean(use_merged_data=True):
    """Step 2: Clean data"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Cleaning Data")
    logger.info("="*60)
    
    try:
        run_pipeline(use_merged_data=use_merged_data)
        logger.info("[OK] Cleaning pipeline completed")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Cleaning pipeline failed: {e}")
        return False


def run_step_3_embeddings(batch_size=256, use_gpu=True, test_mode=False, sample_size=1000):
    """Step 3: Generate embeddings"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Generating Embeddings")
    logger.info("="*60)
    
    # Get paths
    base_dir = settings.PROJECT_ROOT
    input_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")
    output_dir = os.path.join(base_dir, settings.DATA_EMBEDDINGS_PATH)
    
    # Check if input exists
    if not os.path.exists(input_path):
        logger.error(f"[ERROR] Cleaned data not found: {input_path}")
        logger.info("Please run Step 2 (Clean Data) first")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    if test_mode:
        logger.info(f"[TEST MODE] Running with {sample_size:,} samples")
        
        # Create sample file
        import pandas as pd
        sample_path = os.path.join(output_dir, "sample.csv")
        df = pd.read_csv(input_path, nrows=sample_size)
        df.to_csv(sample_path, index=False)
        
        # Generate embeddings for sample
        generator = EmbeddingGenerator(
            model_name=settings.MODEL_NAME,
            batch_size=min(batch_size, 128),
            use_gpu=use_gpu
        )
        
        try:
            embeddings_path, metadata = generator.run(
                input_path=sample_path,
                output_dir=os.path.join(output_dir, "test")
            )
            logger.info(f"[OK] Test embeddings generated: {embeddings_path}")
            logger.info(f"[OK] Test metadata shape: {metadata.shape}")
            
            # Clean up
            try:
                os.remove(sample_path)
            except:
                pass
            
            return True
        except Exception as e:
            logger.error(f"[ERROR] Test embedding generation failed: {e}")
            return False
    else:
        logger.info("[FULL MODE] Generating embeddings for all documents")
        
        try:
            generator = EmbeddingGenerator(
                model_name=settings.MODEL_NAME,
                batch_size=batch_size,
                use_gpu=use_gpu
            )
            
            embeddings_path, metadata = generator.run(
                input_path=input_path,
                output_dir=output_dir
            )
            
            logger.info(f"[OK] Embeddings generated successfully")
            logger.info(f"   Path: {embeddings_path}")
            logger.info(f"   Shape: {metadata.shape}")
            logger.info(f"   Categories: {metadata['Issue_Category'].nunique()}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_step_4_all(batch_size=256, use_gpu=True):
    """Step 4: Run complete pipeline"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Running Complete Pipeline")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Step 1: Merge
    if not run_step_1_merge():
        logger.error("Step 1 failed. Stopping pipeline.")
        return False
    
    # Step 2: Clean
    if not run_step_2_clean():
        logger.error("Step 2 failed. Stopping pipeline.")
        return False
    
    # Step 3: Embeddings
    if not run_step_3_embeddings(batch_size=batch_size, use_gpu=use_gpu):
        logger.error("Step 3 failed.")
        return False
    
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info(f"[OK] Complete pipeline finished in {elapsed_time/60:.2f} minutes")
    logger.info("="*60)
    
    return True


def parse_user_choice(choice_str):
    """Parse user input and return list of steps to run"""
    if choice_str.lower() == 'q':
        return None
    if choice_str.lower() == 'all':
        return [1, 2, 3, 4]
    
    # Parse comma-separated numbers
    steps = []
    for part in choice_str.split(','):
        part = part.strip()
        if part.isdigit():
            step_num = int(part)
            if 1 <= step_num <= 4:
                steps.append(step_num)
            else:
                print(f"Invalid step number: {step_num}. Please enter 1-4")
        elif '-' in part:
            # Handle ranges like 1-3
            start, end = map(int, part.split('-'))
            steps.extend(range(start, min(end, 4) + 1))
    
    return sorted(set(steps))


def get_pipeline_parameters():
    """Get user preferences for pipeline parameters"""
    print("\n" + "="*60)
    print("PIPELINE CONFIGURATION")
    print("="*60)
    
    # GPU selection
    use_gpu = input("\nUse GPU for embeddings? (y/n) [y]: ").lower().strip() != 'n'
    
    # Batch size
    batch_size_input = input("Batch size for embeddings [256]: ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 256
    
    # Test mode for embeddings
    test_mode = input("Run embeddings in test mode with sample? (y/n) [n]: ").lower().strip() == 'y'
    
    sample_size = 1000
    if test_mode:
        sample_size_input = input("Sample size for test mode [1000]: ").strip()
        sample_size = int(sample_size_input) if sample_size_input else 1000
    
    # Category options
    categorize = input("Categorize tweets? (y/n) [y]: ").lower().strip() != 'n'
    overwrite = input("Overwrite existing categories? (y/n) [n]: ").lower().strip() == 'y'
    
    return {
        'use_gpu': use_gpu,
        'batch_size': batch_size,
        'test_mode': test_mode,
        'sample_size': sample_size,
        'categorize_tweets': categorize,
        'overwrite_categories': overwrite
    }


def run_pipeline_interactive():
    """Interactive mode to choose which steps to run"""
    # Get parameters once
    params = get_pipeline_parameters()
    
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        
        steps = parse_user_choice(choice)
        
        if steps is None:
            print("\nExiting. Goodbye!")
            break
        
        if not steps:
            print("Invalid choice. Please try again.")
            continue
        
        print(f"\nRunning steps: {steps}")
        start_time = time.time()
        
        # Run selected steps
        for step in steps:
            if step == 1:
                run_step_1_merge(
                    categorize_tweets=params['categorize_tweets'],
                    overwrite_categories=params['overwrite_categories']
                )
            elif step == 2:
                run_step_2_clean(use_merged_data=True)
            elif step == 3:
                run_step_3_embeddings(
                    batch_size=params['batch_size'],
                    use_gpu=params['use_gpu'],
                    test_mode=params['test_mode'],
                    sample_size=params['sample_size']
                )
            elif step == 4:
                run_step_4_all(
                    batch_size=params['batch_size'],
                    use_gpu=params['use_gpu']
                )
        
        elapsed_time = time.time() - start_time
        print(f"\n[OK] Selected steps completed in {elapsed_time/60:.2f} minutes")
        
        input("\nPress Enter to continue...")


def run_pipeline_non_interactive(args):
    """Non-interactive mode with command line arguments"""
    print("\n" + "="*60)
    print("Running in NON-INTERACTIVE mode")
    print("="*60)
    
    start_time = time.time()
    
    # Determine which steps to run
    steps_to_run = []
    if args.all:
        steps_to_run = [1, 2, 3]
    else:
        if args.step1:
            steps_to_run.append(1)
        if args.step2:
            steps_to_run.append(2)
        if args.step3:
            steps_to_run.append(3)
    
    if not steps_to_run:
        print("No steps selected. Use --step1, --step2, --step3, or --all")
        return
    
    print(f"Running steps: {steps_to_run}")
    
    # Run selected steps
    for step in steps_to_run:
        if step == 1:
            run_step_1_merge(
                categorize_tweets=not args.no_categorize,
                overwrite_categories=args.overwrite_categories
            )
        elif step == 2:
            run_step_2_clean(use_merged_data=True)
        elif step == 3:
            run_step_3_embeddings(
                batch_size=args.batch_size,
                use_gpu=not args.no_gpu,
                test_mode=args.test,
                sample_size=args.sample_size
            )
    
    elapsed_time = time.time() - start_time
    print(f"\n[OK] Selected steps completed in {elapsed_time/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for ticket classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python preprocessing_run.py
  
  # Non-interactive mode - run specific steps
  python preprocessing_run.py --step1 --step2
  
  # Run all steps
  python preprocessing_run.py --all
  
  # Run only embeddings with test mode
  python preprocessing_run.py --step3 --test --sample-size 5000
  
  # Run merge and clean only
  python preprocessing_run.py --step1 --step2 --no-gpu
        """
    )
    
    # Step selection
    parser.add_argument("--step1", action="store_true", help="Run Step 1: Merge datasets")
    parser.add_argument("--step2", action="store_true", help="Run Step 2: Clean data")
    parser.add_argument("--step3", action="store_true", help="Run Step 3: Generate embeddings")
    parser.add_argument("--all", action="store_true", help="Run all steps (1, 2, and 3)")
    
    # Configuration options
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embeddings")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for embeddings")
    parser.add_argument("--test", action="store_true", help="Run embeddings in test mode with sample")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size for test mode")
    parser.add_argument("--no-categorize", action="store_true", help="Don't categorize tweets")
    parser.add_argument("--overwrite-categories", action="store_true", help="Overwrite existing categories")
    
    # Mode selection
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check if any steps were selected for non-interactive mode
    has_step_selection = args.step1 or args.step2 or args.step3 or args.all
    
    # Run in appropriate mode
    if args.interactive or (not has_step_selection and not args.all):
        # Interactive mode (default if no steps specified)
        run_pipeline_interactive()
    else:
        # Non-interactive mode
        run_pipeline_non_interactive(args)