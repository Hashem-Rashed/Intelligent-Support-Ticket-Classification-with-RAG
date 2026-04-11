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
from src.models.baseline.tweet_labeler import run_tweet_labeling
from src.models.baseline.train import run_baseline_training
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

TWEETS_HIGH_CONFIDENCE_FILENAME = "tweets_high_confidence.csv"
COMBINED_CLEANED_FILENAME = "combined_cleaned.csv"


def print_menu():
    """Print the interactive menu"""
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE MENU")
    print("="*60)
    print("\nAvailable Pipeline Steps:")
    print("  1. Clean CRM tickets")
    print("       - outputs data/processed/tickets_cleaned.csv")
    print("  2. Label tweets with baseline model")
    print("       - requires a trained baseline model")
    print("       - outputs data/processed/tweets_high_confidence.csv")
    print("  3. Merge cleaned tickets + labeled tweets")
    print("       - requires steps 1 and 2")
    print("       - outputs data/processed/combined_cleaned.csv")
    print("  4. Generate embeddings from combined cleaned data")
    print("       - uses combined cleaned data if available")
    print("  5. Run complete preprocessing pipeline (steps 1-4)")
    print("\nOptions:")
    print("  Enter numbers separated by commas (e.g., 1,2,3,4)")
    print("  Enter 'all' to run steps 1-4")
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


def run_step_1_merge(categorize_tweets=True, overwrite_categories=False, force=False):
    """Step 1: Merge datasets"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Merging Datasets")
    logger.info("="*60)
    
    output_path = os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, "merged_support_data.csv")
    
    # Check if output already exists
    if os.path.exists(output_path) and not force:
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping dataset merge.")
            return True
    
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


def run_step_2_clean(use_merged_data=True, force=False):
    """Step 2: Clean data"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Cleaning Data")
    logger.info("="*60)
    
    output_path = os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")
    
    # Check if output already exists
    if os.path.exists(output_path) and not force:
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping data cleaning.")
            return True
    
    try:
        run_pipeline(use_merged_data=use_merged_data)
        logger.info("[OK] Cleaning pipeline completed")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Cleaning pipeline failed: {e}")
        return False


def run_step_2_label_tweets(
    confidence_threshold: float = 0.7,
    tweet_chunksize: int = 50000,
    prediction_batch_size: int = 5000,
    force: bool = False,
):
    """Step 2: Label tweets with the trained baseline classifier."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Labeling Tweets")
    logger.info("="*60)

    model_path = get_baseline_model_path()
    ensure_baseline_model(model_path=model_path, force=force)

    output_path = get_high_confidence_tweets_path()
    
    # Check if output already exists
    if os.path.exists(output_path) and not force:
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping tweet labeling.")
            return True
    elif os.path.exists(output_path) and force:
        logger.info("Removing existing high-confidence tweet output: %s", output_path)
        os.remove(output_path)

    try:
        run_tweet_labeling(
            model_path=model_path,
            output_path=output_path,
            confidence_threshold=confidence_threshold,
            tweet_chunksize=tweet_chunksize,
            prediction_batch_size=prediction_batch_size,
            force_rebuild=True,
        )
        logger.info("[OK] Tweet labeling completed")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Tweet labeling failed: {e}")
        return False


def run_step_3_merge_cleaned_data(
    tickets_path: str | None = None,
    tweets_path: str | None = None,
    output_path: str | None = None,
    force: bool = False,
) -> bool:
    """Step 3: Merge cleaned tickets and high-confidence tweets."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Merging Cleaned Data")
    logger.info("="*60)

    if output_path is None:
        output_path = get_combined_cleaned_path()
    
    # Check if output already exists
    if os.path.exists(output_path) and not force:
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping cleaned data merge.")
            return True

    try:
        merge_cleaned_ticket_and_tweets(
            tickets_path=tickets_path,
            tweets_path=tweets_path,
            output_path=output_path,
        )
        logger.info("[OK] Cleaned data merge completed")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Merging cleaned data failed: {e}")
        return False


def run_step_4_generate_embeddings(
    batch_size: int = 256,
    use_gpu: bool = True,
    test_mode: bool = False,
    sample_size: int = 1000,
    input_path: str | None = None,
    force: bool = False,
) -> bool:
    """Step 4: Generate embeddings from the combined cleaned dataset."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Generating Embeddings")
    logger.info("="*60)
    
    # Check if embeddings already exist
    embeddings_path = os.path.join(settings.PROJECT_ROOT, settings.DATA_EMBEDDINGS_PATH, "ticket_embeddings.npy")
    
    if os.path.exists(embeddings_path) and not force and not test_mode:
        response = input(f"Embeddings file {embeddings_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping embedding generation.")
            return True
    elif os.path.exists(embeddings_path) and force:
        logger.info(f"Force flag set. Will overwrite {embeddings_path}")
    
    return run_step_3_embeddings(
        batch_size=batch_size,
        use_gpu=use_gpu,
        test_mode=test_mode,
        sample_size=sample_size,
        input_path=input_path,
        force=force
    )


def get_high_confidence_tweets_path() -> str:
    return os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, TWEETS_HIGH_CONFIDENCE_FILENAME)


def get_combined_cleaned_path() -> str:
    return os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, COMBINED_CLEANED_FILENAME)


def get_baseline_model_path() -> str:
    return os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, "baseline_ticket_classifier.pkl")


def ensure_baseline_model(model_path: str | None = None, force: bool = False) -> str:
    model_path = model_path or get_baseline_model_path()
    
    if os.path.exists(model_path) and not force:
        logger.info("Baseline model already exists at %s", model_path)
        return model_path
    elif os.path.exists(model_path) and force:
        logger.info("Force flag set. Retraining baseline model...")
        os.remove(model_path)

    logger.info("Baseline model not found, training from cleaned ticket data")
    run_baseline_training(
        raw_path=None,
        cleaned_path=os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv"),
        model_output_path=model_path,
        test_size=0.2,
        class_weight=None,
        force_rebuild=False,
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Baseline model training completed but output file not found: {model_path}")

    return model_path


def merge_cleaned_ticket_and_tweets(
    tickets_path: str | None = None,
    tweets_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Merge cleaned tickets with high-confidence labeled tweets."""
    if tickets_path is None:
        tickets_path = os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")
    if tweets_path is None:
        tweets_path = get_high_confidence_tweets_path()
    if output_path is None:
        output_path = get_combined_cleaned_path()

    logger.info("Merging cleaned tickets and high-confidence tweets")

    if not os.path.exists(tickets_path):
        raise FileNotFoundError(f"Cleaned tickets file not found: {tickets_path}")
    if not os.path.exists(tweets_path):
        raise FileNotFoundError(f"High-confidence tweets file not found: {tweets_path}")

    import pandas as pd

    tickets = pd.read_csv(tickets_path, dtype={"clean_text": str, "Issue_Category": str})
    tweets = pd.read_csv(tweets_path, dtype={"clean_text": str, "predicted_category": str})

    if "predicted_category" not in tweets.columns:
        raise ValueError("Expected 'predicted_category' column in high confidence tweets")

    tweets = tweets.rename(columns={"predicted_category": "Issue_Category"})
    tweets = tweets[["clean_text", "Issue_Category"]].copy()
    tweets["source"] = "twitter"

    tickets = tickets[["clean_text", "Issue_Category"]].copy()
    tickets["source"] = "ticket"

    combined = pd.concat([tickets, tweets], ignore_index=True)
    combined.to_csv(output_path, index=False)

    logger.info("Combined cleaned dataset saved to %s", output_path)
    logger.info("Combined dataset size: %d", len(combined))
    logger.info("Category distribution:\n%s", combined["Issue_Category"].value_counts().to_dict())

    return output_path


def run_step_3_embeddings(batch_size=256, use_gpu=True, test_mode=False, sample_size=1000, input_path=None, force=False):
    """Step 3: Generate embeddings"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Generating Embeddings")
    logger.info("="*60)
    
    base_dir = settings.PROJECT_ROOT
    if input_path is None:
        combined_path = get_combined_cleaned_path()
        ticket_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")
        input_path = combined_path if os.path.exists(combined_path) else ticket_path

    output_dir = os.path.join(base_dir, settings.DATA_EMBEDDINGS_PATH)
    
    # Check if input exists
    if not os.path.exists(input_path):
        logger.error(f"[ERROR] Cleaned data not found: {input_path}")
        logger.info("Please run Step 2 (Clean Data) or the full preprocessing pipeline first")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if embeddings already exist
    embeddings_path = os.path.join(output_dir, "ticket_embeddings.npy")
    if os.path.exists(embeddings_path) and not force and not test_mode:
        logger.info(f"Embeddings already exist at {embeddings_path}")
        response = input("Overwrite existing embeddings? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping embedding generation.")
            return True
    
    if test_mode:
        logger.info(f"[TEST MODE] Running with {sample_size:,} samples")
        
        import pandas as pd
        sample_path = os.path.join(output_dir, "sample.csv")
        df = pd.read_csv(input_path, nrows=sample_size)
        df.to_csv(sample_path, index=False)
        
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


def run_step_5_complete_pipeline(batch_size=256, use_gpu=True, force_rerun=False):
    """Run complete pipeline with ticket cleaning, tweet labeling, merging, and embedding generation.
    
    Args:
        batch_size: Batch size for embeddings
        use_gpu: Whether to use GPU
        force_rerun: If True, rerun all steps even if outputs exist
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Running Complete Preprocessing Pipeline (Steps 1-4)")
    logger.info("="*60)
    
    start_time = time.time()

    # Step 1: Clean tickets only
    if not run_step_2_clean(use_merged_data=False, force=force_rerun):
        logger.error("Ticket cleaning failed. Stopping pipeline.")
        return False

    # Step 2: Ensure baseline model exists and label tweets
    try:
        tweet_label_output = get_high_confidence_tweets_path()
        model_path = get_baseline_model_path()
        ensure_baseline_model(model_path=model_path, force=force_rerun)

        if os.path.exists(tweet_label_output) and force_rerun:
            logger.info("Removing existing high-confidence tweet output: %s", tweet_label_output)
            os.remove(tweet_label_output)

        run_tweet_labeling(
            model_path=model_path,
            output_path=tweet_label_output,
            confidence_threshold=0.7,
            tweet_chunksize=50000,
            prediction_batch_size=5000,
            force_rebuild=True,
        )
    except Exception as exc:
        logger.error("Tweet labeling failed: %s", exc)
        return False

    # Step 3: Merge cleaned tickets and labeled tweets
    try:
        merge_cleaned_ticket_and_tweets()
    except Exception as exc:
        logger.error("Merging cleaned ticket and tweet datasets failed: %s", exc)
        return False

    # Step 4: Generate embeddings from the combined cleaned dataset
    if not run_step_3_embeddings(batch_size=batch_size, use_gpu=use_gpu, test_mode=False, sample_size=1000, force=force_rerun):
        logger.error("Embedding generation failed.")
        return False

    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info(f"[OK] Complete preprocessing pipeline finished in {elapsed_time/60:.2f} minutes")
    logger.info("="*60)

    return True


def parse_user_choice(choice_str):
    """Parse user input and return list of steps to run"""
    if choice_str.lower() == 'q':
        return None
    if choice_str.lower() == 'all':
        # Run steps 1-4 only (not step 5 which would repeat them)
        return [1, 2, 3, 4]
    
    # Parse comma-separated numbers
    steps = []
    for part in choice_str.split(','):
        part = part.strip()
        if part.isdigit():
            step_num = int(part)
            if 1 <= step_num <= 5:
                steps.append(step_num)
            else:
                print(f"Invalid step number: {step_num}. Please enter 1-5")
        elif '-' in part:
            # Handle ranges like 1-4
            start, end = map(int, part.split('-'))
            steps.extend(range(start, min(end, 5) + 1))
    
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
    
    # High-confidence tweet threshold
    confidence_threshold_input = input("High-confidence tweet threshold [0.7]: ").strip()
    confidence_threshold = float(confidence_threshold_input) if confidence_threshold_input else 0.7

    # Tweet batch settings
    tweet_chunksize_input = input("Tweet chunk size [50000]: ").strip()
    tweet_chunksize = int(tweet_chunksize_input) if tweet_chunksize_input else 50000
    prediction_batch_size_input = input("Tweet prediction batch size [5000]: ").strip()
    prediction_batch_size = int(prediction_batch_size_input) if prediction_batch_size_input else 5000

    # Force rerun option
    force_rerun = input("Force rerun all steps (ignore cached files)? (y/n) [n]: ").lower().strip() == 'y'

    return {
        'use_gpu': use_gpu,
        'batch_size': batch_size,
        'test_mode': test_mode,
        'sample_size': sample_size,
        'confidence_threshold': confidence_threshold,
        'tweet_chunksize': tweet_chunksize,
        'prediction_batch_size': prediction_batch_size,
        'force_rerun': force_rerun,
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
        print(f"Force rerun mode: {params['force_rerun']}")
        start_time = time.time()
        
        # Track which steps we've run to avoid duplication
        steps_run = set()
        
        # Run selected steps
        for step in steps:
            if step == 1:
                if 1 not in steps_run:
                    run_step_2_clean(use_merged_data=False, force=params['force_rerun'])
                    steps_run.add(1)
            elif step == 2:
                if 2 not in steps_run:
                    run_step_2_label_tweets(
                        confidence_threshold=params['confidence_threshold'],
                        tweet_chunksize=params['tweet_chunksize'],
                        prediction_batch_size=params['prediction_batch_size'],
                        force=params['force_rerun']
                    )
                    steps_run.add(2)
            elif step == 3:
                if 3 not in steps_run:
                    run_step_3_merge_cleaned_data(force=params['force_rerun'])
                    steps_run.add(3)
            elif step == 4:
                if 4 not in steps_run:
                    run_step_4_generate_embeddings(
                        batch_size=params['batch_size'],
                        use_gpu=params['use_gpu'],
                        test_mode=params['test_mode'],
                        sample_size=params['sample_size'],
                        input_path=get_combined_cleaned_path(),
                        force=params['force_rerun']
                    )
                    steps_run.add(4)
            elif step == 5:
                # Only run complete pipeline if we haven't already run steps 1-4
                if not (1 in steps_run or 2 in steps_run or 3 in steps_run or 4 in steps_run):
                    run_step_5_complete_pipeline(
                        batch_size=params['batch_size'],
                        use_gpu=params['use_gpu'],
                        force_rerun=params['force_rerun']
                    )
                else:
                    logger.warning("Skipping step 5 because steps 1-4 were already run (would cause duplication)")
        
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
        steps_to_run = [1, 2, 3, 4]
    elif args.step5:
        steps_to_run = [5]
    else:
        if args.step1:
            steps_to_run.append(1)
        if args.step2:
            steps_to_run.append(2)
        if args.step3:
            steps_to_run.append(3)
        if args.step4:
            steps_to_run.append(4)
    
    if not steps_to_run:
        print("No steps selected. Use --step1, --step2, --step3, --step4, --step5, or --all")
        return
    
    print(f"Running steps: {steps_to_run}")
    print(f"Force mode: {args.force}")
    
    # Track which steps we've run
    steps_run = set()
    
    # Run selected steps
    for step in steps_to_run:
        if step == 1:
            if 1 not in steps_run:
                run_step_2_clean(use_merged_data=False, force=args.force)
                steps_run.add(1)
        elif step == 2:
            if 2 not in steps_run:
                run_step_2_label_tweets(
                    confidence_threshold=args.confidence_threshold,
                    tweet_chunksize=args.tweet_chunksize,
                    prediction_batch_size=args.prediction_batch_size,
                    force=args.force
                )
                steps_run.add(2)
        elif step == 3:
            if 3 not in steps_run:
                run_step_3_merge_cleaned_data(force=args.force)
                steps_run.add(3)
        elif step == 4:
            if 4 not in steps_run:
                run_step_4_generate_embeddings(
                    batch_size=args.batch_size,
                    use_gpu=not args.no_gpu,
                    test_mode=args.test,
                    sample_size=args.sample_size,
                    input_path=get_combined_cleaned_path(),
                    force=args.force
                )
                steps_run.add(4)
        elif step == 5:
            if not (1 in steps_run or 2 in steps_run or 3 in steps_run or 4 in steps_run):
                run_step_5_complete_pipeline(
                    batch_size=args.batch_size,
                    use_gpu=not args.no_gpu,
                    force_rerun=args.force
                )
            else:
                logger.warning("Skipping step 5 because steps 1-4 were already run")
    
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
  python preprocessing_run.py --step1 --step2 --step3 --step4
  
  # Run the complete preprocessing pipeline (steps 1-4)
  python preprocessing_run.py --all
  
  # Force rerun all steps (overwrite existing files)
  python preprocessing_run.py --all --force
  
  # Run step 5 (complete pipeline with caching checks)
  python preprocessing_run.py --step5
  
  # Generate embeddings only from combined cleaned data
  python preprocessing_run.py --step4 --test --sample-size 5000
  
  # Label tweets only with a custom confidence threshold
  python preprocessing_run.py --step2 --confidence-threshold 0.75
  
  # Merge cleaned tickets and labeled tweets only
  python preprocessing_run.py --step3
        """
    )
    
    # Step selection
    parser.add_argument("--step1", action="store_true", help="Run Step 1: Clean CRM tickets")
    parser.add_argument("--step2", action="store_true", help="Run Step 2: Label tweets")
    parser.add_argument("--step3", action="store_true", help="Run Step 3: Merge cleaned tickets and tweets")
    parser.add_argument("--step4", action="store_true", help="Run Step 4: Generate embeddings from combined cleaned data")
    parser.add_argument("--step5", action="store_true", help="Run Step 5: Complete preprocessing pipeline (steps 1-4 with caching)")
    parser.add_argument("--all", action="store_true", help="Run steps 1-4 (same as --step1 --step2 --step3 --step4)")
    
    # Configuration options
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embeddings")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for embeddings")
    parser.add_argument("--test", action="store_true", help="Run embeddings in test mode with sample")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size for test mode")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="High-confidence threshold for tweet labeling")
    parser.add_argument("--tweet-chunksize", type=int, default=50000, help="Tweet chunk size for labeled processing")
    parser.add_argument("--prediction-batch-size", type=int, default=5000, help="Batch size for tweet predictions")
    parser.add_argument("--force", action="store_true", help="Force rerun all steps even if cached files exist")
    
    # Mode selection
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check if any steps were selected for non-interactive mode
    has_step_selection = args.step1 or args.step2 or args.step3 or args.step4 or args.step5 or args.all
    
    # Run in appropriate mode
    if args.interactive or (not has_step_selection and not args.all):
        # Interactive mode (default if no steps specified)
        run_pipeline_interactive()
    else:
        # Non-interactive mode
        run_pipeline_non_interactive(args)