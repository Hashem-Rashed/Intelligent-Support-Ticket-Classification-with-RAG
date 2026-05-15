"""
Main preprocessing pipeline runner
Handles: Ticket cleaning, ML training, Twitter processing, merging, embeddings
Uses only 5 categories: Account, Billing, Fraud, General Inquiry, Technical.
Steps are numbered in logical order.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.embedding_generator import EmbeddingGenerator
from src.preprocessing.data_merger import merge_datasets
from src.preprocessing.twitter_processor import process_twitter_data, MLTweetCategorizer
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_menu():
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE (5 Categories: Account, Billing, Fraud, Technical, General Inquiry)")
    print("="*70)
    print("\nSteps in recommended order:")
    print("  1. Clean CRM tickets (5 categories only)")
    print("       - outputs data/processed/tickets_cleaned.csv")
    print("  2. Train ML classifier on tickets (save model)")
    print("       - uses tickets_cleaned.csv, outputs models/twitter_classifier.pkl")
    print("  3. Process Twitter data (uses ML if available)")
    print("       - outputs data/processed/tweets_processed.csv")
    print("  4. Merge tickets + Twitter data (preserves 5 categories)")
    print("       - outputs data/processed/merged_support_data.csv")
    print("  5. Generate embeddings from merged data")
    print("       - outputs data/embeddings/ticket_embeddings.npy")
    print("  6. Run ALL preprocessing steps (1-5 in order)")
    print("\nOptions:")
    print("  Enter numbers separated by commas (e.g., 1,2,3,4,5)")
    print("  Enter 'all' to run steps 1-5 in order")
    print("  Enter 'q' to quit")
    print("\n" + "="*70)


def step1_clean_tickets(force=False):
    """Step 1: Clean CRM tickets (5 categories only)"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Cleaning CRM Tickets (5 categories)")
    logger.info("="*60)

    output_path = Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"

    if output_path.exists() and not force:
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping ticket cleaning.")
            return True

    try:
        run_pipeline(use_merged_data=False)
        logger.info("Ticket cleaning completed")
        return True
    except Exception as e:
        logger.error(f"Ticket cleaning failed: {e}")
        return False


def step2_train_ml_classifier(ticket_data_path=None, output_model_path=None, force=False):
    """Step 2: Train ML classifier on tickets and save model."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Train ML Classifier on Tickets")
    logger.info("="*60)

    if ticket_data_path is None:
        ticket_data_path = Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"
    else:
        ticket_data_path = Path(ticket_data_path)

    if not ticket_data_path.exists():
        logger.error(f"Ticket data not found: {ticket_data_path}")
        logger.info("Please run step 1 (clean tickets) first.")
        return False

    if output_model_path is None:
        output_model_path = Path(settings.PROJECT_ROOT) / "models" / "twitter_classifier.pkl"
    else:
        output_model_path = Path(output_model_path)

    if output_model_path.exists() and not force:
        response = input(f"Model {output_model_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping ML training.")
            return True

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        classifier = MLTweetCategorizer(ticket_data_path=str(ticket_data_path))
        classifier.save(str(output_model_path))
        logger.info(f"ML classifier saved to {output_model_path}")
        return True
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        return False


def step3_process_twitter(
    confidence_threshold: float = 0.55,
    min_text_length: int = 15,
    sample_size: int = None,
    force: bool = False,
    use_ml: bool = True,
    ticket_data_path: str = None,
    ml_model_path: str = None
):
    """Step 3: Process Twitter data using ML (or keyword fallback)"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Processing Twitter Data (5 categories)")
    logger.info("="*60)

    output_path = Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / "tweets_processed.csv"

    if output_path.exists() and not force:
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping Twitter processing.")
            return True

    # If ML is enabled, try to use pre-trained model or ticket data
    if use_ml:
        if ml_model_path is None:
            default_model = Path(settings.PROJECT_ROOT) / "models" / "twitter_classifier.pkl"
            if default_model.exists():
                ml_model_path = str(default_model)
                logger.info(f"Using pre-trained ML model from {ml_model_path}")
        if ml_model_path is None:
            if ticket_data_path is None:
                default_ticket_path = Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"
                if default_ticket_path.exists():
                    ticket_data_path = str(default_ticket_path)
                    logger.info(f"Will train ML model on tickets from {ticket_data_path}")
                else:
                    logger.warning("No ticket data found; ML classifier will not be trained. Falling back to keyword rules.")
                    use_ml = False

    try:
        tweets_df = process_twitter_data(
            confidence_threshold=confidence_threshold,
            min_text_length=min_text_length,
            sample_size=sample_size,
            use_ml=use_ml,
            ticket_data_path=ticket_data_path,
            ml_model_path=ml_model_path
        )
        logger.info(f"Twitter processing completed. {len(tweets_df)} tweets processed.")
        return True
    except Exception as e:
        logger.error(f"Twitter processing failed: {e}")
        return False


def step4_merge_data(force=False):
    """Step 4: Merge tickets and Twitter data (5 categories only)"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Merging Tickets and Twitter Data (5 categories)")
    logger.info("="*60)

    output_path = Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / "merged_support_data.csv"

    if output_path.exists() and not force:
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping data merge.")
            return True

    try:
        merge_datasets()
        logger.info("Data merge completed")
        return True
    except Exception as e:
        logger.error(f"Data merge failed: {e}")
        return False


def step5_generate_embeddings(
    batch_size: int = 256,
    use_gpu: bool = True,
    test_mode: bool = False,
    sample_size: int = 1000,
    force: bool = False
):
    """Step 5: Generate embeddings from merged data"""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Generating Embeddings")
    logger.info("="*60)

    base_dir = Path(settings.PROJECT_ROOT)
    merged_path = base_dir / settings.DATA_PROCESSED_PATH / "merged_support_data.csv"
    tickets_path = base_dir / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"

    input_path = merged_path if merged_path.exists() else tickets_path
    logger.info(f"Using data from: {input_path}")

    output_dir = base_dir / settings.DATA_EMBEDDINGS_PATH

    if not input_path.exists():
        logger.error(f"Cleaned data not found: {input_path}")
        logger.info("Please run steps 1-4 first")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "ticket_embeddings.npy"
    if embeddings_path.exists() and not force and not test_mode:
        logger.info(f"Embeddings already exist at {embeddings_path}")
        response = input("Overwrite existing embeddings? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping embedding generation.")
            return True

    try:
        generator = EmbeddingGenerator(
            model_name=settings.MODEL_NAME,
            batch_size=batch_size if not test_mode else min(batch_size, 128),
            use_gpu=use_gpu
        )

        if test_mode:
            logger.info(f"TEST MODE: Running with {sample_size:,} samples")
            import pandas as pd
            sample_path = output_dir / "sample.csv"
            df = pd.read_csv(input_path, nrows=sample_size)
            df.to_csv(sample_path, index=False)
            generator.run(input_path=str(sample_path), output_dir=str(output_dir / "test"))
            sample_path.unlink()
        else:
            generator.run(input_path=str(input_path), output_dir=str(output_dir))

        logger.info("Embeddings generated successfully")
        return True
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return False


def step6_run_all(
    twitter_confidence: float = 0.55,
    batch_size: int = 256,
    use_gpu: bool = True,
    test_mode: bool = False,
    sample_size: int = 1000,
    force_rerun: bool = False,
    use_ml: bool = True
):
    """Run all steps 1-5 in correct order."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING ALL STEPS (1 -> 2 -> 3 -> 4 -> 5)")
    logger.info("="*60)

    start_time = time.time()

    if not step1_clean_tickets(force=force_rerun):
        logger.error("Step 1 failed. Stopping.")
        return False

    if not step2_train_ml_classifier(force=force_rerun):
        logger.error("Step 2 failed. Stopping.")
        return False

    if not step3_process_twitter(
        confidence_threshold=twitter_confidence,
        force=force_rerun,
        use_ml=use_ml
    ):
        logger.error("Step 3 failed. Stopping.")
        return False

    if not step4_merge_data(force=force_rerun):
        logger.error("Step 4 failed. Stopping.")
        return False

    if not step5_generate_embeddings(
        batch_size=batch_size, use_gpu=use_gpu,
        test_mode=test_mode, sample_size=sample_size, force=force_rerun
    ):
        logger.error("Step 5 failed. Stopping.")
        return False

    elapsed = time.time() - start_time
    logger.info(f"\nALL STEPS COMPLETED in {elapsed/60:.2f} minutes")
    return True


def parse_choice(choice_str):
    if choice_str.lower() == 'q':
        return None
    if choice_str.lower() == 'all' or choice_str == '6':
        return [1, 2, 3, 4, 5]   # enforce correct order
    steps = []
    for part in choice_str.split(','):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= 5:
            steps.append(int(part))
        elif '-' in part:
            start, end = map(int, part.split('-'))
            steps.extend(range(start, min(end, 5) + 1))
    return sorted(set(steps))


def get_params():
    print("\n" + "="*60)
    print("PIPELINE CONFIGURATION (5 categories)")
    print("="*60)

    twitter_conf = input("\nTwitter confidence threshold [0.55]: ").strip()
    twitter_conf = float(twitter_conf) if twitter_conf else 0.55

    use_ml = input("\nUse ML classifier for tweets? (y/n) [y]: ").lower().strip() != 'n'

    use_gpu = input("\nUse GPU for embeddings? (y/n) [y]: ").lower().strip() != 'n'
    batch_size = int(input("Batch size [256]: ").strip() or 256)
    test_mode = input("Test mode? (y/n) [n]: ").lower().strip() == 'y'
    sample_size = int(input("Sample size [1000]: ").strip() or 1000) if test_mode else 1000
    force_rerun = input("\nForce rerun? (y/n) [n]: ").lower().strip() == 'y'

    return {
        'twitter_confidence': twitter_conf,
        'use_ml': use_ml,
        'use_gpu': use_gpu,
        'batch_size': batch_size,
        'test_mode': test_mode,
        'sample_size': sample_size,
        'force_rerun': force_rerun,
    }


def run_interactive():
    params = get_params()

    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()

        steps = parse_choice(choice)
        if steps is None:
            print("\nExiting. Goodbye!")
            break
        if not steps:
            print("Invalid choice.")
            continue

        # Handle "all" (choice 6) separately
        if choice.lower() == 'all' or choice == '6':
            step6_run_all(**params)
        else:
            start = time.time()
            for step in steps:
                if step == 1:
                    step1_clean_tickets(force=params['force_rerun'])
                elif step == 2:
                    step2_train_ml_classifier(force=params['force_rerun'])
                elif step == 3:
                    step3_process_twitter(
                        confidence_threshold=params['twitter_confidence'],
                        use_ml=params['use_ml'],
                        force=params['force_rerun']
                    )
                elif step == 4:
                    step4_merge_data(force=params['force_rerun'])
                elif step == 5:
                    step5_generate_embeddings(
                        batch_size=params['batch_size'],
                        use_gpu=params['use_gpu'],
                        test_mode=params['test_mode'],
                        sample_size=params['sample_size'],
                        force=params['force_rerun']
                    )
            print(f"\nCompleted in {(time.time()-start)/60:.2f} minutes")
            input("\nPress Enter to continue...")


def run_non_interactive(args):
    if args.all:
        step6_run_all(
            twitter_confidence=args.twitter_confidence,
            batch_size=args.batch_size,
            use_gpu=not args.no_gpu,
            test_mode=args.test,
            sample_size=args.sample_size,
            force_rerun=args.force,
            use_ml=not args.no_ml
        )
    else:
        if args.step1:
            step1_clean_tickets(force=args.force)
        if args.step2:
            step2_train_ml_classifier(force=args.force)
        if args.step3:
            step3_process_twitter(
                confidence_threshold=args.twitter_confidence,
                use_ml=not args.no_ml,
                force=args.force
            )
        if args.step4:
            step4_merge_data(force=args.force)
        if args.step5:
            step5_generate_embeddings(
                batch_size=args.batch_size,
                use_gpu=not args.no_gpu,
                test_mode=args.test,
                sample_size=args.sample_size,
                force=args.force
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Pipeline Runner (5 categories)")
    parser.add_argument("--step1", action="store_true", help="Clean CRM tickets")
    parser.add_argument("--step2", action="store_true", help="Train ML classifier on tickets")
    parser.add_argument("--step3", action="store_true", help="Process Twitter data")
    parser.add_argument("--step4", action="store_true", help="Merge datasets")
    parser.add_argument("--step5", action="store_true", help="Generate embeddings")
    parser.add_argument("--all", action="store_true", help="Run steps 1-5 in order")
    parser.add_argument("--twitter-confidence", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML classifier for tweets (use keyword)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    if args.interactive or (not any([args.step1, args.step2, args.step3, args.step4, args.step5, args.all])):
        run_interactive()
    else:
        run_non_interactive(args)