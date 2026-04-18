"""
Main preprocessing pipeline runner
Handles: Twitter processing, ticket cleaning, merging, embeddings
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

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.embedding_generator import EmbeddingGenerator
from src.preprocessing.data_merger import merge_datasets
from src.preprocessing.twitter_processor import process_twitter_data
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# In print_menu() function, update the description:

def print_menu():
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE MENU")
    print("="*60)
    print("\nAvailable Pipeline Steps:")
    print("  1. Process Twitter data (8 categories with keyword classification)")
    print("       - Categories: Fraud, Billing, Technical, Account, Delivery")
    print("         Feature Request, Customer Support, General Inquiry")
    print("       - outputs data/processed/tweets_processed.csv")
    print("  2. Clean CRM tickets (5 categories only)")
    print("       - Categories: Account, Billing, Fraud, General Inquiry, Technical")
    print("       - outputs data/processed/tickets_cleaned.csv")
    print("  3. Merge tickets + Twitter data (preserves all categories)")
    print("       - outputs data/processed/merged_support_data.csv")
    print("  4. Generate embeddings from merged data")
    print("  5. Run ALL preprocessing steps (1-4)")
    print("\nOptions:")
    print("  Enter numbers separated by commas (e.g., 1,2,3,4)")
    print("  Enter 'all' to run steps 1-4")
    print("  Enter 'q' to quit")
    print("\n" + "="*60)


def run_step_1_process_twitter(
    confidence_threshold: float = 0.3,
    min_text_length: int = 15,
    sample_size: int = None,
    force: bool = False
):
    """Step 1: Process Twitter data with keyword categorization"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Processing Twitter Data (Keyword Categorization)")
    logger.info("="*60)

    output_path = os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, "tweets_processed.csv")

    if os.path.exists(output_path) and not force:
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping Twitter processing.")
            return True

    try:
        tweets_df = process_twitter_data(
            confidence_threshold=confidence_threshold,
            min_text_length=min_text_length,
            sample_size=sample_size
        )
        logger.info(f"Twitter processing completed. {len(tweets_df)} tweets processed.")
        return True
    except Exception as e:
        logger.error(f"Twitter processing failed: {e}")
        return False


def run_step_2_clean_tickets(force=False):
    """Step 2: Clean CRM tickets"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Cleaning CRM Tickets")
    logger.info("="*60)

    output_path = os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")

    if os.path.exists(output_path) and not force:
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


def run_step_3_merge_data(force=False):
    """Step 3: Merge tickets and Twitter data"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Merging Tickets and Twitter Data")
    logger.info("="*60)

    output_path = os.path.join(settings.PROJECT_ROOT, settings.DATA_PROCESSED_PATH, "merged_support_data.csv")

    if os.path.exists(output_path) and not force:
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


def run_step_4_generate_embeddings(
    batch_size: int = 256,
    use_gpu: bool = True,
    test_mode: bool = False,
    sample_size: int = 1000,
    force: bool = False
):
    """Step 4: Generate embeddings from merged data"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Generating Embeddings")
    logger.info("="*60)

    base_dir = settings.PROJECT_ROOT
    merged_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "merged_support_data.csv")
    tickets_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")

    input_path = merged_path if os.path.exists(merged_path) else tickets_path
    logger.info(f"Using data from: {input_path}")

    output_dir = os.path.join(base_dir, settings.DATA_EMBEDDINGS_PATH)

    if not os.path.exists(input_path):
        logger.error(f"Cleaned data not found: {input_path}")
        logger.info("Please run steps 1-3 first")
        return False

    os.makedirs(output_dir, exist_ok=True)

    embeddings_path = os.path.join(output_dir, "ticket_embeddings.npy")
    if os.path.exists(embeddings_path) and not force and not test_mode:
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
            sample_path = os.path.join(output_dir, "sample.csv")
            df = pd.read_csv(input_path, nrows=sample_size)
            df.to_csv(sample_path, index=False)
            generator.run(input_path=sample_path, output_dir=os.path.join(output_dir, "test"))
            os.remove(sample_path)
        else:
            generator.run(input_path=input_path, output_dir=output_dir)

        logger.info("Embeddings generated successfully")
        return True

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return False


def run_all_steps(
    twitter_confidence: float = 0.3,
    batch_size: int = 256,
    use_gpu: bool = True,
    test_mode: bool = False,
    sample_size: int = 1000,
    force_rerun: bool = False
):
    """Run all preprocessing steps (1-4)"""
    logger.info("\n" + "="*60)
    logger.info("RUNNING ALL PREPROCESSING STEPS (1-4)")
    logger.info("="*60)

    start_time = time.time()

    if not run_step_1_process_twitter(confidence_threshold=twitter_confidence, force=force_rerun):
        logger.error("Step 1 failed. Stopping.")
        return False

    if not run_step_2_clean_tickets(force=force_rerun):
        logger.error("Step 2 failed. Stopping.")
        return False

    if not run_step_3_merge_data(force=force_rerun):
        logger.error("Step 3 failed. Stopping.")
        return False

    if not run_step_4_generate_embeddings(
        batch_size=batch_size, use_gpu=use_gpu,
        test_mode=test_mode, sample_size=sample_size, force=force_rerun
    ):
        logger.error("Step 4 failed. Stopping.")
        return False

    elapsed = time.time() - start_time
    logger.info(f"\nALL STEPS COMPLETED in {elapsed/60:.2f} minutes")
    return True


def parse_choice(choice_str):
    if choice_str.lower() == 'q':
        return None
    if choice_str.lower() == 'all' or choice_str == '5':
        return [1, 2, 3, 4]

    steps = []
    for part in choice_str.split(','):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= 4:
            steps.append(int(part))
        elif '-' in part:
            start, end = map(int, part.split('-'))
            steps.extend(range(start, min(end, 4) + 1))
    return sorted(set(steps))


def get_params():
    print("\n" + "="*60)
    print("PIPELINE CONFIGURATION")
    print("="*60)

    twitter_conf = input("\nTwitter confidence threshold [0.3]: ").strip()
    twitter_conf = float(twitter_conf) if twitter_conf else 0.3

    use_gpu = input("\nUse GPU for embeddings? (y/n) [y]: ").lower().strip() != 'n'
    batch_size = int(input("Batch size [256]: ").strip() or 256)
    test_mode = input("Test mode? (y/n) [n]: ").lower().strip() == 'y'
    sample_size = int(input("Sample size [1000]: ").strip() or 1000) if test_mode else 1000
    force_rerun = input("\nForce rerun? (y/n) [n]: ").lower().strip() == 'y'

    return {
        'twitter_confidence': twitter_conf,
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

        if choice.lower() == 'all' or choice == '5':
            run_all_steps(**params)
        else:
            start = time.time()
            for step in steps:
                if step == 1:
                    run_step_1_process_twitter(confidence_threshold=params['twitter_confidence'], force=params['force_rerun'])
                elif step == 2:
                    run_step_2_clean_tickets(force=params['force_rerun'])
                elif step == 3:
                    run_step_3_merge_data(force=params['force_rerun'])
                elif step == 4:
                    run_step_4_generate_embeddings(
                        batch_size=params['batch_size'], use_gpu=params['use_gpu'],
                        test_mode=params['test_mode'], sample_size=params['sample_size'],
                        force=params['force_rerun']
                    )
            print(f"\nCompleted in {(time.time()-start)/60:.2f} minutes")
            input("\nPress Enter to continue...")


def run_non_interactive(args):
    if args.all:
        run_all_steps(
            twitter_confidence=args.twitter_confidence,
            batch_size=args.batch_size,
            use_gpu=not args.no_gpu,
            test_mode=args.test,
            sample_size=args.sample_size,
            force_rerun=args.force
        )
    else:
        if args.step1:
            run_step_1_process_twitter(confidence_threshold=args.twitter_confidence, force=args.force)
        if args.step2:
            run_step_2_clean_tickets(force=args.force)
        if args.step3:
            run_step_3_merge_data(force=args.force)
        if args.step4:
            run_step_4_generate_embeddings(
                batch_size=args.batch_size, use_gpu=not args.no_gpu,
                test_mode=args.test, sample_size=args.sample_size, force=args.force
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Pipeline Runner")
    parser.add_argument("--step1", action="store_true", help="Process Twitter data")
    parser.add_argument("--step2", action="store_true", help="Clean CRM tickets")
    parser.add_argument("--step3", action="store_true", help="Merge datasets")
    parser.add_argument("--step4", action="store_true", help="Generate embeddings")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--twitter-confidence", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    if args.interactive or (not any([args.step1, args.step2, args.step3, args.step4, args.all])):
        run_interactive()
    else:
        run_non_interactive(args)