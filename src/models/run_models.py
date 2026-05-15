"""
Unified Model Training Runner – Fixed for 8GB GPU and 5 categories.
Now supports non‑interactive mode and memory‑safe transformer training.
"""
import os
import sys
import time
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingTimer:
    def __init__(self, total_steps=None, description="Training"):
        self.start_time = time.time()
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0

    def update(self, step=None, increment=1):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += increment
        elapsed = time.time() - self.start_time
        if self.total_steps and self.current_step > 0:
            remaining = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f", ETA: {remaining:.0f}s"
        else:
            eta_str = ""
        sys.stdout.write(f"\r{self.description}: {elapsed:.1f}s elapsed{eta_str}     ")
        sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.start_time
        sys.stdout.write(f"\r{self.description}: {elapsed:.1f}s elapsed (done)     \n")
        sys.stdout.flush()


def load_data(data_path=None):
    if data_path is None:
        data_path = project_root / settings.DATA_PROCESSED_PATH / "merged_support_data.csv"
    else:
        data_path = Path(data_path)
    print(f"Loading data from {data_path}...", flush=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}\nRun preprocessing steps first.")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} rows from {data_path}")
    return df


def balance_dataset(df, target_col='category', samples_per_class=None, random_state=42):
    if samples_per_class is None or samples_per_class <= 0:
        logger.info("No balancing applied.")
        return df
    logger.info(f"Balancing dataset to {samples_per_class} samples per class")
    balanced_dfs = []
    for cat in df[target_col].unique():
        cat_df = df[df[target_col] == cat]
        if len(cat_df) >= samples_per_class:
            sampled = cat_df.sample(n=samples_per_class, random_state=random_state)
        else:
            sampled = cat_df.sample(n=samples_per_class, replace=True, random_state=random_state)
        balanced_dfs.append(sampled)
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    logger.info(f"Balanced dataset size: {len(balanced_df):,}")
    return balanced_df


def prepare_train_test(df, text_col='clean_text', label_col='category', test_size=0.2, random_state=42):
    X = df[text_col].astype(str).tolist()
    y = df[label_col].values
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=random_state, stratify=y_train_full
    )
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def find_best_thresholds(model, X_val, y_val, classes_to_tune=None):
    if classes_to_tune is None:
        classes_to_tune = ['Fraud']
    proba = model.predict_proba(X_val)
    y_val_encoded = model.label_encoder.transform(y_val)
    class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}
    best_thresholds = {}
    for cls in classes_to_tune:
        if cls not in class_indices:
            continue
        cls_idx = class_indices[cls]
        cls_proba = proba[:, cls_idx]
        best_f1 = 0
        best_thr = 0.5
        for thr in np.arange(0.5, 0.95, 0.05):
            y_pred_cls = (cls_proba >= thr).astype(int)
            y_true_cls = (y_val_encoded == cls_idx).astype(int)
            f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        best_thresholds[cls] = best_thr
        logger.info(f"Best threshold for {cls}: {best_thr:.2f} (F1={best_f1:.3f})")
    for cls in model.classes_:
        if cls not in best_thresholds:
            best_thresholds[cls] = 0.5
    return best_thresholds


def train_baseline(X_train, y_train, output_dir, max_features=15000, C=1.0, max_df=0.5, min_df=2,
                   ngram_range=(1,4), use_smote=False):
    from src.models.baseline.tfidf_logreg import TFIDFLogReg
    logger.info("Training TF-IDF + Logistic Regression...")
    timer = TrainingTimer(description="Baseline training")
    model = TFIDFLogReg(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        C=C,
        class_weight='balanced',
    )
    X_train_tfidf = model.vectorizer.fit_transform(X_train)
    y_train_encoded = model.label_encoder.fit_transform(y_train)
    model.classes_ = model.label_encoder.classes_.tolist()

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from collections import Counter
            class_counts = Counter(y_train_encoded)
            fraud_idx = model.label_encoder.transform(['Fraud'])[0]
            target_dict = {}
            if class_counts[fraud_idx] < 30000:
                target_dict[fraud_idx] = 30000
            if target_dict:
                sm = SMOTE(sampling_strategy=target_dict, random_state=42)
                X_train_tfidf, y_train_encoded = sm.fit_resample(X_train_tfidf, y_train_encoded)
                logger.info(f"After SMOTE: X shape {X_train_tfidf.shape}")
            else:
                logger.info("SMOTE skipped – Fraud already has >=30000 samples.")
        except ImportError:
            logger.warning("imbalanced-learn not installed, skipping SMOTE.")

    model.classifier.fit(X_train_tfidf, y_train_encoded)
    model.is_fitted = True
    timer.finish()
    train_time = time.time() - timer.start_time
    output_path = Path(output_dir) / "tfidf_logreg_model.pkl"
    model.save(str(output_path))
    logger.info(f"Model saved to {output_path}")
    return model, train_time


def train_transformer(X_train, y_train, output_dir, model_name='distilbert-base-uncased',
                      epochs=3, batch_size=2, gradient_accumulation=8, max_length=96):
    from src.models.transformer.bert_finetune import BERTFineTune
    import torch
    # Force safe values for 8GB GPU
    safe_batch = 2
    safe_grad_acc = 8
    safe_len = 96
    logger.info(f"Using memory‑safe settings: batch_size={safe_batch}, grad_acc={safe_grad_acc}, max_len={safe_len}, epochs={epochs}")
    timer = TrainingTimer(description=f"Transformer ({epochs} epochs)", total_steps=epochs)
    model = BERTFineTune(
        model_name=model_name,
        epochs=epochs,
        batch_size=safe_batch,
        gradient_accumulation_steps=safe_grad_acc,
        use_gpu=True,
        max_length=safe_len,
        learning_rate=3e-5,
    )
    start = time.time()
    def epoch_callback(step):
        timer.update(step=step)
    model.fit(X_train, y_train, progress_callback=epoch_callback)
    timer.finish()
    train_time = time.time() - start
    output_path = Path(output_dir)
    model.save(str(output_path))
    logger.info(f"Model saved to {output_path}")
    return model, train_time


def evaluate_model(model, X_test, y_test, save_dir=None, apply_thresholds=True, model_type="baseline"):
    from src.models.evaluation import evaluate_model as eval_func, plot_confusion_matrix
    if model_type == "baseline" and apply_thresholds:
        y_pred = model.predict(X_test, apply_thresholds=apply_thresholds)
    else:
        y_pred = model.predict(X_test)
    metrics = eval_func(y_test, y_pred)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, 'classes_'):
            labels = model.classes_
        elif hasattr(model, 'label_encoder'):
            labels = model.label_encoder.classes_.tolist()
        else:
            labels = sorted(set(y_test))
        plot_confusion_matrix(y_test, y_pred, labels=labels,
                              save_path=save_dir / "confusion_matrix.png")
    return metrics, y_pred


def compute_auto_samples_per_class(df, target_col='category', model_type='baseline', transformer_max=10000):
    class_counts = df[target_col].value_counts()
    min_class_size = class_counts.min()
    if model_type == 'transformer':
        auto_samples = min(min_class_size, transformer_max)
        logger.info(f"Smallest class: {min_class_size}, capped at {transformer_max} -> using {auto_samples} per class.")
    else:
        auto_samples = min_class_size
        logger.info(f"Smallest class: {min_class_size} -> using {auto_samples} per class.")
    return auto_samples


def run_model_step(data_path, balance, samples_per_class, test_size, model_type,
                   output_dir, force_retrain, auto_limit, transformer_max=10000,
                   use_smote=False, **kwargs):
    df = load_data(data_path)
    if auto_limit and balance:
        samples_per_class = compute_auto_samples_per_class(df, model_type=model_type, transformer_max=transformer_max)
    if model_type == 'transformer' and not balance and len(df) > 100000:
        logger.warning(f"Full dataset size {len(df)} is large; auto-enabling balancing to avoid OOM.")
        balance = True
        if samples_per_class is None or samples_per_class > transformer_max:
            samples_per_class = transformer_max
        df = balance_dataset(df, samples_per_class=samples_per_class)
    elif balance:
        df = balance_dataset(df, samples_per_class=samples_per_class)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test(df, test_size=test_size)
    output_path = Path(output_dir)
    if not force_retrain and output_path.exists():
        if model_type == 'baseline':
            model_file = output_path / "tfidf_logreg_model.pkl"
        else:
            model_file = output_path / "config.json"
        if model_file.exists():
            resp = input(f"Model already exists at {model_file}. Overwrite? (y/n): ").lower().strip()
            if resp != 'y':
                logger.info("Skipping training.")
                return None, None

    if model_type == 'baseline':
        model, train_time = train_baseline(
            X_train, y_train, output_dir,
            max_features=kwargs.get('max_features', 15000),
            C=kwargs.get('C', 1.0),
            max_df=kwargs.get('max_df', 0.5),
            min_df=kwargs.get('min_df', 2),
            ngram_range=kwargs.get('ngram_range', (1,4)),
            use_smote=use_smote
        )
        logger.info("Tuning per-class thresholds on validation set...")
        best_thresholds = find_best_thresholds(model, X_val, y_val, classes_to_tune=['Fraud'])
        model.set_thresholds(best_thresholds)
    elif model_type == 'transformer':
        try:
            model, train_time = train_transformer(
                X_train, y_train, output_dir,
                model_name=kwargs.get('model_name', 'distilbert-base-uncased'),
                epochs=kwargs.get('epochs', 3),
                batch_size=kwargs.get('batch_size', 2),
                gradient_accumulation=kwargs.get('gradient_accumulation', 8),
                max_length=kwargs.get('max_length', 96)
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.error(f"Transformer training failed due to GPU memory: {e}")
                logger.info("Falling back to baseline model only. Skipping transformer.")
                return None, None
            else:
                raise
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    metrics, y_pred = evaluate_model(model, X_test, y_test, save_dir=output_dir,
                                     apply_thresholds=True, model_type=model_type)
    logger.info(f"Evaluation metrics: {metrics}")
    metrics_path = output_path / f"{model_type}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_type': model_type,
            'train_time_sec': train_time,
            'test_size': len(X_test),
            **metrics,
        }, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    return model, metrics


def print_menu():
    print("\n" + "="*70)
    print("MODEL TRAINING MENU (Optimized for RTX A2000 8GB) – 5 categories")
    print("="*70)
    print("\nAvailable Models:")
    print("  1. TF-IDF + Logistic Regression (fast, CPU)")
    print("  2. Transformer (DistilBERT) - memory‑safe (GPU, 8GB)")
    print("  3. Both (run both sequentially)")
    print("\nOptions:")
    print("  Enter numbers separated by commas (e.g., 1,2)")
    print("  Enter 'all' to run all models")
    print("  Enter 'q' to quit")
    print("\n" + "="*70)


def get_default_params():
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION (memory‑safe defaults for transformer)")
    print("="*70)
    balance = input("\nBalance dataset? (y/n) [y]: ").lower().strip() != 'n'
    auto_limit = False
    samples = None
    transformer_max = 15000
    if balance:
        auto_limit = input("Auto‑limit samples per class based on data? (y/n) [y]: ").lower().strip() != 'n'
        if auto_limit:
            transformer_max = int(input("Transformer max samples per class (cap) [15000]: ").strip() or 15000)
        else:
            samples = int(input("Samples per class (0 = no balancing) [25000]: ").strip() or 25000)
            if samples == 0:
                balance = False
    test_size = float(input("Test size fraction [0.2]: ").strip() or 0.2)
    force = input("Force retrain (overwrite existing)? (y/n) [n]: ").lower().strip() == 'y'
    use_smote = input("Apply SMOTE oversampling for Fraud (baseline only)? (y/n) [n]: ").lower().strip() == 'y'

    max_features = int(input("TF-IDF max features [15000]: ").strip() or 15000)
    C = float(input("Logistic Regression C (regularization) [1.0]: ").strip() or 1.0)

    # Transformer settings are forced to safe values
    transformer_model = "distilbert-base-uncased"
    transformer_epochs = 3
    transformer_batch = 2
    transformer_grad_acc = 8
    print(f"\nTransformer will use: epochs={transformer_epochs}, batch_size={transformer_batch}, grad_acc={transformer_grad_acc} (memory‑safe).")

    return {
        'balance': balance,
        'samples_per_class': samples,
        'auto_limit': auto_limit,
        'transformer_max': transformer_max,
        'test_size': test_size,
        'force_retrain': force,
        'use_smote': use_smote,
        'max_features': max_features,
        'C': C,
        'transformer_model': transformer_model,
        'transformer_epochs': transformer_epochs,
        'transformer_batch': transformer_batch,
        'transformer_grad_acc': transformer_grad_acc,
    }


def run_interactive():
    params = get_default_params()
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip().lower()
        if choice == 'q':
            print("Exiting. Goodbye!")
            break
        if choice == 'all' or choice == '3':
            models_to_run = ['baseline', 'transformer']
        else:
            mapping = {'1': 'baseline', '2': 'transformer'}
            models_to_run = [mapping.get(c.strip()) for c in choice.split(',') if c.strip() in mapping]
        if not models_to_run:
            print("Invalid choice.")
            continue

        base_output = project_root / "models" / "saved"
        for model_type in models_to_run:
            output_dir = base_output / model_type
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"\n{'='*70}")
            logger.info(f"RUNNING MODEL: {model_type.upper()}")
            logger.info(f"{'='*70}")
            kwargs = {
                'max_features': params['max_features'],
                'C': params['C'],
                'model_name': params['transformer_model'],
                'epochs': params['transformer_epochs'],
                'batch_size': params['transformer_batch'],
                'gradient_accumulation': params['transformer_grad_acc'],
                'max_df': 0.5,
                'min_df': 2,
                'ngram_range': (1,4),
                'max_length': 256,
            }
            run_model_step(
                data_path=None,
                balance=params['balance'],
                samples_per_class=params['samples_per_class'],
                test_size=params['test_size'],
                model_type=model_type,
                output_dir=output_dir,
                force_retrain=params['force_retrain'],
                auto_limit=params['auto_limit'],
                transformer_max=params['transformer_max'],
                use_smote=params['use_smote'],
                **kwargs,
            )
        input("\nPress Enter to continue...")


def run_non_interactive(args):
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / "saved" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.auto_limit:
        samples_per_class = None
    else:
        samples_per_class = args.samples_per_class
    kwargs = {
        'max_features': args.max_features,
        'C': args.C,
        'model_name': args.transformer_model,
        'epochs': args.epochs if args.epochs else 3,
        'batch_size': args.batch_size if args.batch_size else 2,
        'gradient_accumulation': args.gradient_accumulation if args.gradient_accumulation else 8,
        'max_df': 0.5,
        'min_df': 2,
        'ngram_range': (1,4),
        'max_length': 96,
    }
    run_model_step(
        data_path=args.data_path,
        balance=args.balance,
        samples_per_class=samples_per_class,
        test_size=args.test_size,
        model_type=args.model,
        output_dir=output_dir,
        force_retrain=args.force,
        auto_limit=args.auto_limit,
        transformer_max=args.transformer_max if args.transformer_max else 15000,
        use_smote=args.use_smote,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Runner")
    parser.add_argument("--model", type=str, choices=['baseline', 'transformer'], help="Model to train")
    parser.add_argument("--data-path", type=str, help="Path to merged CSV")
    parser.add_argument("--balance", action="store_true", help="Balance dataset")
    parser.add_argument("--samples-per-class", type=int, default=25000, help="Samples per class if balancing")
    parser.add_argument("--auto-limit", action="store_true", help="Auto‑cap samples to smallest class or transformer_max")
    parser.add_argument("--transformer-max", type=int, default=15000, help="Max samples per class for transformer")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE for Fraud (baseline)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--max-features", type=int, default=15000, help="TF-IDF max features")
    parser.add_argument("--C", type=float, default=1.0, help="Logistic Regression C")
    parser.add_argument("--transformer-model", type=str, default="distilbert-base-uncased", help="Transformer model name")
    parser.add_argument("--epochs", type=int, default=3, help="Transformer epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Transformer batch size (memory‑safe default=2)")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--output-dir", type=str, help="Output directory for model")
    parser.add_argument("--force", action="store_true", help="Overwrite existing model")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode")
    args = parser.parse_args()

    if args.interactive or (not args.model):
        run_interactive()
    else:
        run_non_interactive(args)