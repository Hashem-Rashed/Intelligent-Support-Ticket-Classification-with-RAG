"""
BERT / DistilBERT fine-tuning with class weighting and focal loss.
Optimized for RTX A2000 8GB – now with max_length=128 default and gradient checkpointing.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TicketDataset(Dataset):
    """PyTorch dataset for ticket classification."""
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ProgressCallback(TrainerCallback):
    """Callback to report epoch progress."""
    def __init__(self, timer_callback: Callable):
        self.timer_callback = timer_callback
        self.current_epoch = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        self.current_epoch += 1
        if self.timer_callback:
            self.timer_callback(step=self.current_epoch)


class FocalLossTrainer(Trainer):
    """Trainer with focal loss to down-weight easy examples."""
    def __init__(self, gamma=2.0, alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return (focal_loss, outputs) if return_outputs else focal_loss


class WeightedLossTrainer(Trainer):
    """Custom trainer that applies class weights to loss."""
    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class BERTFineTune:
    """
    Fine-tuned transformer model with class weighting and memory optimization.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = None,
        max_length: int = 256,          # reduced from 256 for memory
        learning_rate: float = 3e-5,
        epochs: int = 5,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 2,
        use_gpu: bool = True,
        class_weights: Optional[Dict[int, float]] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

        if self.use_gpu:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"Initializing model: {model_name}")

    def _compute_class_weights(self, y_train_encoded: np.ndarray) -> torch.Tensor:
        classes = np.unique(y_train_encoded)
        weights = np.zeros(len(classes), dtype=np.float32)
        total = len(y_train_encoded)
        for c in classes:
            count = np.sum(y_train_encoded == c)
            weights[c] = total / (len(classes) * count)
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    def fit(
        self,
        X: List[str],
        y: np.ndarray,
        validation_split: float = 0.1,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_labels = len(self.label_encoder.classes_)
        self.classes_ = self.label_encoder.classes_.tolist()
        if self.class_weights is None:
            class_weights_tensor = self._compute_class_weights(y_encoded)
        else:
            class_weights_tensor = torch.tensor(list(self.class_weights.values()), dtype=torch.float32)
        logger.info(f"Class weights: {class_weights_tensor.tolist()}")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, random_state=42, stratify=y_encoded
        )
        logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        # Enable gradient checkpointing to save memory (trade compute for memory)
        self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

        train_dataset = TicketDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = TicketDataset(X_val, y_val, self.tokenizer, self.max_length)
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        logging_steps = max(100, len(train_dataset) // (effective_batch_size * 5))
        total_training_steps = len(train_dataset) * self.epochs // effective_batch_size
        warmup_steps = int(0.1 * total_training_steps)
        training_args = TrainingArguments(
            output_dir="./bert_checkpoints",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_steps=logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            remove_unused_columns=True,
            fp16=self.use_gpu,
            report_to="none",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="weighted")
            return {"accuracy": acc, "f1": f1}
        if self.use_focal_loss:
            trainer = FocalLossTrainer(
                gamma=self.focal_gamma,
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )
        else:
            trainer = WeightedLossTrainer(
                class_weights=class_weights_tensor,
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )
        if progress_callback:
            trainer.add_callback(ProgressCallback(progress_callback))
        logger.info("Starting fine-tuning...")
        trainer.train()
        self.model = trainer.model
        logger.info("Fine-tuning complete")

    def predict(self, X: List[str], batch_size: int = 64) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.eval()
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            encodings = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                encodings = {k: v.to(self.device) for k, v in encodings.items()}
                outputs = self.model(**encodings)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
        return self.label_encoder.inverse_transform(np.array(predictions))

    def predict_proba(self, X: List[str], batch_size: int = 64) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.eval()
        all_probs = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            encodings = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                encodings = {k: v.to(self.device) for k, v in encodings.items()}
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def save(self, filepath: str) -> None:
        save_path = Path(filepath)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        import pickle
        with open(save_path / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        import json
        config_path = save_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}
        config["classes"] = self.classes_
        config["model_name"] = self.model_name
        config["num_labels"] = self.num_labels
        config["max_length"] = self.max_length
        config["class_weights"] = self.class_weights
        config["use_focal_loss"] = self.use_focal_loss
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, filepath: str) -> "BERTFineTune":
        load_path = Path(filepath)
        import pickle, json
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)
        with open(load_path / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        instance = cls(
            model_name=config["model_name"],
            num_labels=config["num_labels"],
            max_length=config.get("max_length", 256),
            class_weights=config.get("class_weights", None),
            use_focal_loss=config.get("use_focal_loss", False),
        )
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        instance.label_encoder = label_encoder
        instance.classes_ = label_encoder.classes_.tolist()
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.model.to(instance.device)
        logger.info(f"Model loaded from {load_path}")
        return instance