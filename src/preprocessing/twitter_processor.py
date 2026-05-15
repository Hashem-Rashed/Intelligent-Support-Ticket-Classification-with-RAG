"""
Twitter data processor - Advanced categorization with ML + keyword fallback.
Now uses ticket data to train a classifier for better accuracy.
Categories: Account, Billing, Fraud, General Inquiry, Technical.
All Unicode symbols replaced with ASCII for Windows console safety.
"""

import pandas as pd
import re
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
from collections import defaultdict
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


# ============================================================
# 1. Keyword-based classifier (fallback & for cold start)
# ============================================================

@dataclass
class MatchResult:
    category: str
    score: float
    matched_terms: List[str] = field(default_factory=list)


class KeywordTweetCategorizer:
    """
    Rule-based categorizer for 5 classes.
    Used as fallback when ML model is not available.
    """

    CATEGORY_KEYWORDS = {
        'Fraud': {
            'primary': ['fraud', 'scam', 'unauthorized', 'stolen', 'hack', 'compromised'],
            'secondary': ['unknown transaction', "wasn't me", "didn't authorize"],
            'tertiary': ['theft', 'impersonation', 'scammer', 'hacker']
        },
        'Billing': {
            'primary': ['refund', 'billing', 'payment', 'invoice', 'subscription', 'chargeback'],
            'secondary': ['overcharged', 'double charge', 'wrong amount'],
            'tertiary': ['receipt', 'transaction', 'statement', 'balance']
        },
        'Technical': {
            'primary': ['crash', 'bug', 'freeze', 'not working', 'broken', 'error'],
            'secondary': ['slow', 'lag', 'delay', 'timeout', 'connection'],
            'tertiary': ['battery drain', 'memory', 'performance', 'compatibility']
        },
        'Account': {
            'primary': ['login', 'password', 'access', 'locked', 'reset password'],
            'secondary': ['verification', '2fa', 'two factor', 'authenticator', 'code'],
            'tertiary': ['profile', 'username', 'email', 'sign up', 'register']
        },
        'General Inquiry': {
            'primary': ['how to', 'what is', 'when will', 'where can', 'question about'],
            'secondary': ['explain', 'clarify', 'understand', 'tell me'],
            'tertiary': ['details', 'guidance', 'tutorial', 'documentation']
        }
    }

    EXACT_PHRASES = {
        'identity theft': 'Fraud',
        'credit card fraud': 'Fraud',
        'unauthorized transaction': 'Fraud',
        'reset password': 'Account',
        'forgot password': 'Account',
        'locked out': 'Account',
        'battery drain': 'Technical',
        'keep crashing': 'Technical',
        'money back': 'Billing',
        'double charge': 'Billing',
        'cancel subscription': 'Billing',
        'how do i': 'General Inquiry'
    }

    NEGATION_WORDS = {'not', 'no', 'never', "don't", "doesn't", "didn't", "won't", "can't"}
    INTENSIFIERS = {'very', 'extremely', 'really', 'so', 'absolutely', 'totally'}

    @classmethod
    def categorize(cls, text: str) -> Tuple[str, float, Dict]:
        text_lower = text.lower()
        # Check exact phrases first
        for phrase, cat in cls.EXACT_PHRASES.items():
            if phrase in text_lower:
                return cat, 0.95, {'matched_phrase': phrase}

        scores = {cat: 0.0 for cat in cls.CATEGORY_KEYWORDS}
        matched_terms = {cat: [] for cat in cls.CATEGORY_KEYWORDS}

        for cat, levels in cls.CATEGORY_KEYWORDS.items():
            for level in ['primary', 'secondary', 'tertiary']:
                weight = 3.0 if level == 'primary' else (2.0 if level == 'secondary' else 1.0)
                for kw in levels.get(level, []):
                    if kw in text_lower:
                        pos = text_lower.find(kw)
                        context = text_lower[max(0, pos-20):pos]
                        negated = any(neg in context for neg in cls.NEGATION_WORDS)
                        if not negated:
                            scores[cat] += weight
                            matched_terms[cat].append(kw)

        best_cat = max(scores, key=lambda c: scores[c])
        best_score = scores[best_cat]

        if best_score == 0:
            return 'General Inquiry', 0.4, {'reason': 'no_keywords'}

        confidence = min(0.5 + (best_score / 10.0), 0.9)
        return best_cat, confidence, {'matched_terms': matched_terms[best_cat][:3]}


# ============================================================
# 2. ML-based classifier (trained on ticket data)
# ============================================================

class MLTweetCategorizer:
    """
    Uses a model trained on ticket data (TF‑IDF + LogReg) to classify tweets.
    Falls back to keyword classifier if model not available.
    """
    def __init__(self, ticket_data_path: Optional[str] = None, model_path: Optional[str] = None):
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_fitted = False

        if model_path and Path(model_path).exists():
            self.load(model_path)
        elif ticket_data_path:
            self.train_from_tickets(ticket_data_path)

    def train_from_tickets(self, ticket_data_path: str):
        """Train TF‑IDF + Logistic Regression on ticket data."""
        logger.info(f"Training ML classifier on tickets from {ticket_data_path}")
        df = pd.read_csv(ticket_data_path)
        if 'clean_text' not in df.columns or 'category' not in df.columns:
            if 'Issue_Category' in df.columns:
                df = df.rename(columns={'Issue_Category': 'category'})
            elif 'category' not in df.columns:
                raise ValueError("Ticket data must have 'clean_text' and 'category' columns")
        target_cats = ['Account', 'Billing', 'Fraud', 'General Inquiry', 'Technical']
        df = df[df['category'].isin(target_cats)]
        X = df['clean_text'].fillna('').tolist()
        y = df['category'].tolist()

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)
        X_tfidf = self.vectorizer.fit_transform(X)

        self.classifier = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', n_jobs=-1)
        self.classifier.fit(X_tfidf, y_encoded)
        self.is_fitted = True
        logger.info(f"ML classifier trained on {len(X)} tickets, {len(self.label_encoder.classes_)} classes")

    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        if not self.is_fitted:
            cats = []
            confs = []
            for t in texts:
                c, conf, _ = KeywordTweetCategorizer.categorize(t)
                cats.append(c)
                confs.append(conf)
            return cats, confs

        X_tfidf = self.vectorizer.transform(texts)
        proba = self.classifier.predict_proba(X_tfidf)
        y_pred_idx = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        categories = self.label_encoder.inverse_transform(y_pred_idx)
        return categories.tolist(), confidences.tolist()

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'label_encoder': self.label_encoder
            }, f)
        logger.info(f"ML classifier saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        self.is_fitted = True
        logger.info(f"ML classifier loaded from {path}")


# ============================================================
# 3. Main processing function with optional ML
# ============================================================

def clean_tweet_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'@\w+\s+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s\.\?\!]', ' ', text)
    text = text.lower()
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
        "can't": "cannot", "isn't": "is not", "aren't": "are not",
        "wasn't": "was not", "weren't": "were not", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not"
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_customer_tweets(df: pd.DataFrame) -> pd.DataFrame:
    original_count = len(df)
    if 'inbound' in df.columns:
        customer_mask = df['inbound'] == True
    else:
        text_col = df.get('text', df.iloc[:, 0])
        customer_mask = (
            ~text_col.astype(str).str.contains(r'^@\w+\s+', na=False) &
            ~text_col.astype(str).str.contains(r'thanks for contacting|our team|please reach out', case=False, na=False)
        )
    customer_tweets = df[customer_mask].copy()
    logger.info(f"  Kept {len(customer_tweets):,} customer tweets ({len(customer_tweets)/original_count*100:.1f}%)")
    return customer_tweets


def process_twitter_data(
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    min_text_length: int = 15,
    sample_size: Optional[int] = None,
    confidence_threshold: float = 0.5,
    use_ml: bool = True,
    ticket_data_path: Optional[str] = None,
    ml_model_path: Optional[str] = None
) -> pd.DataFrame:
    base_dir = Path(settings.PROJECT_ROOT)
    if input_path is None:
        input_path = base_dir / settings.DATA_RAW_PATH / "twcs.csv"
    if output_path is None:
        output_path = base_dir / settings.DATA_PROCESSED_PATH / "tweets_processed.csv"

    logger.info("=" * 70)
    logger.info("TWITTER PROCESSING (with ML classifier if available)")
    logger.info("=" * 70)

    df = pd.read_csv(input_path, low_memory=False, on_bad_lines='skip')
    logger.info(f"Loaded {len(df):,} rows")
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Using sample: {len(df):,} rows")

    df = extract_customer_tweets(df)
    df['clean_text'] = df['text'].astype(str).apply(clean_tweet_text)

    before = len(df)
    df = df[df['clean_text'].str.len() >= min_text_length]
    logger.info(f"Removed {before - len(df)} short tweets")
    before = len(df)
    df = df.drop_duplicates(subset=['clean_text'], keep='first')
    logger.info(f"Removed {before - len(df)} duplicates")

    categorizer = None
    if use_ml:
        try:
            if ml_model_path and Path(ml_model_path).exists():
                categorizer = MLTweetCategorizer(model_path=ml_model_path)
            else:
                if ticket_data_path is None:
                    default_ticket_path = base_dir / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"
                    if default_ticket_path.exists():
                        ticket_data_path = str(default_ticket_path)
                if ticket_data_path and Path(ticket_data_path).exists():
                    categorizer = MLTweetCategorizer(ticket_data_path=ticket_data_path)
                else:
                    logger.warning("No ticket data found for ML training; using keyword fallback.")
                    categorizer = None
        except Exception as e:
            logger.warning(f"ML classifier failed: {e}; falling back to keyword.")
            categorizer = None

    texts = df['clean_text'].tolist()
    if categorizer:
        categories, confidences = categorizer.predict(texts)
    else:
        categories, confidences = [], []
        for t in texts:
            cat, conf, _ = KeywordTweetCategorizer.categorize(t)
            categories.append(cat)
            confidences.append(conf)

    df['category'] = categories
    df['confidence'] = confidences

    high_conf = df[df['confidence'] >= confidence_threshold].copy()
    medium_conf = df[(df['confidence'] >= 0.3) & (df['confidence'] < confidence_threshold)].copy()
    low_conf = df[df['confidence'] < 0.3]

    logger.info(f"\nClassification results:")
    logger.info(f"  High confidence (>={confidence_threshold}): {len(high_conf)}")
    logger.info(f"  Medium confidence: {len(medium_conf)}")
    logger.info(f"  Low confidence: {len(low_conf)}")

    logger.info("\nCategory distribution (high confidence):")
    for cat, count in high_conf['category'].value_counts().items():
        avg_conf = high_conf[high_conf['category'] == cat]['confidence'].mean()
        logger.info(f"  {cat:20s}: {count:5d} (avg conf {avg_conf:.2f})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    high_conf[['clean_text', 'category', 'confidence']].to_csv(output_path, index=False)
    logger.info(f"\nSaved {len(high_conf)} tweets to {output_path}")

    if len(medium_conf) > 0:
        review_path = output_path.parent / "tweets_medium_confidence.csv"
        medium_conf[['clean_text', 'category', 'confidence']].to_csv(review_path, index=False)
        logger.info(f"Saved {len(medium_conf)} medium-confidence tweets to {review_path}")

    if categorizer and categorizer.is_fitted and not ml_model_path:
        model_save_path = base_dir / "models" / "twitter_classifier.pkl"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        categorizer.save(str(model_save_path))
        logger.info(f"Trained ML classifier saved to {model_save_path}")

    return high_conf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int)
    parser.add_argument("--use-ml", action="store_true", default=True)
    parser.add_argument("--ticket-data", type=str)
    parser.add_argument("--ml-model", type=str)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        test_texts = [
            "Someone stole my credit card and made unauthorized purchases!",
            "The app keeps crashing every time I try to open it",
            "How do I reset my password? I'm locked out",
            "Why was I double charged for my subscription?",
            "What are your business hours?",
        ]
        print("\nTesting keyword classifier:")
        for t in test_texts:
            cat, conf, meta = KeywordTweetCategorizer.categorize(t)
            print(f"{t[:50]:50} -> {cat} ({conf:.2f})")
        import sys
        sys.exit(0)
    df_out = process_twitter_data(
        sample_size=args.sample,
        use_ml=args.use_ml,
        ticket_data_path=args.ticket_data,
        ml_model_path=args.ml_model,
        confidence_threshold=args.confidence
    )
    print(f"\nDone. Processed {len(df_out)} tweets.")