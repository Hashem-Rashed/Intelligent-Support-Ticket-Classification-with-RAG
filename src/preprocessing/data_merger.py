import os
import pandas as pd
import re
from sklearn.utils import resample
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =========================
# 🔥 TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# =========================
# 🔥 ADVANCED CATEGORIZATION
# =========================
def categorize_tweet(text):
    """Advanced categorization with imbalance handling"""

    if pd.isna(text):
        return "General Inquiry"

    text = clean_text(text)

    categories = {
        "Fraud": {"keywords": ["fraud", "scam", "unauthorized", "stolen", "phishing"], "weight": 2.0},
        "Billing": {"keywords": ["refund", "charge", "billing", "payment", "invoice"], "weight": 1.6},
        "Technical": {"keywords": ["error", "bug", "crash", "not working", "failed"], "weight": 1.3},
        "Account": {"keywords": ["login", "password", "account", "signin"], "weight": 1.3},
        "Delivery": {"keywords": ["delivery", "shipping", "package"], "weight": 1.2},
        "Data & Privacy": {"keywords": ["privacy", "data breach", "gdpr"], "weight": 1.8},
        "Integration & API": {"keywords": ["api", "integration", "webhook"], "weight": 1.5},
        "Mobile App": {"keywords": ["android", "ios", "app"], "weight": 1.1},
        "Feature Request": {"keywords": ["feature request", "suggest", "improvement"], "weight": 1.0},
        "Customer Support": {"keywords": ["help", "support", "assist"], "weight": 0.7},
        "General Inquiry": {"keywords": ["how", "what", "question"], "weight": 0.4},
    }

    # 🔴 Priority rules
    if any(k in text for k in ["fraud", "scam", "unauthorized", "stolen"]):
        return "Fraud"

    if any(k in text for k in ["refund", "charged", "billing issue"]):
        return "Billing"

    scores = {}

    for category, info in categories.items():
        score = 0

        for keyword in info["keywords"]:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            matches = len(re.findall(pattern, text))
            if matches > 0:
                score += matches * info["weight"]

        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)

    return "General Inquiry"


# =========================
# 🔥 CATEGORY CHECK
# =========================
def should_categorize(category_value):
    if pd.isna(category_value):
        return True

    category_str = str(category_value).strip().lower()

    generic_categories = [
        "customer inquiry", "general", "general inquiry",
        "unknown", "other", "nan", "none", "", "inquiry"
    ]

    return category_str in generic_categories


# =========================
# 🔥 DATA BALANCING (NEW 🔥)
# =========================
def balance_dataset(df, target_col="category", max_per_class=100000):
    """Balance dataset using hybrid sampling"""

    logger.info("Balancing dataset...")

    balanced_dfs = []

    for category in df[target_col].unique():
        df_cat = df[df[target_col] == category]

        if len(df_cat) > max_per_class:
            # 🔻 Undersample large classes
            df_cat = df_cat.sample(max_per_class, random_state=42)
        else:
            # 🔺 Oversample small classes
            df_cat = resample(
                df_cat,
                replace=True,
                n_samples=min(max_per_class, len(df_cat) * 3),
                random_state=42
            )

        balanced_dfs.append(df_cat)

    balanced_df = pd.concat(balanced_dfs)

    logger.info("Balanced distribution:\n{}".format(
        balanced_df[target_col].value_counts()
    ))

    return balanced_df


# =========================
# 🚀 MAIN PIPELINE
# =========================
def merge_datasets(
    tickets_path=None,
    tweets_path=None,
    output_path=None,
    categorize_tweets=True,
    overwrite_existing_categories=False,
    balance_data=False   # 🔥 NEW OPTION
):
    base_dir = settings.PROJECT_ROOT

    if tickets_path is None:
        tickets_path = os.path.join(base_dir, settings.DATA_RAW_PATH, "tickets.csv")

    if tweets_path is None:
        tweets_path = os.path.join(base_dir, settings.DATA_RAW_PATH, "twcs.csv")

    if output_path is None:
        output_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "merged_support_data.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================
    logger.info("Loading datasets...")
    tickets = pd.read_csv(tickets_path, on_bad_lines='skip')
    tweets = pd.read_csv(tweets_path, on_bad_lines='skip')

    # =========================
    # TICKETS
    # =========================
    tickets_std = tickets[['Ticket_ID', 'Ticket_Description', 'Issue_Category']].copy()
    tickets_std.rename(columns={
        'Ticket_ID': 'id',
        'Ticket_Description': 'text',
        'Issue_Category': 'category'
    }, inplace=True)

    tickets_std['source'] = 'ticket'

    # =========================
    # TWEETS
    # =========================
    tweets_std = tweets[['tweet_id', 'text']].copy()
    tweets_std.rename(columns={'tweet_id': 'id'}, inplace=True)

    if 'category' in tweets.columns:
        tweets_std['category'] = tweets['category'].astype(str)

        if categorize_tweets:
            if overwrite_existing_categories:
                tweets_std['category'] = tweets_std['text'].apply(categorize_tweet)
            else:
                mask = tweets_std['category'].apply(should_categorize)
                tweets_std.loc[mask, 'category'] = tweets_std.loc[mask, 'text'].apply(categorize_tweet)
    else:
        tweets_std['category'] = tweets_std['text'].apply(categorize_tweet)

    tweets_std['source'] = 'twitter'

    # =========================
    # MERGE
    # =========================
    merged = pd.concat([tickets_std, tweets_std], ignore_index=True)

    # =========================
    # CLEAN TEXT
    # =========================
    logger.info("Cleaning text...")
    merged['text'] = merged['text'].apply(clean_text)

    # =========================
    # BALANCE (OPTIONAL 🔥)
    # =========================
    if balance_data:
        merged = balance_dataset(merged)

    # =========================
    # LOG STATS
    # =========================
    logger.info(f"Total records: {len(merged)}")
    logger.info(f"Distribution:\n{merged['category'].value_counts(normalize=True)}")

    # =========================
    # SAVE
    # =========================
    merged.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    return merged