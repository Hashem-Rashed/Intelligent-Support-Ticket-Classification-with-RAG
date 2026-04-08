import os
import pandas as pd
import re
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def categorize_tweet(text):
    """Categorize a tweet based on its content using keyword matching."""
    if pd.isna(text):
        return "General Inquiry"
    
    text = str(text).lower()
    
    # Category definitions - Consolidated to avoid duplicates
    categories = {
        "Account": {
            "keywords": [
                "account", "login", "password", "sign in", "signin", "log in",
                "username", "credentials", "reset password", "forgot password",
                "2fa", "two factor", "verification", "authenticator", "profile",
                "update email", "change password", "security", "hacked", "compromised"
            ],
            "weight": 1
        },
        "Billing": {
            "keywords": [
                "billing", "invoice", "subscription", "renewal", "cancel subscription",
                "monthly fee", "annual fee", "bill", "payment method", "refund", 
                "charge", "overcharged", "unauthorized charge", "wrong charge", 
                "dispute", "didn't authorize", "unrecognized transaction", 
                "double charged", "incorrect charge", "wrong amount", "disputed charge",
                "payment", "credit card", "debit card", "paypal", "transaction", 
                "money back", "price", "cost", "discount", "promo code", "coupon", 
                "paid", "card declined"
            ],
            "weight": 1
        },
        "Technical": {
            "keywords": [
                "not working", "error", "bug", "crash", "freeze", "glitch",
                "failed", "issue with", "problem with", "broken", "stuck",
                "loading", "technical difficulties", "system error", "server error",
                "slow", "laggy", "performance", "response time", "takes forever",
                "long load", "buffering", "high latency", "delayed", "unresponsive"
            ],
            "weight": 1
        },
        "Delivery": {
            "keywords": [
                "delivery", "shipping", "track order", "package", "arrived",
                "shipment", "courier", "delivered", "shipping address", "dispatch",
                "out for delivery", "late delivery", "missing package", "damaged package"
            ],
            "weight": 1
        },
        "Feature Request": {
            "keywords": [
                "suggest", "feature request", "would like", "wish list", "add feature",
                "improvement", "enhancement", "missing feature", "capability",
                "functionality", "option to", "ability to", "recommend"
            ],
            "weight": 0.8
        },
        "Customer Support": {
            "keywords": [
                "help", "support", "assistance", "customer service", "live chat",
                "contact us", "reach out", "get help", "ticket", "representative",
                "agent", "escalate", "complaint", "dissatisfied", "frustrated"
            ],
            "weight": 0.9
        },
        "Fraud": {
            "keywords": [
                "fraud", "scam", "unauthorized", "stolen", "identity theft",
                "phishing", "fake", "suspicious activity", "hijacked", 
                "compromised account", "fake account", "impersonation"
            ],
            "weight": 1.2
        },
        "Data & Privacy": {
            "keywords": [
                "privacy", "data breach", "personal information", "gdpr", "ccpa",
                "data leak", "private data", "opt out", "delete account", "remove data"
            ],
            "weight": 1
        },
        "Integration & API": {
            "keywords": [
                "api", "integration", "webhook", "connect with", "zapier",
                "third party", "plugin", "extension", "sync with"
            ],
            "weight": 0.7
        },
        "Mobile App": {
            "keywords": [
                "android", "ios", "mobile app", "iphone", "ipad", "play store",
                "app store", "mobile version", "app update", "app crash"
            ],
            "weight": 0.8
        },
        "General Inquiry": {
            "keywords": [
                "question", "how to", "tutorial", "guide", "documentation",
                "info", "information about", "tell me about", "what is"
            ],
            "weight": 0.5
        }
    }
    
    # Score each category based on keyword matches
    scores = {}
    for category, info in categories.items():
        score = 0
        for keyword in info["keywords"]:
            if keyword in text:
                # Weighted scoring - longer keywords might be more specific
                score += info["weight"] * (1 + len(keyword) / 30)
        if score > 0:
            scores[category] = score
    
    # Return the category with highest score, or default
    if scores:
        return max(scores, key=scores.get)
    else:
        return "General Inquiry"

def should_categorize(category_value):
    """Determine if a tweet needs categorization."""
    if pd.isna(category_value):
        return True
    
    category_str = str(category_value).strip().lower()
    
    # List of generic/default categories that should be replaced
    generic_categories = [
        "customer inquiry", 
        "general", 
        "general inquiry", 
        "unknown", 
        "other",
        "nan",
        "none",
        "",
        "inquiry"
    ]
    
    return category_str in generic_categories

def merge_datasets(
    tickets_path: str | None = None,
    tweets_path: str | None = None,
    output_path: str | None = None,
    categorize_tweets: bool = True,
    overwrite_existing_categories: bool = False
):
    """Merge CRM tickets and Twitter datasets into a unified format.
    
    Args:
        tickets_path: Path to CRM tickets CSV
        tweets_path: Path to Twitter dataset CSV
        output_path: Path to save merged dataset
        categorize_tweets: Whether to categorize tweets
        overwrite_existing_categories: If True, overwrite all tweet categories.
                                       If False, only categorize tweets with generic/empty categories.
    
    Returns:
        Merged DataFrame
    """
    base_dir = settings.PROJECT_ROOT
    
    if tickets_path is None:
        tickets_path = os.path.join(base_dir, settings.DATA_RAW_PATH, "tickets.csv")
    
    if tweets_path is None:
        tweets_path = os.path.join(base_dir, settings.DATA_RAW_PATH, "twcs.csv")
    
    if output_path is None:
        output_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "merged_support_data.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Loading CRM tickets from {tickets_path}")
    tickets = pd.read_csv(tickets_path, on_bad_lines='skip')
    
    logger.info(f"Loading Twitter data from {tweets_path}")
    tweets = pd.read_csv(tweets_path, on_bad_lines='skip')
    
    # Standardize CRM tickets - PRESERVE original category names exactly as they are
    logger.info("Standardizing CRM tickets")
    tickets_standardized = tickets[['Ticket_ID', 'Ticket_Description', 'Issue_Category']].copy()
    tickets_standardized['Ticket_ID'] = tickets_standardized['Ticket_ID'].astype(str)
    
    # Keep EXACT category names from tickets - NO CHANGES
    tickets_standardized.rename(columns={
        'Ticket_ID': 'id',
        'Ticket_Description': 'text',
        'Issue_Category': 'category'
    }, inplace=True)
    tickets_standardized['source'] = 'ticket'
    
    # Keep original fields for backward compatibility
    if 'Ticket_Subject' in tickets.columns:
        tickets_standardized['Ticket_Subject'] = tickets['Ticket_Subject'].astype(str)
    if 'Ticket_Description' in tickets.columns:
        tickets_standardized['Ticket_Description'] = tickets['Ticket_Description'].astype(str)
    
    # Log ticket categories to see what we have (for debugging)
    unique_ticket_categories = tickets_standardized['category'].unique()
    logger.info(f"Original ticket categories (preserved as-is): {unique_ticket_categories}")
    
    # Standardize Twitter dataset
    logger.info("Standardizing Twitter dataset")
    tweets_standardized = tweets[['tweet_id', 'text']].copy()
    tweets_standardized['tweet_id'] = tweets_standardized['tweet_id'].astype(str)
    
    tweets_standardized.rename(columns={
        'tweet_id': 'id'
    }, inplace=True)
    
    # Check if category column exists in tweets data
    if 'category' in tweets.columns:
        logger.info("Twitter data already has categories")
        tweets_standardized['category'] = tweets['category'].astype(str)
        
        # Apply intelligent categorization based on parameters
        if categorize_tweets:
            if overwrite_existing_categories:
                logger.info("Overwriting ALL tweet categories with intelligent categorization")
                tweets_standardized['category'] = tweets_standardized['text'].apply(categorize_tweet)
            else:
                logger.info("Only categorizing tweets with generic/empty categories")
                # Create a mask for tweets that need categorization
                needs_categorization = tweets_standardized['category'].apply(should_categorize)
                logger.info(f"Found {needs_categorization.sum()} tweets that need categorization out of {len(tweets_standardized)}")
                
                # Only categorize those that need it
                tweets_to_categorize = tweets_standardized[needs_categorization]
                categorized_texts = tweets_to_categorize['text'].apply(categorize_tweet)
                
                # Update only the rows that need categorization
                tweets_standardized.loc[needs_categorization, 'category'] = categorized_texts
    else:
        logger.info("Twitter data has no categories, applying intelligent categorization to all tweets")
        if categorize_tweets:
            tweets_standardized['category'] = tweets_standardized['text'].apply(categorize_tweet)
        else:
            tweets_standardized['category'] = "General Inquiry"
    
    # Log category distribution for tweets
    category_dist = tweets_standardized['category'].value_counts()
    logger.info(f"Tweet category distribution:\n{category_dist}")
    
    tweets_standardized['source'] = 'twitter'
    
    # Add placeholder columns for backward compatibility
    tweets_standardized['Ticket_Subject'] = ''
    tweets_standardized['Ticket_Description'] = tweets_standardized['text'].astype(str)
    
    # Ensure all columns have consistent dtypes before saving
    for col in ['id', 'text', 'category', 'source', 'Ticket_Subject', 'Ticket_Description']:
        if col in tickets_standardized.columns:
            tickets_standardized[col] = tickets_standardized[col].astype(str)
        if col in tweets_standardized.columns:
            tweets_standardized[col] = tweets_standardized[col].astype(str)
    
    # Merge datasets
    logger.info("Merging datasets")
    merged_data = pd.concat([tickets_standardized, tweets_standardized], ignore_index=True)
    
    # Log statistics - showing EXACT categories from both sources
    logger.info(f"Total records: {len(merged_data)}")
    logger.info(f"Records per source:\n{merged_data['source'].value_counts()}")
    logger.info(f"All unique categories in merged data (preserved from tickets + new from tweets):")
    logger.info(f"{merged_data['category'].value_counts()}")
    
    # Save merged dataset
    merged_data.to_csv(output_path, index=False)
    logger.info(f"Merged dataset saved to {output_path}")
    
    return merged_data