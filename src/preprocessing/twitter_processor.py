"""
Twitter data processor - Optimized keyword categorization for 8 categories.
Maximum keyword coverage for high accuracy.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Optional, Union
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


class TweetCategorizer:
    """
    Optimized keyword-based tweet categorizer - 8 categories.
    Each category has primary, secondary, and tertiary keywords for maximum coverage.
    """

    # Complete keyword database for 8 categories
    CATEGORY_KEYWORDS = {
        # FRAUD - High priority, strong indicators
        'Fraud': {
            'primary': [
                'fraud', 'scam', 'unauthorized', 'stolen', 'hack', 'compromised',
                'identity theft', 'security breach', 'fake transaction', 'phishing',
                'money stolen', 'card stolen', 'account takeover', 'breach'
            ],
            'secondary': [
                'suspicious', 'fake', 'not me', 'wasn\'t me', 'didn\'t authorize',
                'did not authorize', 'unknown charge', 'unrecognized', 'fraudulent'
            ],
            'tertiary': [
                'stole', 'theft', 'hijack', 'impersonation', 'scammer', 'hacker'
            ],
            'weight': 3.0
        },

        # BILLING - Payment, refund, subscription issues
        'Billing': {
            'primary': [
                'refund', 'charge', 'billing', 'payment', 'invoice', 'subscription',
                'overcharged', 'double charge', 'wrong charge', 'billing issue',
                'payment failed', 'chargeback', 'credit card', 'debit card'
            ],
            'secondary': [
                'money', 'cost', 'price', 'overcharge', 'cancel subscription',
                'renew subscription', 'bill', 'charged', 'refund request',
                'money back', 'return money', 'get refund'
            ],
            'tertiary': [
                'fee', 'tax', 'discount', 'coupon', 'promo', 'voucher', 'credit',
                'statement', 'receipt', 'transaction', 'withdrawal'
            ],
            'weight': 2.5
        },

        # TECHNICAL - Bugs, crashes, errors, performance
        'Technical': {
            'primary': [
                'crash', 'error', 'bug', 'freeze', 'not working', 'failed',
                'broken', 'glitch', 'down', 'offline', 'unresponsive',
                'won\'t load', 'won\'t open', 'won\'t start', 'stuck',
                'loading', 'spinning wheel', 'white screen', 'black screen'
            ],
            'secondary': [
                'slow', 'battery drain', 'battery life', 'update broke',
                'ios update', 'android update', 'connection issue',
                'wifi issue', 'internet issue', 'signal', 'network',
                'lag', 'delay', 'timeout', 'server error', '500 error',
                '404 error', 'login error', 'sync issue'
            ],
            'tertiary': [
                'hang', 'stutter', 'fps', 'performance', 'memory', 'storage',
                'installation', 'setup', 'configure', 'compatibility'
            ],
            'weight': 2.0
        },

        # ACCOUNT - Login, password, profile, access
        'Account': {
            'primary': [
                'login', 'password', 'account', 'signin', 'access', 'locked',
                'reset password', 'forgot password', 'change password',
                'verify account', 'authentication', 'two factor', '2fa',
                'email verification', 'phone verification'
            ],
            'secondary': [
                'reset', 'verification', 'code', 'profile', 'email',
                'username', 'user id', 'credentials', 'log in', 'sign in',
                'can\'t login', 'can\'t access', 'locked out'
            ],
            'tertiary': [
                'register', 'sign up', 'create account', 'delete account',
                'deactivate', 'reactivate', 'recover', 'restore'
            ],
            'weight': 2.0
        },

        # DELIVERY - Shipping, packages, orders, tracking
        'Delivery': {
            'primary': [
                'delivery', 'shipping', 'package', 'order', 'track',
                'arrive', 'delivered', 'shipment', 'courier', 'mail',
                'package not received', 'order not delivered', 'late delivery',
                'tracking number', 'tracking update', 'estimated delivery'
            ],
            'secondary': [
                'delay', 'waiting', 'received', 'ship', 'dispatch',
                'out for delivery', 'in transit', 'shipped', 'fulfillment',
                'lost package', 'missing package', 'damaged package',
                'return', 'exchange', 'refund shipping'
            ],
            'tertiary': [
                'box', 'envelope', 'parcel', 'freight', 'expedite',
                'rush', 'priority', 'standard shipping', 'free shipping'
            ],
            'weight': 1.8
        },

        # FEATURE REQUEST - Suggestions, improvements, wishes
        'Feature Request': {
            'primary': [
                'suggest', 'feature', 'improvement', 'would like', 'wish',
                'add feature', 'new feature', 'enhancement', 'capability',
                'functionality', 'option to', 'ability to'
            ],
            'secondary': [
                'idea', 'recommend', 'propose', 'request', 'want to see',
                'hope to see', 'it would be great', 'it would be nice',
                'missing feature', 'lack of', 'doesn\'t have'
            ],
            'tertiary': [
                'upgrade', 'update suggestion', 'future release', 'roadmap',
                'planned', 'consider adding', 'please add'
            ],
            'weight': 1.5
        },

        # CUSTOMER SUPPORT - Help, assistance, service quality
        'Customer Support': {
            'primary': [
                'help', 'support', 'assist', 'customer service', 'representative',
                'live chat', 'call center', 'helpline', 'tech support',
                'customer care', 'service team', 'support team'
            ],
            'secondary': [
                'respond', 'answer', 'reply', 'contact', 'agent',
                'speak to', 'talk to', 'reach out', 'get through',
                'wait time', 'hold time', 'escalate'
            ],
            'tertiary': [
                'complaint', 'poor service', 'bad support', 'unhelpful',
                'great service', 'excellent support', 'helpful agent'
            ],
            'weight': 1.2
        },

        # GENERAL INQUIRY - Questions, information, how-to
        'General Inquiry': {
            'primary': [
                'how', 'what', 'where', 'when', 'why', 'who', 'which',
                'question', 'information', 'about', 'tell me', 'explain',
                'clarify', 'understand', 'confused', 'unsure'
            ],
            'secondary': [
                'help me understand', 'can you tell', 'do you know',
                'is it possible', 'how does it work', 'what is the',
                'where can I', 'when will', 'why is it'
            ],
            'tertiary': [
                'info', 'details', 'guidance', 'instruction', 'tutorial',
                'walkthrough', 'example', 'documentation'
            ],
            'weight': 1.0
        }
    }

    # Common phrase mapping (higher priority than keyword matching)
    PHRASE_MAPPING = {
        # Technical
        'battery drain': 'Technical',
        'battery life': 'Technical',
        'update ruined': 'Technical',
        'update broke': 'Technical',
        'ios update': 'Technical',
        'won\'t load': 'Technical',
        'keep crashing': 'Technical',
        'keeps freezing': 'Technical',
        'spinning wheel': 'Technical',
        'white screen': 'Technical',
        'black screen': 'Technical',
        'not responding': 'Technical',
        'taking forever': 'Technical',
        
        # Billing
        'double charge': 'Billing',
        'wrong charge': 'Billing',
        'money back': 'Billing',
        'refund please': 'Billing',
        'cancel subscription': 'Billing',
        
        # Account
        'can\'t login': 'Account',
        'can\'t access': 'Account',
        'locked out': 'Account',
        'forgot password': 'Account',
        'reset password': 'Account',
        
        # Fraud
        'identity theft': 'Fraud',
        'security breach': 'Fraud',
        'unauthorized transaction': 'Fraud',
        
        # Delivery
        'never received': 'Delivery',
        'didn\'t arrive': 'Delivery',
        'where is my package': 'Delivery',
        'tracking number': 'Delivery',
        'out for delivery': 'Delivery',
        
        # Feature Request
        'would be nice': 'Feature Request',
        'would be great': 'Feature Request',
        'i wish': 'Feature Request',
        'please add': 'Feature Request',
        
        # Customer Support
        'customer service': 'Customer Support',
        'live chat': 'Customer Support',
        'talk to agent': 'Customer Support',
        'speak to representative': 'Customer Support',
    }

    @classmethod
    def categorize(cls, text: str):
        """
        Categorize tweet with confidence score.
        Returns: (category, confidence)
        """
        if not text or len(text) < 5:
            return 'General Inquiry', 0.5

        text = text.lower()
        scores = {}

        # Step 1: Check phrase mapping (highest priority)
        for phrase, category in cls.PHRASE_MAPPING.items():
            if phrase in text:
                return category, 0.95

        # Step 2: Keyword scoring
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            score = 0.0

            # Primary keywords (highest weight)
            for keyword in keywords['primary']:
                if keyword in text:
                    score += keywords['weight'] * 2.0
                    # Bonus for exact word match
                    if re.search(rf'\b{re.escape(keyword)}\b', text):
                        score += keywords['weight']

            # Secondary keywords (medium weight)
            for keyword in keywords['secondary']:
                if keyword in text:
                    score += keywords['weight']

            # Tertiary keywords (lowest weight)
            for keyword in keywords.get('tertiary', []):
                if keyword in text:
                    score += keywords['weight'] * 0.5

            if score > 0:
                scores[category] = score

        # Step 3: If no matches, return General Inquiry
        if not scores:
            return 'General Inquiry', 0.5

        # Step 4: Get best category and calculate confidence
        best_category = max(scores, key=scores.get)
        max_score = scores[best_category]
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        confidence = min(confidence, 0.95)  # Cap at 95%

        return best_category, confidence


def extract_customer_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only customer tweets (not support responses)."""
    logger.info("Extracting customer tweets from conversation data...")

    original_count = len(df)

    if 'inbound' in df.columns:
        customer_tweets = df[df['inbound'] == True].copy()
        logger.info(f"  Using inbound=True: {len(customer_tweets)} customer tweets")
    else:
        customer_tweets = df.copy()
        logger.warning("  No inbound column - keeping all tweets")

    removed = original_count - len(customer_tweets)
    logger.info(f"  Removed {removed} support response tweets")

    return customer_tweets


def clean_tweet_text(text: str) -> str:
    """Clean individual tweet text."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove emojis and special characters (keep letters, numbers, spaces, punctuation)
    text = re.sub(r'[^\w\s\.\?\!]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Lowercase
    text = text.lower()

    # Keep only first 2 sentences
    sentences = re.split(r'[.!?]', text)
    if len(sentences) > 2:
        text = '.'.join(sentences[:2]) + '.'

    # Remove common Twitter noise
    noise_patterns = [r'rt\s+', r'via\s+', r'https?\S+', r'&amp;', r'&lt;', r'&gt;']
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def process_twitter_data(
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    min_text_length: int = 15,
    sample_size: Optional[int] = None,
    confidence_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Process twcs.csv - cleans tweets and categorizes using optimized keywords.
    Returns 8 categories for tweets.

    Returns:
        DataFrame with columns: clean_text, category, confidence, source
    """
    base_dir = Path(settings.PROJECT_ROOT)

    if input_path is None:
        input_path = base_dir / settings.DATA_RAW_PATH / "twcs.csv"

    if output_path is None:
        output_path = base_dir / settings.DATA_PROCESSED_PATH / "tweets_processed.csv"

    logger.info("=" * 70)
    logger.info("PROCESSING TWITTER DATA (8 CATEGORY KEYWORD CLASSIFICATION)")
    logger.info("=" * 70)
    logger.info("Categories: Fraud, Billing, Technical, Account, Delivery, Feature Request, Customer Support, General Inquiry")

    # Load data
    df = pd.read_csv(input_path, low_memory=False, on_bad_lines='skip')
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Using sample of {len(df):,} rows")

    # Extract customer tweets
    df = extract_customer_tweets(df)

    # Clean text
    logger.info("Cleaning tweet text...")
    df['clean_text'] = df['text'].astype(str).apply(clean_tweet_text)

    # Show examples
    logger.info("Text Cleaning Examples:")
    for i in range(min(3, len(df))):
        original = df.iloc[i]['text'][:60] if pd.notna(df.iloc[i]['text']) else "N/A"
        cleaned = df.iloc[i]['clean_text'][:60]
        logger.info(f"  Original: {original}...")
        logger.info(f"  Cleaned:  {cleaned}...")

    # Remove short texts
    before = len(df)
    df = df[df['clean_text'].str.len() >= min_text_length]
    logger.info(f"Removed {before - len(df):,} tweets with text < {min_text_length} chars")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['clean_text'], keep='first')
    logger.info(f"Removed {before - len(df):,} duplicate tweets")

    # Categorize using optimized keywords
    logger.info("Categorizing tweets with optimized keyword matching...")

    categories = []
    confidences = []

    for text in df['clean_text']:
        cat, conf = TweetCategorizer.categorize(text)
        categories.append(cat)
        confidences.append(conf)

    df['category'] = categories
    df['confidence'] = confidences

    # Filter by confidence
    high_confidence = df[df['confidence'] >= confidence_threshold].copy()
    logger.info(f"High confidence tweets (>= {confidence_threshold}): {len(high_confidence):,}/{len(df):,} ({len(high_confidence)/len(df)*100:.1f}%)")

    # Category distribution
    logger.info("\nCategory Distribution (8 categories):")
    cat_dist = high_confidence['category'].value_counts()
    for cat, count in cat_dist.items():
        logger.info(f"  {cat}: {count:,} ({count/len(high_confidence)*100:.1f}%)")

    # Sample categorized tweets
    logger.info("\nSample Categorized Tweets:")
    for category in high_confidence['category'].unique():
        sample = high_confidence[high_confidence['category'] == category].iloc[0]
        logger.info(f"  [{category}] (conf: {sample['confidence']:.2f})")
        logger.info(f"    {sample['clean_text'][:70]}...")

    # Prepare output
    output_df = high_confidence[['clean_text', 'category', 'confidence']].copy()
    output_df['source'] = 'twitter'

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(f"\nSUCCESS! Saved {len(output_df):,} categorized tweets to {output_path}")
    logger.info(f"  Categories: {sorted(output_df['category'].unique())}")

    return output_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process Twitter data with optimized keyword categorization")
    parser.add_argument("--sample", type=int, help="Sample size for testing")
    parser.add_argument("--min-length", type=int, default=15, help="Minimum text length")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")

    args = parser.parse_args()

    tweets_df = process_twitter_data(
        sample_size=args.sample,
        min_text_length=args.min_length,
        confidence_threshold=args.confidence
    )

    print(f"\nTwitter processing complete!")
    print(f"Output: data/processed/tweets_processed.csv")
    print(f"Total tweets: {len(tweets_df):,}")
    print(f"Categories: {sorted(tweets_df['category'].unique())}")