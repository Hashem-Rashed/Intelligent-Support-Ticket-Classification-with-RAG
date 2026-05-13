"""
Twitter data processor - Advanced categorization with NLP techniques.
Features: lemmatization, negation detection, context scoring, conflict resolution.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Set
from collections import defaultdict
from dataclasses import dataclass, field
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


@dataclass
class MatchResult:
    """Store matching results for analysis."""
    category: str
    score: float
    matched_terms: List[str] = field(default_factory=list)
    context: str = ""


class AdvancedTweetCategorizer:
    """
    Production-grade tweet categorizer with advanced NLP capabilities.
    Features:
    - Lemmatization for word variations
    - Negation detection (prevents false positives)
    - Context window scoring
    - Inter-category conflict resolution
    - Confidence recalibration
    - N-gram matching for phrases
    """
    
    # Stop words to ignore in scoring
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
        'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was',
        'were', 'will', 'with', 'i', 'you', 'we', 'they', 'my', 'your', 'our'
    }
    
    # Negation words that flip meaning
    NEGATION_WORDS = {
        'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor',
        "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't",
        "cant", "cannot", "isnt", "arent", "wasnt", "werent", "havent", "hasnt",
        "hadnt", "dont", "doesnt", "didnt"
    }
    
    # Intensifiers that increase score
    INTENSIFIERS = {
        'very', 'extremely', 'highly', 'really', 'so', 'too', 'absolutely',
        'completely', 'totally', 'utterly', 'horribly', 'terribly', 'awfully'
    }
    
    # Category hierarchy for conflict resolution
    CATEGORY_PRIORITY = {
        'Fraud': 10,
        'Security': 9,
        'Technical': 8,
        'Account': 7,
        'Billing': 6,
        'Delivery': 5,
        'Customer Support': 4,
        'Feature Request': 3,
        'General Inquiry': 2
    }
    
    # Category synonyms (same meaning, different names)
    CATEGORY_SYNONYMS = {
        'Security': ['Fraud', 'Privacy', 'Data Breach'],
        'Billing': ['Payment', 'Invoice', 'Subscription'],
        'Technical': ['Bug', 'Crash', 'Error', 'Performance']
    }
    
    # Context windows for phrase detection
    CONTEXT_WINDOWS = {
        'Fraud': {
            'triggers': ['scam', 'fraud', 'stolen', 'hack', 'breach'],
            'context_boost': 1.5,
            'min_context_matches': 2
        },
        'Billing': {
            'triggers': ['charge', 'payment', 'refund', 'bill', 'subscription'],
            'context_boost': 1.3,
            'min_context_matches': 1
        }
    }
    
    # Complete keyword database with lemmatized forms
    # CHANGED: Reduced weight for ambiguous terms (suspicious, trust, safe)
    CATEGORY_KEYWORDS = {
        'Fraud': {
            'primary': {
                'keywords': ['fraud', 'scam', 'unauthorized', 'stolen', 'hack', 'compromised'],
                'weight': 4.0,
                'lemmas': ['fraud', 'scam', 'steal', 'hack', 'compromise', 'breach']
            },
            'secondary': {
                'keywords': ['unknown transaction', 'wasn\'t me', 'didn\'t authorize'],  # removed 'suspicious'
                'weight': 2.5,
                'lemmas': ['unknown', 'authorize', 'recognize']
            },
            'tertiary': {
                'keywords': ['theft', 'impersonation', 'scammer', 'hacker', 'phish'],
                'weight': 1.5,
                'lemmas': ['theft', 'impersonate', 'scam', 'hack', 'phish']
            }
        },
        
        'Security': {
            'primary': {
                'keywords': ['security', 'privacy', 'data breach', 'exposed', 'leak', 'vulnerability'],
                'weight': 4.0,
                'lemmas': ['secure', 'privacy', 'breach', 'expose', 'leak', 'vulnerable']
            },
            'secondary': {
                'keywords': ['2fa', 'two factor', 'authentication', 'verify', 'verification'],
                'weight': 2.5,
                'lemmas': ['authenticate', 'verify', 'validate']
            },
            'tertiary': {
                'keywords': ['encryption', 'private'],  # removed 'trust' and 'safe'
                'weight': 1.5,
                'lemmas': ['encrypt', 'private']
            }
        },
        
        'Billing': {
            'primary': {
                'keywords': ['refund', 'billing', 'payment', 'invoice', 'subscription', 'chargeback'],
                'weight': 3.5,
                'lemmas': ['refund', 'bill', 'pay', 'invoice', 'subscribe', 'chargeback']
            },
            'secondary': {
                'keywords': ['overcharged', 'double charge', 'wrong amount', 'credit card charge'],
                'weight': 2.5,
                'lemmas': ['overcharge', 'double', 'wrong', 'credit']
            },
            'tertiary': {
                'keywords': ['receipt', 'transaction', 'withdrawal', 'statement', 'balance'],
                'weight': 1.5,
                'lemmas': ['receipt', 'transaction', 'withdraw', 'statement', 'balance']
            }
        },
        
        'Technical': {
            'primary': {
                'keywords': ['crash', 'bug', 'freeze', 'not working', 'broken', 'error'],
                'weight': 3.5,
                'lemmas': ['crash', 'bug', 'freeze', 'work', 'break', 'error']
            },
            'secondary': {
                'keywords': ['slow', 'lag', 'delay', 'timeout', 'connection', 'network'],
                'weight': 2.5,
                'lemmas': ['slow', 'lag', 'delay', 'timeout', 'connect', 'network']
            },
            'tertiary': {
                'keywords': ['battery drain', 'memory', 'storage', 'performance', 'compatibility'],
                'weight': 1.5,
                'lemmas': ['drain', 'memory', 'storage', 'perform', 'compatible']
            }
        },
        
        'Account': {
            'primary': {
                'keywords': ['login', 'password', 'access', 'locked', 'reset password'],
                'weight': 3.5,
                'lemmas': ['login', 'password', 'access', 'lock', 'reset']
            },
            'secondary': {
                'keywords': ['verification', '2fa', 'two factor', 'authenticator', 'code'],
                'weight': 2.5,
                'lemmas': ['verify', 'authenticate', 'code']
            },
            'tertiary': {
                'keywords': ['profile', 'username', 'email', 'sign up', 'register'],
                'weight': 1.5,
                'lemmas': ['profile', 'username', 'email', 'register', 'signup']
            }
        },
        
        'Delivery': {
            'primary': {
                'keywords': ['shipping', 'delivery', 'package', 'track', 'courier'],
                'weight': 3.0,
                'lemmas': ['ship', 'deliver', 'package', 'track', 'courier']
            },
            'secondary': {
                'keywords': ['lost package', 'damaged', 'not received', 'late arrival'],
                'weight': 2.5,
                'lemmas': ['lost', 'damage', 'receive', 'late', 'arrive']
            },
            'tertiary': {
                'keywords': ['return', 'exchange', 'fulfillment', 'dispatch', 'transit'],
                'weight': 1.5,
                'lemmas': ['return', 'exchange', 'fulfill', 'dispatch', 'transit']
            }
        },
        
        'Customer Support': {
            'primary': {
                'keywords': ['customer service', 'support', 'agent', 'representative', 'help'],
                'weight': 3.0,
                'lemmas': ['customer', 'support', 'agent', 'represent', 'help']
            },
            'secondary': {
                'keywords': ['wait time', 'hold', 'escalate', 'response time', 'live chat'],
                'weight': 2.0,
                'lemmas': ['wait', 'hold', 'escalate', 'response', 'chat']
            },
            'tertiary': {
                'keywords': ['unhelpful', 'rude', 'great service', 'excellent support'],
                'weight': 1.5,
                'lemmas': ['helpful', 'rude', 'service', 'excellent']
            }
        },
        
        'Feature Request': {
            'primary': {
                'keywords': ['suggest', 'feature', 'improvement', 'enhancement', 'would like'],
                'weight': 3.0,
                'lemmas': ['suggest', 'feature', 'improve', 'enhance', 'like']
            },
            'secondary': {
                'keywords': ['missing', 'lack of', 'wish had', 'please add', 'it would be great'],
                'weight': 2.5,
                'lemmas': ['miss', 'lack', 'wish', 'add', 'great']
            },
            'tertiary': {
                'keywords': ['roadmap', 'future release', 'planned', 'consider adding'],
                'weight': 1.5,
                'lemmas': ['roadmap', 'future', 'plan', 'consider']
            }
        },
        
        'General Inquiry': {
            'primary': {
                'keywords': ['how to', 'what is', 'when will', 'where can', 'question about'],
                'weight': 2.0,
                'lemmas': ['how', 'what', 'when', 'where', 'question']
            },
            'secondary': {
                'keywords': ['explain', 'clarify', 'understand', 'tell me', 'info on'],
                'weight': 1.5,
                'lemmas': ['explain', 'clarify', 'understand', 'tell', 'info']
            },
            'tertiary': {
                'keywords': ['details', 'guidance', 'tutorial', 'documentation', 'example'],
                'weight': 1.0,
                'lemmas': ['detail', 'guide', 'tutorial', 'document', 'example']
            }
        }
    }
    
    # Multi-word phrases with exact matching (higher priority)
    EXACT_PHRASES = {
        'identity theft': 'Fraud',
        'security breach': 'Security',
        'data leak': 'Security',
        'privacy violation': 'Security',
        'credit card fraud': 'Fraud',
        'unauthorized transaction': 'Fraud',
        'two factor authentication': 'Security',
        'reset password': 'Account',
        'forgot password': 'Account',
        'locked out': 'Account',
        'battery drain': 'Technical',
        'white screen of death': 'Technical',
        'won\'t load': 'Technical',
        'keep crashing': 'Technical',
        'money back': 'Billing',
        'double charge': 'Billing',
        'cancel subscription': 'Billing',
        'tracking number': 'Delivery',
        'out for delivery': 'Delivery',
        'where is my order': 'Delivery',
        'customer service': 'Customer Support',
        'live chat': 'Customer Support',
        'speak to agent': 'Customer Support',
        'would be nice': 'Feature Request',
        'please add': 'Feature Request',
        'how do i': 'General Inquiry',
        'what is the': 'General Inquiry'
    }
    
    # Mutual exclusion rules (if A appears with B, it's actually C)
    MUTUAL_EXCLUSION = {
        ('Fraud', 'Billing'): 'Fraud',
        ('Security', 'Fraud'): 'Security',
        ('Technical', 'Account'): 'Technical',
        ('Feature Request', 'General Inquiry'): 'Feature Request',
        ('Delivery', 'Billing'): 'Delivery',
    }

    # NEW: Override for phrases that clearly indicate not Fraud/Security
    NEGATION_OVERRIDE = {
        'this is not a scam': 'General Inquiry',
        "it's not fraud": 'General Inquiry',
        'not a security issue': 'General Inquiry',
        'no fraud': 'General Inquiry',
        'not hacked': 'Technical',
    }

    @classmethod
    def _lemmatize_word(cls, word: str) -> str:
        """Simple rule-based lemmatization for English."""
        word = word.lower()
        
        # Common irregular forms
        lemmas = {
            'ran': 'run', 'went': 'go', 'saw': 'see', 'ate': 'eat',
            'bought': 'buy', 'thought': 'think', 'caught': 'catch',
            'stole': 'steal', 'stolen': 'steal', 'broke': 'break',
            'brought': 'bring', 'fought': 'fight', 'found': 'find',
            'gave': 'give', 'got': 'get', 'heard': 'hear', 'knew': 'know',
            'left': 'leave', 'lost': 'lose', 'made': 'make', 'paid': 'pay',
            'said': 'say', 'sold': 'sell', 'sent': 'send', 'spoke': 'speak',
            'stood': 'stand', 'took': 'take', 'taught': 'teach', 'told': 'tell',
            'woke': 'wake', 'wore': 'wear', 'won': 'win', 'wrote': 'write'
        }
        
        if word in lemmas:
            return lemmas[word]
        
        # Regular rules
        if word.endswith('ing'):
            if len(word) > 4:
                return word[:-3]
        elif word.endswith('ed'):
            if len(word) > 3:
                return word[:-2]
        elif word.endswith('s') and not word.endswith(('ss', 'sh', 'ch', 'x', 'z')):
            return word[:-1]
        elif word.endswith('ies'):
            return word[:-3] + 'y'
        elif word.endswith('es') and word[-3] in 'sxz':
            return word[:-2]
        
        return word
    
    @classmethod
    def _detect_negation(cls, text: str, keyword_pos: int, window: int = 5) -> bool:
        """Detect if keyword is negated within context window."""
        words = text.split()
        start = max(0, keyword_pos - window)
        context_words = words[start:keyword_pos]
        
        for neg_word in cls.NEGATION_WORDS:
            if neg_word in ' '.join(context_words):
                return True
        return False
    
    @classmethod
    def _get_intensifier_factor(cls, text: str, keyword_pos: int, window: int = 3) -> float:
        """Get intensifier multiplier if keyword has intensifier nearby."""
        words = text.split()
        start = max(0, keyword_pos - window)
        end = min(len(words), keyword_pos + window + 1)
        context = ' '.join(words[start:end])
        
        for intensifier in cls.INTENSIFIERS:
            if intensifier in context:
                return 1.5
        return 1.0
    
    @classmethod
    def _extract_ngrams(cls, text: str, n: int = 3) -> List[str]:
        """Extract n-grams from text."""
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    @classmethod
    def _apply_context_scoring(cls, text: str, matches: List[MatchResult]) -> List[MatchResult]:
        """Apply context-based scoring adjustments."""
        for match in matches:
            # Check for context windows
            if match.category in cls.CONTEXT_WINDOWS:
                context_config = cls.CONTEXT_WINDOWS[match.category]
                trigger_count = sum(1 for trigger in context_config['triggers'] if trigger in text.lower())
                
                if trigger_count >= context_config['min_context_matches']:
                    match.score *= context_config['context_boost']
                    match.context = f"context_boost_{trigger_count}"
        
        return matches
    
    @classmethod
    def _resolve_conflicts(cls, matches: List[MatchResult]) -> MatchResult:
        """Resolve conflicts between categories using rules."""
        if len(matches) <= 1:
            return matches[0] if matches else None
        
        # Sort by score
        matches.sort(key=lambda x: (-x.score, -cls.CATEGORY_PRIORITY.get(x.category, 0)))
        
        # Check mutual exclusion rules
        for i, match1 in enumerate(matches):
            for match2 in matches[i+1:]:
                key = (match1.category, match2.category)
                if key in cls.MUTUAL_EXCLUSION:
                    winner = cls.MUTUAL_EXCLUSION[key]
                    return next(m for m in matches if m.category == winner)
                
                key_rev = (match2.category, match1.category)
                if key_rev in cls.MUTUAL_EXCLUSION:
                    winner = cls.MUTUAL_EXCLUSION[key_rev]
                    return next(m for m in matches if m.category == winner)
        
        # If scores are very close, use priority
        if matches[0].score - matches[1].score < 0.3:
            # Check if higher priority category exists
            for match in matches:
                if cls.CATEGORY_PRIORITY.get(match.category, 0) > cls.CATEGORY_PRIORITY.get(matches[0].category, 0):
                    return match
        
        return matches[0]
    
    @classmethod
    def categorize(cls, text: str) -> Tuple[str, float, Dict]:
        """
        Advanced categorization with full NLP pipeline.
        Returns: (category, confidence, metadata)
        """
        if not text or len(text) < 5:
            return 'General Inquiry', 0.3, {'reason': 'text_too_short'}
        
        original_text = text
        text_lower = text.lower().strip()
        
        # NEW: Check negation overrides first
        for phrase, override_cat in cls.NEGATION_OVERRIDE.items():
            if phrase in text_lower:
                return override_cat, 0.75, {'reason': f'negation_override_{phrase}'}
        
        # Step 1: Check exact phrases (highest confidence)
        for phrase, category in cls.EXACT_PHRASES.items():
            if phrase in text_lower:
                return category, 0.98, {'matched_phrase': phrase}
        
        # Step 2: Extract n-grams for better matching
        ngrams = cls._extract_ngrams(text_lower, 2) + cls._extract_ngrams(text_lower, 3)
        
        # Step 3: Tokenize and lemmatize
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        lemmatized_words = [cls._lemmatize_word(w) for w in words]
        
        # Step 4: Score each category
        matches = []
        
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            score = 0.0
            matched_terms = []
            
            # Process each keyword level
            for level in ['primary', 'secondary', 'tertiary']:
                level_data = keywords[level]
                level_weight = level_data['weight']
                
                # Check exact keywords
                for keyword in level_data['keywords']:
                    if keyword in text_lower or keyword in ngrams:
                        # Check for negation
                        keyword_pos = text_lower.find(keyword)
                        if keyword_pos >= 0 and cls._detect_negation(text_lower, keyword_pos):
                            score -= level_weight * 0.5  # Penalize negation
                        else:
                            score += level_weight
                            matched_terms.append(keyword)
                
                # Check lemmatized forms
                for lemma in level_data.get('lemmas', []):
                    if lemma in lemmatized_words:
                        idx = lemmatized_words.index(lemma)
                        if not cls._detect_negation(text_lower, idx):
                            intensifier = cls._get_intensifier_factor(text_lower, idx)
                            score += level_weight * intensifier
                            matched_terms.append(f"{lemma} (lemma)")
            
            if score > 0:
                matches.append(MatchResult(
                    category=category,
                    score=score,
                    matched_terms=matched_terms,
                    context=""
                ))
        
        # Step 5: Apply context scoring
        matches = cls._apply_context_scoring(text_lower, matches)
        
        # Step 6: Handle no matches
        if not matches:
            # Check if it's clearly a question
            if any(q in text_lower for q in ['?', 'how', 'what', 'when', 'where', 'why', 'can you', 'could you']):
                return 'General Inquiry', 0.6, {'reason': 'question_detected'}
            return 'General Inquiry', 0.4, {'reason': 'no_matches'}
        
        # Step 7: Resolve conflicts
        best_match = cls._resolve_conflicts(matches)
        
        # Step 8: Calculate confidence score
        total_score = sum(m.score for m in matches)
        confidence = best_match.score / total_score if total_score > 0 else 0.5
        confidence = min(confidence, 0.98)  # Cap at 98%
        
        # Adjust confidence based on match quality
        if len(best_match.matched_terms) >= 3:
            confidence = min(confidence + 0.1, 0.98)
        elif len(best_match.matched_terms) == 1:
            confidence = max(confidence - 0.1, 0.3)
        
        metadata = {
            'matched_terms': best_match.matched_terms[:5],
            'total_matches': len(matches),
            'score': best_match.score,
            'context': best_match.context
        }
        
        return best_match.category, confidence, metadata
    
    @classmethod
    def batch_categorize(cls, texts: List[str]) -> List[Tuple[str, float]]:
        """Categorize multiple texts efficiently."""
        results = []
        for text in texts:
            category, confidence, _ = cls.categorize(text)
            results.append((category, confidence))
        return results


def clean_tweet_text(text: str, advanced: bool = True) -> str:
    """
    Advanced text cleaning with preservation of intent.
    
    Args:
        text: Raw tweet text
        advanced: If True, preserves negation and intensifiers
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove @mentions but keep their context if they're part of a complaint
    text = re.sub(r'@\w+\s+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Replace emojis with text representations
    emoji_map = {
        '😡': 'angry', '😠': 'angry', '💢': 'angry',
        '😢': 'sad', '😭': 'crying', '😤': 'frustrated',
        '💰': 'money', '💳': 'credit card', '📦': 'package',
        '🚚': 'delivery', '🔒': 'secure', '🔓': 'unlocked',
        '⚠️': 'warning', '❌': 'wrong', '✅': 'correct'
    }
    
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, f' {replacement} ')
    
    # Remove remaining emojis and special characters
    text = re.sub(r'[^\w\s\.\?\!]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Keep case for negation detection (don't lowercast entirely)
    # But lower first letter for consistency
    if text:
        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
    
    # Preserve important punctuation
    text = re.sub(r'([!?])+', r'\1', text)  # Normalize multiple punctuation
    
    # Remove common Twitter noise but preserve meaning
    noise_patterns = [
        (r'\brt\s+', ''),  # Retweet
        (r'\bvia\s+', ''),  # Via
        (r'&amp;', 'and'),  # HTML entities
        (r'&lt;', '<'),
        (r'&gt;', '>'),
        (r'#\w+', lambda m: m.group(0)[1:])  # Remove hash symbol but keep word
    ]
    
    for pattern, replacement in noise_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Fix common contractions
    contraction_map = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
        "shouldn't": "should not", "isn't": "is not", "aren't": "are not",
        "wasn't": "was not", "weren't": "were not", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not", "can't": "cannot"
    }
    
    for contraction, expanded in contraction_map.items():
        text = re.sub(rf'\b{contraction}\b', expanded, text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Keep meaningful length (don't truncate for better context)
    if len(text) > 500:
        # Try to cut at sentence boundary
        sentences = re.split(r'[.!?]', text)
        truncated = []
        current_len = 0
        
        for sent in sentences:
            if current_len + len(sent) < 500:
                truncated.append(sent)
                current_len += len(sent)
            else:
                break
        
        text = '.'.join(truncated) + '.' if truncated else text[:500]
    
    return text


def extract_customer_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only customer tweets with advanced filtering."""
    logger.info("Extracting customer tweets from conversation data...")
    
    original_count = len(df)
    
    # Multiple methods to identify customer tweets
    if 'inbound' in df.columns:
        customer_mask = df['inbound'] == True
    else:
        # Heuristic: customer tweets often have less than 280 chars and contain questions/complaints
        text_col = df.get('text', df.iloc[:, 0])  # Assume first col is text if no 'text' col
        text_lengths = text_col.astype(str).str.len()
        
        # Customers typically write shorter, more emotional tweets
        customer_mask = (
            (text_lengths < 280) &
            (text_lengths > 15) &
            (~text_col.astype(str).str.contains(r'^@\w+\s+(help|support|thanks)', case=False, na=False))
        )
    
    customer_tweets = df[customer_mask].copy()
    logger.info(f"  Identified {len(customer_tweets):,} customer tweets ({len(customer_tweets)/original_count*100:.1f}%)")
    
    removed = original_count - len(customer_tweets)
    logger.info(f"  Removed {removed} support response/non-customer tweets")
    
    return customer_tweets


def process_twitter_data(
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    min_text_length: int = 15,
    sample_size: Optional[int] = None,
    confidence_threshold: float = 0.35,
    advanced_cleaning: bool = True
) -> pd.DataFrame:
    """
    Process twitter data with advanced NLP categorization.
    Returns enhanced DataFrame with categories, confidence, and metadata.
    """
    base_dir = Path(settings.PROJECT_ROOT)
    
    if input_path is None:
        input_path = base_dir / settings.DATA_RAW_PATH / "twcs.csv"
    
    if output_path is None:
        output_path = base_dir / settings.DATA_PROCESSED_PATH / "tweets_processed.csv"
    
    logger.info("=" * 80)
    logger.info("ADVANCED TWITTER DATA PROCESSING")
    logger.info("Features: Lemma matching | Negation detection | Context scoring | Conflict resolution")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False, on_bad_lines='skip')
    logger.info(f"Loaded {len(df):,} raw rows")
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Using sample: {len(df):,} rows")
    
    # Extract customer tweets
    df = extract_customer_tweets(df)
    
    # Clean text
    logger.info("Cleaning tweet text (advanced mode)...")
    df['clean_text'] = df['text'].astype(str).apply(lambda x: clean_tweet_text(x, advanced=advanced_cleaning))
    
    # Show cleaning examples
    logger.info("\nText Cleaning Examples:")
    for i in range(min(3, len(df))):
        original = df.iloc[i]['text'][:80] if pd.notna(df.iloc[i]['text']) else "N/A"
        cleaned = df.iloc[i]['clean_text'][:80]
        logger.info(f"  Original: {original}...")
        logger.info(f"  Cleaned:  {cleaned}...")
    
    # Remove short texts
    before = len(df)
    df = df[df['clean_text'].str.len() >= min_text_length]
    logger.info(f"\nRemoved {before - len(df):,} tweets with text < {min_text_length} chars")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['clean_text'], keep='first')
    logger.info(f"Removed {before - len(df):,} duplicate tweets")
    
    # Reset index and capture the exact number of rows to process
    df = df.reset_index(drop=True)
    n_rows = len(df)
    logger.info(f"Processing {n_rows:,} tweets after cleaning and dedup")
    
    # ========== CRITICAL: Extract clean text as a plain Python list ==========
    clean_texts = df['clean_text'].tolist()   # length = n_rows
    
    # Pre-allocate result lists
    categories = [''] * n_rows
    confidences = [0.0] * n_rows
    metadatas = [{}] * n_rows
    
    logger.info("\nRunning advanced categorization...")
    
    for idx, text in enumerate(clean_texts):
        cat, conf, meta = AdvancedTweetCategorizer.categorize(text)
        categories[idx] = cat
        confidences[idx] = conf
        metadatas[idx] = meta
        
        if (idx + 1) % 5000 == 0:
            logger.info(f"  Processed {idx + 1:,}/{n_rows:,} tweets...")
    
    # ========== Assign results back to a fresh DataFrame copy ==========
    # Create a clean output DataFrame with only the columns we need
    output_df = pd.DataFrame({
        'clean_text': clean_texts,
        'category': categories,
        'confidence': confidences,
        'metadata': metadatas
    })
    
    # Also keep original index if needed? Not necessary.
    
    # Filter by confidence
    high_confidence = output_df[output_df['confidence'] >= confidence_threshold].copy()
    medium_confidence = output_df[(output_df['confidence'] >= 0.2) & (output_df['confidence'] < confidence_threshold)].copy()
    
    # Replace Unicode symbol to avoid console error
    logger.info(f"\nClassification Results:")
    logger.info(f"  High confidence (>= {confidence_threshold}): {len(high_confidence):,} ({len(high_confidence)/len(output_df)*100:.1f}%)")
    logger.info(f"  Medium confidence (0.2-{confidence_threshold}): {len(medium_confidence):,} ({len(medium_confidence)/len(output_df)*100:.1f}%)")
    logger.info(f"  Low confidence (<0.2): {len(output_df) - len(high_confidence) - len(medium_confidence):,}")
    
    # Category distribution
    logger.info("\n" + "=" * 60)
    logger.info("CATEGORY DISTRIBUTION (High Confidence Tweets)")
    logger.info("=" * 60)
    
    cat_dist = high_confidence['category'].value_counts()
    for cat, count in cat_dist.items():
        avg_conf = high_confidence[high_confidence['category'] == cat]['confidence'].mean()
        logger.info(f"  {cat:20s}: {count:6,} ({count/len(high_confidence)*100:5.1f}%) - avg conf: {avg_conf:.2f}")
    
    # Sample tweets
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE CLASSIFICATIONS WITH METADATA")
    logger.info("=" * 60)
    
    for category in high_confidence['category'].unique():
        sample = high_confidence[high_confidence['category'] == category].iloc[0]
        meta = sample['metadata']
        logger.info(f"\n[{category}] (conf: {sample['confidence']:.2f})")
        logger.info(f"  Text: {sample['clean_text'][:100]}...")
        if 'matched_terms' in meta:
            logger.info(f"  Matched: {meta['matched_terms'][:3]}")
        if 'matched_phrase' in meta:
            logger.info(f"  Phrase: {meta['matched_phrase']}")
    
    # Prepare final output
    final_df = high_confidence[['clean_text', 'category', 'confidence']].copy()
    final_df['source'] = 'twitter'
    final_df['text_length'] = final_df['clean_text'].str.len()
    
    # Add matched terms as string (optional)
    final_df['matched_terms'] = high_confidence['metadata'].apply(lambda m: str(m.get('matched_terms', [])))
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    if len(medium_confidence) > 0:
        medium_path = output_path.parent / "tweets_medium_confidence_review.csv"
        medium_confidence[['clean_text', 'category', 'confidence']].to_csv(medium_path, index=False)
        logger.info(f"\nSaved {len(medium_confidence)} medium-confidence tweets to {medium_path} for review")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SUCCESS! Saved {len(final_df):,} categorized tweets to {output_path}")
    logger.info(f"Categories: {sorted(final_df['category'].unique())}")
    logger.info(f"{'='*60}")
    
    return final_df


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Process Twitter data with advanced categorization")
    parser.add_argument("--sample", type=int, help="Sample size for testing")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum text length")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--test", action="store_true", help="Run test on sample tweets")
    
    args = parser.parse_args()
    
    if args.test:
        # Test with sample tweets
        test_tweets = [
            "Someone stole my credit card and made unauthorized purchases!",
            "The app keeps crashing every time I try to open it",
            "How do I reset my password? I'm locked out",
            "Where's my package? It was supposed to arrive yesterday",
            "I would love it if you added dark mode",
            "Your customer service is amazing, thank you!",
            "Why was I double charged for my subscription?",
            "This is not a scam, just a question about my bill"
        ]
        
        print("\n" + "="*60)
        print("TESTING ADVANCED CATEGORIZER")
        print("="*60)
        
        categorizer = AdvancedTweetCategorizer()
        for tweet in test_tweets:
            category, confidence, metadata = categorizer.categorize(tweet)
            print(f"\nTweet: {tweet}")
            print(f"Category: {category} (conf: {confidence:.2f})")
            print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        print("\n" + "="*60)
    else:
        tweets_df = process_twitter_data(
            sample_size=args.sample,
            min_text_length=args.min_length,
            confidence_threshold=args.confidence,
            advanced_cleaning=True
        )
        
        print(f"\n✓ Twitter processing complete!")
        print(f"✓ Output: data/processed/tweets_processed_advanced.csv")
        print(f"✓ Total tweets: {len(tweets_df):,}")
        print(f"✓ Categories: {sorted(tweets_df['category'].unique())}")