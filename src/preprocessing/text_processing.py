import re
from nltk.corpus import stopwords
from src.utils.logger import get_logger

logger = get_logger(__name__)

def clean_text(text):
    """Normalize and remove noise from a single text string."""
    stop = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    cleaned = " ".join([w for w in text.split() if w not in stop])

    logger.debug(f"Original: {text} | Cleaned: {cleaned}")
    return cleaned

def merge_subject_description(data):
    """Combine subject and description fields into `full_text`."""
    logger.info("Merging Ticket_Subject and Ticket_Description into full_text")
    data["full_text"] = (
        data["Ticket_Subject"].astype(str) + " " +
        data["Ticket_Description"].astype(str)
    )
    return data