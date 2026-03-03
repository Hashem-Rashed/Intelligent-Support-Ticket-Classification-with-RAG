import re
from nltk.corpus import stopwords

def clean_text(text):
    stop = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = " ".join([w for w in text.split() if w not in stop])

    return text

def merge_subject_description(data):

    data["full_text"] = (
        data["Ticket_Subject"].astype(str) + " " +
        data["Ticket_Description"].astype(str)
    )

    return data