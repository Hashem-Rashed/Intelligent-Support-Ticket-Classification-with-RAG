"""
Simple Streamlit UI for ticket classification.
Allows user to choose between Baseline (fast) and Transformer (more accurate).
Now applies the same text cleaning as training.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import torch
from src.api.classifier import ProductionTicketClassifier
from src.preprocessing.text_processing import clean_text

st.set_page_config(page_title="Ticket Classifier", page_icon="🎫", layout="centered")

st.title("🎫 Support Ticket Classification")
st.markdown("Enter a customer support message to classify its category and detect fraud.")

# Model selection
model_option = st.radio(
    "Select Model:",
    ["Baseline (Fast, 91.4% accuracy, runs on CPU)", "Transformer (Accurate, ~97.8%, requires GPU)"],
    help="Baseline is fast and runs on any CPU. Transformer is more accurate but needs a GPU (may be slower)."
)

use_transformer = "Transformer" in model_option

# Load selected model (cached per model type)
@st.cache_resource
def load_baseline():
    model_path = project_root / "models" / "saved" / "baseline"
    classifier = ProductionTicketClassifier(model_path, model_type="baseline")
    return classifier

@st.cache_resource
def load_transformer():
    model_path = project_root / "models" / "saved" / "transformer"
    classifier = ProductionTicketClassifier(model_path, model_type="transformer")
    return classifier

# Load the appropriate model
if use_transformer:
    try:
        classifier = load_transformer()
        st.success("✅ Transformer model loaded (GPU required for speed).")
        if torch.cuda.is_available():
            st.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ No GPU detected. Transformer will be slow on CPU. Consider using baseline model.")
    except Exception as e:
        st.error(f"Failed to load transformer model: {e}")
        st.info("Falling back to baseline model.")
        classifier = load_baseline()
        use_transformer = False
else:
    classifier = load_baseline()
    st.success("✅ Baseline model loaded (fast, CPU).")

# Text input
user_input = st.text_area("Ticket text:", height=150, placeholder="e.g., Someone hacked my account and changed my password...")

# Debug: see cleaned text (optional)
with st.expander("🔧 Debug: see cleaned text (what model actually sees)"):
    if user_input.strip():
        cleaned_example = clean_text(user_input, max_words=8, remove_greetings_flag=True, is_twitter=False)
        st.code(f"Original: {user_input}\n\nCleaned: {cleaned_example}")
    else:
        st.info("Enter some text to see cleaning result.")

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            category, confidence, method, needs_review = classifier.predict(user_input, return_details=True)

        # Display results
        st.success("Classification complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Category", category)
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")

        # Method and review flag
        st.markdown(f"**Method used:** `{method}`")
        if needs_review:
            st.warning("⚠️ This ticket needs human review (low confidence or suspicious keywords).")
        else:
            st.info("✅ Auto-classified (high confidence).")

        # Fraud specific warning
        if category == "Fraud":
            st.error("🚨 **FRAUD ALERT** – Please investigate immediately.")
            st.markdown("Recommended actions: freeze account, contact security team, notify user.")

        # Show raw probabilities for top 3 categories (optional)
        if st.checkbox("Show top predictions"):
            with st.spinner("Computing probabilities..."):
                # Get cleaned text again (or reuse)
                cleaned = clean_text(user_input, max_words=8, remove_greetings_flag=True, is_twitter=False)
                if not use_transformer:
                    X = classifier.vectorizer.transform([cleaned])
                    proba = classifier.classifier.predict_proba(X)[0]
                    class_names = classifier.classes
                else:
                    proba = classifier.transformer_model.predict_proba([cleaned])[0]
                    class_names = classifier.transformer_model.classes_

                top_indices = proba.argsort()[-3:][::-1]
                top_cats = [class_names[i] for i in top_indices]
                top_probs = [proba[i] for i in top_indices]
                st.markdown("**Top 3 predictions:**")
                for cat, p in zip(top_cats, top_probs):
                    st.write(f"- {cat}: {p:.2%}")

st.markdown("---")
st.caption("Powered by TF‑IDF + Logistic Regression (91.4% accuracy) or DistilBERT (97.8% accuracy) with smart fraud detection.")