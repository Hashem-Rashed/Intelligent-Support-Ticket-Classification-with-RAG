"""
Streamlit UI for ticket classification – supports baseline, transformer, and ensemble.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import torch
from src.api.classifier import ProductionTicketClassifier, EnsembleTicketClassifier

st.set_page_config(page_title="Ticket Classifier", page_icon="🎫", layout="centered")

st.title("🎫 Support Ticket Classification")
st.markdown("Enter a customer support message to classify its category (5 classes).")

model_option = st.radio(
    "Select Model:",
    ["Baseline (fast, CPU)", "Transformer (accurate, GPU)", "Ensemble (both, best accuracy)"],
    help="Ensemble averages predictions from both models for highest accuracy."
)
use_ensemble = model_option == "Ensemble (both, best accuracy)"
use_transformer = model_option == "Transformer (accurate, GPU)"

@st.cache_resource
def load_baseline():
    model_path = project_root / "models" / "saved" / "baseline"
    return ProductionTicketClassifier(model_path, model_type="baseline")

@st.cache_resource
def load_transformer():
    model_path = project_root / "models" / "saved" / "transformer"
    return ProductionTicketClassifier(model_path, model_type="transformer")

@st.cache_resource
def load_ensemble():
    baseline_dir = project_root / "models" / "saved" / "baseline"
    transformer_dir = project_root / "models" / "saved" / "transformer"
    return EnsembleTicketClassifier(baseline_dir, transformer_dir)

try:
    if use_ensemble:
        classifier = load_ensemble()
        st.success("✅ Ensemble model loaded (baseline + transformer).")
    elif use_transformer:
        classifier = load_transformer()
        st.success("✅ Transformer model loaded.")
        if torch.cuda.is_available():
            st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ No GPU – transformer will be slow.")
    else:
        classifier = load_baseline()
        st.success("✅ Baseline model loaded (fast, CPU).")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

user_input = st.text_area("Ticket text:", height=150,
                          placeholder="e.g., Someone hacked my account and changed my password...")

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            category, confidence, needs_review = classifier.predict(user_input, return_details=True)

        st.success("Classification complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Category", category)
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")

        if needs_review:
            st.warning("⚠️ Low confidence – please review manually.")
        else:
            st.info("✅ Auto‑classified with high confidence.")

        if category == "Fraud":
            st.error("🚨 **FRAUD ALERT** – Investigate immediately.")
            st.markdown("Recommended: freeze account, contact security, notify user.")

        with st.spinner("Computing top probabilities..."):
            proba = classifier.predict_proba(user_input)
            top_indices = proba.argsort()[-3:][::-1]
            top_cats = [classifier.classes[i] for i in top_indices]
            top_probs = [proba[i] for i in top_indices]
            st.markdown("**Top 3 predictions:**")
            for cat, p in zip(top_cats, top_probs):
                st.write(f"- {cat}: {p:.2%}")

st.markdown("---")
st.caption(f"Model: {model_option} | Categories: Account, Billing, Fraud, General Inquiry, Technical")