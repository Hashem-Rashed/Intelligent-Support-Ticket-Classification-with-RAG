"""
Advanced Streamlit UI for Support Ticket Classification with RAG
Black & white minimalist design, full functionality, batch upload with full text.
"""
import sys
import time
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import torch
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.classifier import ProductionTicketClassifier, EnsembleTicketClassifier
from src.preprocessing.text_processing import clean_text

# ============================================================
# Page config & custom CSS (Black & White)
# ============================================================
st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ============================================================
# Session state initialization
# ============================================================
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""
if 'last_text' not in st.session_state:
    st.session_state['last_text'] = ""
if 'last_category' not in st.session_state:
    st.session_state['last_category'] = ""

# ============================================================
# Sidebar – Configuration (Black background, white text)
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/ffffff/ticket.png", width=60)
    st.title("⚙️ Options")
    
    model_option = st.selectbox(
        "Model",
        ["Ensemble (best)", "Transformer (GPU)", "Baseline (CPU)"],
        help="Ensemble averages baseline and transformer."
    )
    use_ensemble = model_option == "Ensemble (best)"
    use_transformer = model_option == "Transformer (GPU)"
    
    st.divider()
    st.subheader("🎯 Classification")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.65, 0.05)
    use_llm_fallback = st.checkbox("Use LLM fallback (Groq)", value=True)
    
    st.divider()
    st.subheader("🔍 RAG Settings")
    top_k = st.slider("Similar tickets", 1, 10, 3)
    similarity_threshold = st.slider("Min similarity", 0.0, 1.0, 0.3, 0.05)
    
    st.divider()
    st.subheader("📊 Status")
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("No GPU (CPU only)")
    
    try:
        resp = requests.get("http://localhost:8000/api/v1/health", timeout=2)
        if resp.status_code == 200:
            st.success("RAG API: Online")
        else:
            st.error("RAG API: Error")
    except:
        st.error("RAG API: Offline (start `python -m src.api.main`)")

# ============================================================
# Main area title
# ============================================================
st.title("🎫 Ticket Classifier")
st.markdown("**TF‑IDF · DistilBERT · Ensemble · RAG**")
st.caption("Categories: Account | Billing | Fraud | General Inquiry | Technical")

# ============================================================
# Model loading (cached, with selected fallback option)
# ============================================================
@st.cache_resource
def load_baseline(llm):
    return ProductionTicketClassifier(project_root / "models" / "saved" / "baseline",
                                      model_type="baseline", use_llm_fallback=llm)

@st.cache_resource
def load_transformer(llm):
    return ProductionTicketClassifier(project_root / "models" / "saved" / "transformer",
                                      model_type="transformer", use_llm_fallback=llm)

@st.cache_resource
def load_ensemble(llm):
    return EnsembleTicketClassifier(project_root / "models" / "saved" / "baseline",
                                    project_root / "models" / "saved" / "transformer",
                                    use_llm_fallback=llm)

try:
    if use_ensemble:
        classifier = load_ensemble(use_llm_fallback)
    elif use_transformer:
        classifier = load_transformer(use_llm_fallback)
    else:
        classifier = load_baseline(use_llm_fallback)
    st.success("✅ Model ready")
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["📝 Classify", "🔍 RAG Explain", "📁 Batch"])

# ---------- TAB 1: Classify ----------
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area(
            "Ticket text",
            value=st.session_state['input_text'],
            height=150,
            placeholder="Example: Someone stole my credit card..."
        )
        st.session_state['input_text'] = user_input
        st.caption(f"{len(user_input)} characters | {len(user_input.split())} words")
    with col2:
        st.markdown("**Quick examples**")
        examples = [
            "Account: locked out, reset password",
            "Billing: double charged",
            "Fraud: unauthorized transaction",
            "Technical: app crashes",
            "General: how to use feature"
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state['input_text'] = ex
                st.rerun()

    if st.button("Classify", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            try:
                with st.spinner("Classifying..."):
                    start = time.time()
                    category, confidence, needs_review, model_used = classifier.predict(
                        user_input, return_details=True, allow_llm_fallback=use_llm_fallback
                    )
                    latency = (time.time() - start) * 1000
                    
                    # Display metrics
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Category", category)
                    col_b.metric("Confidence", f"{confidence:.2%}")
                    col_c.metric("Latency", f"{latency:.0f} ms")
                    
                    if needs_review:
                        st.warning("⚠️ Low confidence – review recommended.")
                    else:
                        st.success("✅ High confidence – auto‑processed.")
                    
                    if model_used == "llm":
                        st.info("🤖 LLM fallback used (Groq).")
                    if category == "Fraud":
                        st.error("🚨 FRAUD ALERT – investigate immediately.")
                    
                    # Store for RAG tab
                    st.session_state['last_text'] = user_input
                    st.session_state['last_category'] = category
                    
                    # Show full input in expander
                    with st.expander("View full ticket text"):
                        st.write(user_input)
            except Exception as e:
                st.error(f"Classification error: {e}")
                st.code(traceback.format_exc())

# ---------- TAB 2: RAG Explanation ----------
with tab2:
    st.markdown("### 🔍 Retrieve similar tickets + explanation")
    if not st.session_state['last_text']:
        st.info("Classify a ticket first (Tab 1).")
    else:
        if st.button("Get Explanation", type="primary", use_container_width=True):
            with st.spinner("Retrieving and generating..."):
                try:
                    api_url = "http://localhost:8000/api/v1/rag/explain"
                    model_type = "ensemble" if use_ensemble else ("transformer" if use_transformer else "baseline")
                    payload = {
                        "text": st.session_state['last_text'],
                        "model_type": model_type,
                        "return_details": True
                    }
                    response = requests.post(api_url, json=payload, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        similar = data.get("similar_tickets", [])
                        if similar:
                            st.subheader("📖 Similar tickets")
                            for i, t in enumerate(similar, 1):
                                with st.expander(f"#{i} | sim: {t['score']:.3f} | category: {t['category']}"):
                                    st.write(t['text_preview'])
                        else:
                            st.info("No similar tickets found.")
                        
                        st.subheader("💡 Explanation")
                        st.markdown(f"<div class='custom-card'>{data.get('explanation', 'N/A')}</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"RAG API error {response.status_code}")
                except Exception as e:
                    st.error(f"RAG failed: {e}")

# ---------- TAB 3: Batch Upload ----------
with tab3:
    st.markdown("### 📁 Batch classification (CSV)")
    st.info("CSV must contain a **`text`** column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error("Missing 'text' column.")
        else:
            if st.button("Classify all", use_container_width=True):
                progress = st.progress(0)
                results = []
                for i, row in df.iterrows():
                    try:
                        cat, conf, _, _ = classifier.predict(
                            row['text'], return_details=True, allow_llm_fallback=use_llm_fallback
                        )
                        results.append({
                            "text": row['text'],   # full text
                            "category": cat,
                            "confidence": conf
                        })
                    except Exception as e:
                        results.append({"text": row['text'][:100], "category": "ERROR", "confidence": 0.0})
                    progress.progress((i+1)/len(df))
                result_df = pd.DataFrame(results)
                st.subheader("Results")
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    column_config={
                        "text": st.column_config.TextColumn("Ticket Text", width="large"),
                        "category": "Category",
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.2f")
                    }
                )
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "classifications.csv", "text/csv")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("Powered by custom ML models + Chroma + Groq | Black & White UI")