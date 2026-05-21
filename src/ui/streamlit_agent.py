"""
Advanced Streamlit UI for Customer Support Teams – Full control sent to API.
Fast batch classification with progress bar.
Includes Department analytics tab.
"""
import streamlit as st
import requests
import time
import pandas as pd

st.set_page_config(page_title="Support Ticket Classifier - Advanced", page_icon="🎫", layout="wide")

DEPARTMENT_MAP = {
    "Account": "Identity & Access Management",
    "Billing": "Finance & Billing Team",
    "Fraud": "Security & Fraud Team",
    "General Inquiry": "Customer Support Team",
    "Technical": "Technical Support Team"
}

# Session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_text' not in st.session_state:
    st.session_state.last_text = ""
if 'api_healthy' not in st.session_state:
    try:
        r = requests.get("http://localhost:8000/api/v1/health", timeout=2)
        st.session_state.api_healthy = r.status_code == 200
    except:
        st.session_state.api_healthy = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Controls")
    model_type = st.selectbox(
        "Model",
        ["ensemble", "transformer", "baseline"],
        format_func=lambda x: {"ensemble": "Ensemble (best)", "transformer": "Transformer (GPU)", "baseline": "Baseline (CPU)"}[x]
    )
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.65, 0.05,
                                     help="Predictions below this will be flagged for review and may trigger LLM fallback.")
    use_llm_fallback = st.checkbox("Use LLM fallback (Groq)", value=True,
                                   help="If confidence < threshold, call Groq to re‑classify.")
    top_k = st.slider("Number of similar tickets (RAG)", 1, 10, 3)
    similarity_threshold = st.slider("Min similarity (RAG)", 0.0, 1.0, 0.3, 0.05)

    if st.session_state.api_healthy:
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline")

    if st.button("📥 Export History (CSV)") and st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "history.csv", "text/csv")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

st.title("🎫 Support Ticket Classifier")
st.markdown("**Advanced Interface**")
st.caption("Categories: Account | Billing | Fraud | General Inquiry | Technical")

def classify(text):
    url = "http://localhost:8000/api/v1/classify"
    payload = {
        "text": text,
        "model_type": model_type,
        "return_details": True,
        "confidence_threshold": confidence_threshold,
        "use_llm_fallback": use_llm_fallback
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def classify_batch_fast(texts):
    url = "http://localhost:8000/api/v1/classify/batch_fast"
    payload = {
        "texts": texts,
        "model_type": model_type,
        "return_details": True,
        "confidence_threshold": confidence_threshold,
        "use_llm_fallback": use_llm_fallback
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()['results']

def rag_explain(text):
    url = "http://localhost:8000/api/v1/rag/explain"
    payload = {
        "text": text,
        "model_type": model_type,
        "return_details": True,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📝 Classify", "🔍 RAG Explain", "📁 Batch Upload", "📊 Departments"])

# ---------- TAB 1: Classify ----------
with tab1:
    with st.form(key="classify_form"):
        user_input = st.text_area(
            "Ticket description",
            value=st.session_state.user_input,
            height=150,
            placeholder="Paste customer message here..."
        )
        col_examples = st.columns(5)
        examples = {
            "Account": "I cannot log in. Password reset not working.",
            "Billing": "I was double charged for my subscription.",
            "Fraud": "Someone stole my credit card and made purchases.",
            "Technical": "The app crashes every time I open it.",
            "General": "How do I change my notification settings?"
        }
        for col, (label, text) in zip(col_examples, examples.items()):
            if col.form_submit_button(label):
                st.session_state.user_input = text
                st.rerun()
        submitted = st.form_submit_button("Classify", type="primary", use_container_width=True)

    if submitted:
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            try:
                with st.spinner("Classifying..."):
                    start = time.time()
                    result = classify(user_input)
                    latency = (time.time() - start) * 1000

                category = result['category']
                confidence = result['confidence']
                needs_review = result.get('needs_review', False)
                model_used = result.get('model_used', model_type)
                department = DEPARTMENT_MAP.get(category, "Unknown")

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Category", category)
                col_b.metric("Confidence", f"{confidence*100:.1f}%")
                col_c.metric("Department", department)
                col_d.metric("Model Used", model_used)

                color = "#28a745" if confidence >= 0.8 else ("#ffc107" if confidence >= 0.6 else "#dc3545")
                st.markdown(f"""
                <div style="background-color:#e9ecef; border-radius:10px; height:12px; margin:10px 0;">
                    <div style="width:{confidence*100}%; background-color:{color}; height:12px; border-radius:10px;"></div>
                </div>
                """, unsafe_allow_html=True)

                if needs_review:
                    st.warning("⚠️ Low confidence – human review recommended.")
                else:
                    st.success("✅ High confidence – auto‑processed.")

                if category == "Fraud":
                    st.error("🚨 FRAUD ALERT – Investigate immediately.")

                entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "text": user_input[:100] + ("..." if len(user_input) > 100 else ""),
                    "category": category,
                    "confidence": confidence,
                    "department": department,
                    "model": model_used,
                    "needs_review": needs_review
                }
                st.session_state.history.insert(0, entry)
                st.session_state.history = st.session_state.history[:50]  # keep last 50 for display
                st.session_state.last_text = user_input

                with st.expander("View full ticket text"):
                    st.write(user_input)

            except Exception as e:
                st.error(f"Classification error: {e}")

# ---------- TAB 2: RAG Explanation ----------
with tab2:
    st.markdown("### 🔍 Get Explanation with Similar Tickets")
    if not st.session_state.last_text:
        st.info("First classify a ticket in the 'Classify' tab.")
    else:
        st.write(f"**Last classified ticket:** {st.session_state.last_text[:200]}...")
        if st.button("Get Explanation", type="primary", use_container_width=True):
            try:
                with st.spinner("Retrieving..."):
                    data = rag_explain(st.session_state.last_text)

                cls = data['classification']
                dept = DEPARTMENT_MAP.get(cls['category'], "Unknown")
                st.write(f"**Category:** {cls['category']}  |  **Confidence:** {cls['confidence']*100:.1f}%  |  **Department:** {dept}")

                st.subheader("📚 Similar Tickets")
                similar = data.get('similar_tickets', [])
                for i, t in enumerate(similar, 1):
                    with st.expander(f"Ticket #{i} | Similarity: {t['score']:.3f} | Category: {t['category']}"):
                        st.write(t['text_preview'])
                if not similar:
                    st.info("No similar tickets found.")

                st.subheader("💡 Explanation")
                st.write(data['explanation'])

            except Exception as e:
                st.error(f"RAG failed: {e}")

# ---------- TAB 3: Batch Upload ----------
with tab3:
    st.markdown("### 📁 Fast Batch Classification (CSV)")
    st.info("Upload a CSV file with a column named `text`. All tickets will be classified in batches with progress bar.")
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            total = len(df)
            st.write(f"Found {total} tickets. Preview (first 5 rows):")
            st.dataframe(df.head(5))
            if st.button("Classify All (Fast)", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                chunk_size = 50
                texts = df['text'].tolist()
                start_time = time.time()
                try:
                    for i in range(0, total, chunk_size):
                        chunk = texts[i:i+chunk_size]
                        status_text.text(f"Processing batch {i//chunk_size + 1}/{(total + chunk_size - 1)//chunk_size}...")
                        chunk_results = classify_batch_fast(chunk)
                        results.extend(chunk_results)
                        progress = min((i + len(chunk)) / total, 1.0)
                        progress_bar.progress(progress)
                    status_text.text("Done!")
                    progress_bar.progress(1.0)
                    for idx, res in enumerate(results):
                        res['text'] = texts[idx]
                        res['department'] = DEPARTMENT_MAP.get(res['category'], "Unknown")
                    result_df = pd.DataFrame(results)
                    elapsed = time.time() - start_time
                    st.success(f"Classified {total} tickets in {elapsed:.2f} seconds ({(elapsed/total)*1000:.0f} ms per ticket average)")
                    st.dataframe(result_df, use_container_width=True)
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "batch_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"Batch classification failed: {e}")

# ---------- TAB 4: Departments Analytics ----------
with tab4:
    st.markdown("### 📊 Department Performance & Tickets")
    if not st.session_state.history:
        st.info("No classifications yet. Classify some tickets to see department data.")
    else:
        # Build a DataFrame from history
        df_hist = pd.DataFrame(st.session_state.history)
        df_hist['department'] = df_hist['category'].map(DEPARTMENT_MAP)
        # Summary per department
        summary = df_hist.groupby('department').agg(
            ticket_count=('timestamp', 'count'),
            avg_confidence=('confidence', 'mean'),
            last_activity=('timestamp', 'max')
        ).reset_index().sort_values('ticket_count', ascending=False)
        
        st.subheader("📈 Department Summary")
        st.dataframe(summary, use_container_width=True)
        
        # For each department, show recent tickets
        st.subheader("📋 Recent Tickets by Department")
        for dept in summary['department'].unique():
            dept_df = df_hist[df_hist['department'] == dept].head(5)
            with st.expander(f"{dept} – last {len(dept_df)} tickets"):
                for _, row in dept_df.iterrows():
                    st.write(f"**{row['timestamp']}** | conf: {row['confidence']*100:.0f}% | model: {row.get('model', 'N/A')}")
                    st.caption(row['text'])
                    st.divider()

with st.sidebar.expander("📜 Recent Classifications", expanded=False):
    for entry in st.session_state.history[:10]:
        st.write(f"**{entry['timestamp']}** – {entry['category']} ({entry['confidence']*100:.0f}%)")
        st.caption(f"Dept: {entry.get('department', DEPARTMENT_MAP.get(entry['category'], 'Unknown'))} | {entry['text']}")
        st.divider()
    if not st.session_state.history:
        st.write("No classifications yet.")

st.markdown("---")
st.caption("Powered by TF‑IDF · DistilBERT · Ensemble · Groq LLM · Chroma")