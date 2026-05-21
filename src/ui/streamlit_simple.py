"""
Simple Streamlit UI for Normal Users – one‑click classification.
Optimized for speed and ease of use.
"""
import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# Custom CSS for minimal styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #000000;
        color: white;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #333333;
    }
    .fraud-alert {
        background-color: #dc3545;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        height: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        width: 0%;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎫 Support Ticket Classifier")
st.markdown("Enter your support request and get an instant category.")

# Initialize session state for input text
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Input area - bind to session state
user_input = st.text_area(
    "Your message",
    value=st.session_state.user_input,
    height=150,
    placeholder="Example: I cannot log into my account..."
)
# Update session state when user types
st.session_state.user_input = user_input

# Example buttons
st.caption("Try an example:")
cols = st.columns(5)
examples = [
    ("🔐 Account", "I forgot my password and cannot log in."),
    ("💰 Billing", "I was charged twice for my subscription."),
    ("⚠️ Fraud", "Someone stole my credit card and made purchases."),
    ("🐛 Technical", "The app crashes when I open it."),
    ("❓ General", "Can you explain the main differences between your subscription plans and which one is best for small businesses?")
]
for col, (label, text) in zip(cols, examples):
    if col.button(label, use_container_width=True):
        st.session_state.user_input = text
        st.rerun()

# Classification button
if st.button("Classify", type="primary", use_container_width=True):
    if not st.session_state.user_input.strip():
        st.warning("Please enter a message.")
    else:
        try:
            with st.spinner("Analyzing..."):
                start = time.time()
                api_url = "http://localhost:8000/api/v1/classify"
                payload = {
                    "text": st.session_state.user_input,
                    "model_type": "ensemble",
                    "return_details": True
                }
                resp = requests.post(api_url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                latency = (time.time() - start) * 1000

                category = data['category']
                confidence = data['confidence']

                # Display results
                col1, col2 = st.columns(2)
                col1.metric("Category", category)
                col2.metric("Confidence", f"{confidence*100:.1f}%")

                # Confidence bar
                color = "#28a745" if confidence >= 0.8 else ("#ffc107" if confidence >= 0.6 else "#dc3545")
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence*100}%; background-color: {color};"></div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Response time: {latency:.0f} ms")

                if category == "Fraud":
                    st.markdown('<div class="fraud-alert">🚨 FRAUD ALERT – Our team will investigate immediately.</div>', unsafe_allow_html=True)
                else:
                    st.success("Your request has been routed to the appropriate department.")

        except requests.exceptions.ConnectionError:
            st.error("Service temporarily unavailable. Please try again later.")
        except Exception as e:
            st.error(f"Classification error: {e}")

st.markdown("---")
st.caption("Powered by AI | Categories: Account, Billing, Fraud, General Inquiry, Technical")