# streamlit_app.py

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Ensure Backend folder is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "Backend"))

# Import your backend logic
from Backend.llm_agent import generate_recommendation

# Load environment variables
load_dotenv(os.path.join("Backend", ".env"))

# ========== Streamlit UI ==========
st.set_page_config(page_title="📱 Phone Recommendation Agent", layout="wide")
st.title("📱 Intelligent Phone Recommendation Agent")

st.markdown(
    """
    This app fetches phone specifications, retrieves Reddit reviews, and uses an LLM 
    (Mistral) to generate professional recommendations.
    """
)

# --- Sidebar ---
st.sidebar.header("Configuration")
top_k = st.sidebar.slider("Number of top matches to retrieve", 1, 10, 3)
min_score = st.sidebar.slider("Minimum fuzzy match score", 50, 100, 60)

mistral_key = os.getenv("MISTRAL_API_KEY")
reddit_client = os.getenv("REDDIT_CLIENT_ID")
reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")

st.sidebar.write("**Environment Status:**")
st.sidebar.write(f"Mistral Key: {'✅ Found' if mistral_key else '❌ Missing'}")
st.sidebar.write(f"Reddit API: {'✅ Found' if reddit_client and reddit_secret else '❌ Missing'}")

# --- Input box ---
phone_name = st.text_input("🔍 Enter the phone name to analyze:", placeholder="e.g., iPhone 12 Mini")

if st.button("🚀 Generate Recommendation", use_container_width=True):
    if not phone_name.strip():
        st.warning("Please enter a phone name.")
        st.stop()

    with st.spinner(f"Fetching data and generating insights for **{phone_name}**..."):
        try:
            recommendation = generate_recommendation(phone_name, top_k=top_k, min_score=min_score)
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
            st.stop()

    st.success(f"✅ Recommendation generated for **{phone_name}**")
    st.markdown("---")
    st.subheader("📋 Result")

    # Display formatted recommendation
    if isinstance(recommendation, str):
        st.markdown(f"```text\n{recommendation}\n```")
    else:
        st.json(recommendation)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Mistral AI, and Reddit API.")
