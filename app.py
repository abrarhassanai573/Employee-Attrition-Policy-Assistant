import streamlit as st
from common import apply_styling, render_header

st.set_page_config(page_title="InsightIQ | AI Business Intelligence", page_icon="📊", layout="wide")
apply_styling()
render_header(
    "📊 InsightIQ",
    "AI-Powered Business Intelligence — Predictions & Document Insights in One Platform"
)

st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card">
            <h4>🔮 Predictive Analytics</h4>
            <p>Upload any business CSV (employee, sales, customer data) and let AI find
            patterns, predict outcomes, and rank the factors that matter most.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <h4>💬 Document Q&A</h4>
            <p>Upload company PDFs (policies, reports, manuals) and ask questions in
            plain English or Roman Urdu — get answers sourced directly from your documents.</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <h4>🕓 Full History</h4>
            <p>Every question and answer is logged automatically, so teams can review
            past insights and decisions at any time.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Use the sidebar to get started — try **Predictions** for data analysis or **Ask AI** for document Q&A.")
