import streamlit as st
import pandas as pd
from google import genai
from groq import Groq
from sentence_transformers import SentenceTransformer

from ml_utils import train_model
from rag_utils import extract_text_from_pdf, chunk_text, embed_texts, search_chunks, get_ai_response
from db_utils import init_db, log_chat, fetch_history

st.set_page_config(page_title="InsightIQ | AI Business Intelligence", page_icon="📊", layout="wide")

# ---------- Custom Styling (professional SaaS-style dashboard) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetricLabel"] { color: #6B7280; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #111827; font-weight: 700; }

    /* Headings */
    h1 { color: #111827; font-weight: 700; letter-spacing: -0.5px; }
    h2, h3 { color: #1F2937; font-weight: 600; }

    /* Buttons */
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.55em 1.6em;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] * { color: #F9FAFB !important; }
    section[data-testid="stSidebar"] .stRadio label { font-weight: 500; }

    /* Feature cards on the overview page */
    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        height: 100%;
    }
    .feature-card h4 { margin-top: 0; color: #4F46E5; }
    .feature-card p { color: #4B5563; font-size: 0.92em; line-height: 1.5; }

    /* Info/answer box */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

init_db()

# ---------- Branded Header ----------
st.markdown("""
    <div style="padding: 4px 0 18px 0;">
        <h1 style="margin-bottom:0;">📊 InsightIQ</h1>
        <p style="color:#6B7280; font-size:1.05em; margin-top:4px;">
            AI-Powered Business Intelligence — Predictions & Document Insights in One Platform
        </p>
    </div>
""", unsafe_allow_html=True)


# ---------- Cached resources (loaded once, reused across interactions) ----------
@st.cache_resource
def get_clients():
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return gemini_client, groq_client


@st.cache_resource
def get_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


gemini_client, groq_client = get_clients()
embed_model = get_embed_model()

# ---------- Session state (so uploaded docs stay available across clicks) ----------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# ---------- Sidebar Navigation ----------
st.sidebar.markdown("### 📊 InsightIQ")
st.sidebar.caption("AI Business Intelligence Assistant")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔮 Predictions", "💬 Ask AI", "🕓 History"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, scikit-learn, Gemini + Groq")


# ================= OVERVIEW PAGE =================
if page == "🏠 Overview":
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


# ================= PREDICTIONS PAGE =================
elif page == "🔮 Predictions":
    st.title("Predictive Analytics")
    st.write("Apna business CSV upload karo — AI patterns dhoondega aur predictions dega.")

    uploaded_csv = st.file_uploader("CSV file upload karo", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df.head(), use_container_width=True)

        target_col = st.selectbox(
            "Target column select karo (jo predict karna hai — jaise Attrition, Churn, etc.)",
            df.columns
        )

        if st.button("Train Model"):
            with st.spinner("Model train ho raha hai..."):
                model, accuracy, importances, feature_cols = train_model(df, target_col)

            col1, col2, col3 = st.columns(3)
            col1.metric("Model Accuracy", f"{accuracy:.2%}")
            col2.metric("Total Records", len(df))
            col3.metric("Features Used", len(feature_cols))

            st.subheader("Top 10 Factors")
            st.bar_chart(importances.head(10))


# ================= ASK AI PAGE =================
elif page == "💬 Ask AI":
    st.title("Document Q&A Assistant")
    st.write("Company documents (PDF) upload karo — phir unse natural language mein sawal poocho.")

    uploaded_pdfs = st.file_uploader(
        "PDF files upload karo (ek ya zyada)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_pdfs and st.button("Process Documents"):
        with st.spinner("Documents process ho rahe hain..."):
            all_chunks = []
            for pdf_file in uploaded_pdfs:
                text = extract_text_from_pdf(pdf_file)
                all_chunks.extend(chunk_text(text))

            st.session_state.chunks = all_chunks
            st.session_state.embeddings = embed_texts(embed_model, all_chunks)
        st.success(f"{len(all_chunks)} chunks indexed ho gaye!")

    if st.session_state.chunks:
        query = st.text_input("Apna sawal poocho:")
        if query and st.button("Poocho"):
            with st.spinner("Sochte hue..."):
                results = search_chunks(
                    query, st.session_state.chunks, st.session_state.embeddings,
                    embed_model, top_k=2
                )
                context = "\n\n".join([r[0] for r in results])

                prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer isn't in the context, say you don't have that information.

Context:
{context}

Question: {query}

Answer in the same language as the question (English or Roman Urdu)."""

                answer, provider = get_ai_response(prompt, gemini_client, groq_client)
                log_chat(query, answer, provider)

            st.write(f"**Answer** _(via {provider})_:")
            st.info(answer)
    else:
        st.info("Pehle upar documents upload aur process karo.")


# ================= HISTORY PAGE =================
else:
    st.title("Chat History")
    history = fetch_history()
    if history:
        hist_df = pd.DataFrame(history, columns=["Time", "Question", "Answer", "Provider"])
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("Abhi tak koi history nahi hai.")
